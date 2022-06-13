import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class MLP(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=40, output_dim=1):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Task:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sample(self, K):
        x = torch.rand((K, 1)) * 10 - 5  # Sample x in [-5, 5]
        y = self.a * torch.sin(x + self.b)
        loss_fct = nn.MSELoss()

        return x, y, loss_fct


@torch.no_grad()
def sample_task():
    a = torch.rand(1).item() * 4.9 + .1  # Sample the amplitude in [0.1, 5.0]
    b = torch.rand(1).item() * np.pi  # Sample the phase in [0, pi]

    return Task(a, b)


def perform_k_training_steps(model, task, batch_size, inner_training_steps, alpha, device='cpu'):
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)

    for epoch in range(inner_training_steps):
        x_batch, target, loss_fct = task.sample(batch_size)

        loss = loss_fct(model(x_batch.to(device)), target.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Return the new weights after training
    return model.state_dict()


def maml(model, meta_optimizer, inner_training_steps, nb_epochs, batch_size_K, alpha, nb_tasks=10, device='cpu'):
    """
    Algorithm from https://arxiv.org/pdf/1703.03400v3.pdf (MAML for Few-Shot Supervised Learning)
    """
    training_loss = []

    for epoch in tqdm(range(nb_epochs)):  # Line 2 in the pseudocode

        theta_i_prime = []
        D_i_prime = []

        # Sample batch of tasks
        tasks = [sample_task() for _ in range(nb_tasks)]  # Line 3 in the pseudocode
        for task in tasks:
            # Compute adapted parameters with gradient descent (line 7 in the pseudocode)
            model_copy = MLP()
            model_copy.load_state_dict(model.state_dict())
            theta_i_prime.append(
                perform_k_training_steps(model_copy.to(device), task, batch_size_K, inner_training_steps, alpha,
                                         device=device))
            # Sample data points Di' for the meta-update (line 8 in the pseudocode)
            x, y, loss_fct = task.sample(25)
            D_i_prime.append((x, y, loss_fct))

        # Meta update
        meta_optimizer.zero_grad()
        gradients = [torch.zeros_like(p) for p in model.parameters()]  # Initialize gradients

        batch_training_loss = []
        for i in range(nb_tasks):
            x, y, loss_fct = D_i_prime[i]

            f_theta_prime = MLP().to(device)
            f_theta_prime.load_state_dict(theta_i_prime[i])

            # Compute \nabla_theta L(f_theta_i_prime) for task ti
            loss = loss_fct(f_theta_prime(x.to(device)), y.to(device))
            loss.backward()
            batch_training_loss.append(loss.item())

            # Store the gradients in the variable gradients
            for g, p in zip(gradients, f_theta_prime.parameters()):
                g += p.grad

        # Set gradients for the meta optimizer
        for p, g in zip(model.parameters(), gradients):
            p.grad = g

        meta_optimizer.step()  # Line 10 in the pseudocode

        training_loss.append(np.mean(batch_training_loss))

    return training_loss


if __name__ == 'main':
    device = 'cpu'
    model = MLP().to(device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    maml(model, meta_optimizer, 1, 70000, 10, 1e-3, device=device, nb_tasks=10)

    plt.title('MAML, K=10')
    x = torch.linspace(-5, 5, 50)
    y = model(x[..., None])
    plt.plot(x.data.numpy(), y.data.numpy(), c='lightgreen', linestyle='--', linewidth=2.2, label='pre-update')
    # New task
    task = sample_task()
    ground_truth_y = task.a * torch.sin(x + task.b)
    plt.plot(x.data.numpy(), ground_truth_y.data.numpy(), c='red', label='ground truth')
    # Fine-tuning; 1 gradient steps
    perform_k_training_steps(model, task, 1, 10, 1e-3, device='cpu')
    # After 1 gradient steps
    y = model(x[..., None])
    plt.plot(x.data.numpy(), y.data.numpy(), c='darkgreen', linestyle='--', linewidth=2.2, label='1 grad step')
    plt.legend()
    plt.savefig('Imgs/maml.png')
    plt.close()
