import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Callable
import matplotlib.pyplot as plt


class MLP(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, noise):
        return self.network(noise)


def reptile(model, nb_iterations: int, sample_task: Callable, perform_k_training_steps: Callable, k=1, epsilon=0.1):
    for _ in tqdm(range(nb_iterations)):

        task = sample_task()
        phi_tilde = perform_k_training_steps(copy.deepcopy(model), task, k)

        # Update phi
        with torch.no_grad():
            for p, g in zip(model.parameters(), phi_tilde):
                p += epsilon * (g - p)


@torch.no_grad()
def sample_task():
    a = torch.rand(1).item() * 4.9 + .1  # Sample a in [0.1, 5.0]
    b = torch.rand(1).item() * 2 * np.pi  # Sample b in [0, 2pi]

    x = torch.linspace(-5, 5, 50)
    y = a * torch.sin(x + b)

    loss_fct = nn.MSELoss()

    return x, y, loss_fct


def perform_k_training_steps(model, task, k, batch_size=10):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    train_x, train_y, loss_fct = task
    for epoch in range(k * train_x.shape[0] // batch_size):
        ind = torch.randperm(train_x.shape[0])[:batch_size]
        x_batch = train_x[ind].unsqueeze(-1)
        target = train_y[ind].unsqueeze(-1)

        loss = loss_fct(model(x_batch), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Return the new weights after training
    return [p for p in model.parameters()]


if __name__ == "__main__":
    model = MLP()
    reptile(model, 30000, sample_task, perform_k_training_steps)

    with torch.no_grad():
        x = torch.linspace(-5, 5, 50).unsqueeze(-1)
        y_pred_before = model(x)
        new_task = sample_task()
        true_x, true_y, _ = new_task

    # Perform 32 training steps on the new task
    perform_k_training_steps(model, new_task, 32)
    y_pred_after = model(x)

    plt.plot(x.numpy(), y_pred_before.numpy(), label='Before')
    plt.plot(x.numpy(), y_pred_after.data.numpy(), label='After')
    plt.plot(true_x.numpy(), true_y.numpy(), label='True')
    plt.legend(fontsize=11)
    plt.savefig('Imgs/Demonstration_of_reptile.png')
