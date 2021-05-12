import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
from typing import Tuple

torch.manual_seed(8001)

# load (and normalize) mnist dataset
(trainX, trainY), (testX, testy) = load_data()
trainX = (np.float32(trainX) - 127.5) / 127.5
# Training and validation set of 50 samples
valX = torch.tensor(trainX[:50].reshape(50, -1))
valY = torch.tensor(trainY[:50], dtype=torch.long)
trainX = torch.tensor(trainX[-50:].reshape(50, -1))
trainY = torch.tensor(trainY[-50:], dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class WeightDecay(nn.Module):
    def __init__(self, model, device):
        super(WeightDecay, self).__init__()
        self.positive_constraint = torch.nn.Softplus()

        # Set a weight decay for each parameter in the input model
        idx = 0
        self.parameter_dict = {}
        for m in model.parameters():
            self.parameter_dict[str(idx)] = torch.nn.Parameter(torch.rand(m.shape, device=device))
            idx += 1
        self.params = torch.nn.ParameterDict(self.parameter_dict)

    def forward(self, model):
        regularization = 0.
        for coefficients, weights in zip(self.parameters(), model.parameters()):
            regularization += (self.positive_constraint(coefficients) * weights ** 2).sum()
        return regularization


def compute_loss(x, label, model, weight_decay):
    predictions = model(x)
    return torch.nn.CrossEntropyLoss()(predictions, label) + weight_decay(model)


@torch.no_grad()
def compute_pred(x, label, model):
    predictions = model(x)
    predictions = predictions.argmax(1)
    return ((predictions == label).sum() / predictions.shape[0]).item()


def gradient_based_ho(nb_epochs=int(10e3), N=50, device='cpu', batch_size=32):
    model = MLP(input_dim=784, output_dim=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    meta_model = WeightDecay(model, device)
    meta_optimizer = torch.optim.RMSprop(meta_model.parameters(), lr=.1)

    accuracies = {'training': [], 'validation': []}

    for epoch in tqdm(range(nb_epochs)):  # Outer optimization loop
        for k in range(N):  # Inner optimization loop

            batch_idx = torch.randperm(trainX.shape[0])[:batch_size]
            training_loss = compute_loss(trainX[batch_idx].to(device), trainY[batch_idx].to(device), model, meta_model)

            # Update w
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

        training_loss = compute_loss(trainX.to(device), trainY.to(device), model, meta_model)
        validation_loss = compute_loss(valX.to(device), valY.to(device), model, meta_model)

        accuracies['training'].append(compute_pred(trainX.to(device), trainY.to(device), model))
        accuracies['validation'].append(compute_pred(valX.to(device), valY.to(device), model))

        hyper_grads = hypergradient(validation_loss, training_loss, meta_model.parameters, model.parameters)

        meta_optimizer.zero_grad()
        for p, g in zip(meta_model.parameters(), hyper_grads):
            p.grad = g
        meta_optimizer.step()

    return accuracies


def hypergradient(validation_loss: torch.Tensor, training_loss: torch.Tensor, lambda_: torch.Generator,
                  w: torch.Generator):
    # List[torch.Tensor]. v1[i].shape = w[i].shape
    v1 = torch.autograd.grad(validation_loss, w(), retain_graph=True)

    d_train_d_w = torch.autograd.grad(training_loss, w(), create_graph=True)
    # List[torch.Tensor]. v2[i].shape = w[i].shape
    v2 = approxInverseHVP(v1, d_train_d_w, w)

    # List[torch.Tensor]. v3[i].shape = lambda_[i].shape
    v3 = torch.autograd.grad(d_train_d_w, lambda_(), grad_outputs=v2, retain_graph=True, )

    d_val_d_lambda = torch.autograd.grad(validation_loss, lambda_())
    return [d - v for d, v in zip(d_val_d_lambda, v3)]


def approxInverseHVP(v: Tuple[torch.Tensor], f: Tuple[torch.Tensor], w: torch.Generator, i=3, alpha=.1):
    p = v

    for j in range(i):
        grad = torch.autograd.grad(f, w(), grad_outputs=v, retain_graph=True)
        v = [v_ - alpha * g for v_, g in zip(v, grad)]
        p = [p_ + v_ for p_, v_ in zip(p, v)]  # p += v (Typo in the arxiv version of the paper)

    return p


if __name__ == "__main__":
    accuracies = gradient_based_ho(nb_epochs=int(201), N=50, device='cuda')

    plt.figure(figsize=(8, 6))
    plt.plot(1 - np.array(accuracies['training'])[::5], alpha=0.8, linestyle='--', linewidth=2.5, label='Training')
    plt.plot(1 - np.array(accuracies['validation'])[::5], alpha=0.8, c='C0', linewidth=2.5, label='Validation')
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Classification Error', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('Imgs/Classification_error.pdf')
