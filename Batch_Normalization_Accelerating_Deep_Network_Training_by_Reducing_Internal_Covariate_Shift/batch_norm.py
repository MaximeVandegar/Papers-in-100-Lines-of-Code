import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets


class BatchNorm(nn.Module):
    def __init__(self, n_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.register_buffer("population_mean", torch.zeros(n_features))
        self.register_buffer("population_var", torch.ones(n_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            # Moving averages of the population statistics, with the unbiased variance (Sec. 3.1).
            with torch.no_grad():
                batch_size = len(x)
                self.population_mean.lerp_(mean, self.momentum)
                self.population_var.lerp_(var * batch_size / (batch_size - 1), self.momentum)
        else:
            mean = self.population_mean
            var = self.population_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class MLP(nn.Module):
    def __init__(self, use_batch_norm, init_std=0.06):
        super().__init__()
        widths = [784, 100, 100, 100]
        # With batch normalization the bias is dropped: the beta parameter replaces it (Sec. 3.2).
        self.hidden = nn.ModuleList(nn.Linear(n_in, n_out, bias=not use_batch_norm)
                                    for n_in, n_out in zip(widths, widths[1:]))
        self.norms = nn.ModuleList(BatchNorm(100) if use_batch_norm else nn.Identity()
                                   for _ in self.hidden)
        self.out = nn.Linear(100, 10)
        for layer in [*self.hidden, self.out]:
            nn.init.normal_(layer.weight, std=init_std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for linear, norm in zip(self.hidden, self.norms):
            sigmoid_input = norm(linear(x))
            x = torch.sigmoid(sigmoid_input)
        return self.out(x), sigmoid_input


def load_binarized_mnist(device):
    def to_tensors(split):
        images = (split.data > 127).float().flatten(1)  # 28x28 binary images (Sec. 4.1)
        return images.to(device), split.targets.to(device)

    train_x, train_y = to_tensors(datasets.MNIST(".", train=True, download=True))
    test_x, test_y = to_tensors(datasets.MNIST(".", train=False, download=True))
    return train_x, train_y, test_x, test_y


def train(model, optimizer, data, n_steps=50_000, batch_size=60, log_every=250, unit=0):
    train_x, train_y, test_x, test_y = data
    history = {"step": [], "accuracy": [], "percentiles": []}

    for step in tqdm(range(1, n_steps + 1)):
        batch_idx = torch.randint(len(train_x), (batch_size,), device=train_x.device)
        model.train()
        loss = F.cross_entropy(model(train_x[batch_idx])[0], train_y[batch_idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            model.eval()
            with torch.no_grad():
                logits = model(test_x)[0]
                # Fig. 1 (b, c) uses the training network (Alg. 2): batch norm in train mode.
                # On a copy, so this pass does not change the running averages.
                sigmoid_input = copy.deepcopy(model).train()(test_x)[1]
            history["step"].append(step)
            history["accuracy"].append((logits.argmax(1) == test_y).float().mean().item())
            quantiles = torch.tensor([0.15, 0.5, 0.85], device=test_x.device)
            percentiles = torch.quantile(sigmoid_input[:, unit], quantiles)  # Fig. 1 (b, c)
            history["percentiles"].append(percentiles.tolist())
    return history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_binarized_mnist(device)
    fig, (ax_accuracy, ax_without_bn, ax_with_bn) = plt.subplots(
        1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})

    for use_batch_norm, ax_percentiles in [(False, ax_without_bn), (True, ax_with_bn)]:
        torch.manual_seed(0)
        model = MLP(use_batch_norm).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        history = train(model, optimizer, data)

        # (a) Test accuracy during training.
        ax_accuracy.plot(history["step"], history["accuracy"], "b-" if use_batch_norm else "k--",
                         label="With BN" if use_batch_norm else "Without BN")
        # (b, c) 15th, 50th and 85th percentiles of one last-layer sigmoid input during training.
        ax_percentiles.plot(history["step"], history["percentiles"], "k-", linewidth=0.8)
        ax_percentiles.set(xlim=(0, 50_000), xticks=[], ylim=(-2.5, 2.5), yticks=[-2, 0, 2],
                           xlabel="(c) With BN" if use_batch_norm else "(b) Without BN")

    ax_accuracy.set(xlim=(0, 50_000), xticks=[10_000, 20_000, 30_000, 40_000, 50_000],
                    xticklabels=["10K", "20K", "30K", "40K", "50K"],
                    ylim=(0.7, 1), yticks=[0.7, 0.8, 0.9, 1], xlabel="(a)")
    ax_accuracy.legend(loc="lower right", fancybox=False, edgecolor="black")

    plt.tight_layout()
    plt.savefig("Imgs/batch_norm.png", dpi=200, bbox_inches="tight")
