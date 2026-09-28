import math
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets


class Affine(nn.Module):
    def __init__(self, features, train_gain=True):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features), requires_grad=train_gain)
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        broadcastable_shape = (1, -1) + (1,) * (x.dim() - 2)
        return x * self.gain.view(broadcastable_shape) + self.bias.view(broadcastable_shape)


class RMSNorm(Affine):
    def __init__(self, features, p=1.0, eps=1e-6):
        super().__init__(features)
        self.p = p
        self.eps = eps

    def forward(self, x):
        elements = x.flatten(2) if x.dim() == 4 else x[:, None]
        kept = elements[..., :math.ceil(self.p * elements.shape[-1])]  # pRMSNorm
        rms = kept.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return super().forward((elements / rms).view_as(x))


class LayerNorm(RMSNorm):
    def forward(self, x):
        dims = (2, 3) if x.dim() == 4 else 1  # 3D (C, H, W) feature maps or 1D logits
        return super().forward(x - x.mean(dims, keepdim=True))


def normalization(variant, features, n_dims):
    if variant == "Baseline":
        # The gain is initialized, never trained [WeightNorm paper, Salimans & Kingma, Sec. 3].
        return Affine(features, train_gain=False)
    if variant == "BatchNorm":
        return (nn.BatchNorm2d if n_dims == 4 else nn.BatchNorm1d)(features, eps=1e-6)
    if variant == "LayerNorm":
        return LayerNorm(features)
    if variant == "RMSNorm":
        return RMSNorm(features)
    if variant == "pRMSNorm":
        return RMSNorm(features, p=0.125)
    return nn.Identity()


class ConvPoolCNNC(nn.Module):
    def __init__(self, variant, noise_std=0.15):
        super().__init__()
        self.noise_std = noise_std
        use_bias = variant == "WeightNorm"  # the rest have the bias in their normalization layer

        def block(layer, features, n_dims, activation=nn.LeakyReLU(0.1)):
            if variant in ("BatchNorm", "WeightNorm"):
                nn.init.normal_(layer.weight, std=0.05)  # WeightNorm paper, Sec. 3
            else:
                nn.init.xavier_uniform_(layer.weight)
            if variant == "WeightNorm":
                layer = nn.utils.parametrizations.weight_norm(layer)
            return nn.Sequential(layer, normalization(variant, features, n_dims), activation)

        def conv(in_channels, out_channels, kernel, padding):
            layer = nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=use_bias)
            return block(layer, out_channels, n_dims=4)

        self.net = nn.Sequential(
            conv(3, 96, 3, 1), conv(96, 96, 3, 1), conv(96, 96, 3, 1),
            nn.MaxPool2d(2), nn.Dropout(0.5),
            conv(96, 192, 3, 1), conv(192, 192, 3, 1), conv(192, 192, 3, 1),
            nn.MaxPool2d(2), nn.Dropout(0.5),
            conv(192, 192, 3, 0), conv(192, 192, 1, 0), conv(192, 192, 1, 0),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            block(nn.Linear(192, 10, bias=use_bias), 10, n_dims=2,
                  activation=nn.Identity()))

    def forward(self, x):
        if self.training:
            x = x + self.noise_std * torch.randn_like(x)
        return self.net(x)


@torch.no_grad()
def data_dependent_init(model, x):
    """Set the gain and bias after each layer so that its pre-activations over this batch have
    zero mean and unit variance [WeightNorm paper, Salimans & Kingma, Sec. 3]."""
    for stage in model.net:
        if isinstance(stage, nn.Sequential):
            layer = stage[0]
            affine = stage[1]
            if isinstance(affine, Affine):
                gain = affine.gain
                bias = affine.bias
            elif isinstance(affine, nn.Identity):  # WeightNorm keeps its gain in the weight
                gain = layer.parametrizations.weight.original0
                bias = layer.bias
            else:
                raise ValueError(f"no gain to initialize in {stage}")
            bias.zero_()
            pre_activation = layer(x)
            dims = (0, 2, 3) if pre_activation.dim() == 4 else 0
            mean = pre_activation.mean(dims)
            std = pre_activation.std(dims)
            gain /= std.view(gain.shape)
            bias.copy_(-mean / std)
        x = stage(x)


def load_cifar10(device, zca_eps=1e-5):
    def to_tensors(split):
        images = (torch.tensor(split.data).permute(0, 3, 1, 2).float() - 127.5) / 128
        return images.to(device), torch.tensor(split.targets, device=device)

    train_x, train_y = to_tensors(datasets.CIFAR10(".", train=True, download=True))
    test_x, test_y = to_tensors(datasets.CIFAR10(".", train=False, download=True))
    mean = train_x.flatten(1).mean(0)
    centred = train_x.flatten(1) - mean
    eigenvalues, eigenvectors = torch.linalg.eigh(centred.T @ centred / len(centred))
    zca = (eigenvectors / (eigenvalues + zca_eps).sqrt()) @ eigenvectors.T

    def whiten(images):
        return ((images.flatten(1) - mean) @ zca).view_as(images)

    return whiten(train_x), train_y, whiten(test_x), test_y


def adam_settings(epoch, lr, n_epochs):
    """The WeightNorm paper [Salimans & Kingma, Sec. 5.1] drops beta1 to 0.5 and decays the rate
    linearly to zero over the second half of training."""
    half = n_epochs // 2
    if epoch < half:
        return lr, (0.9, 0.999)
    return lr * (n_epochs - epoch) / half, (0.5, 0.999)


@torch.no_grad()
def error_rate(model, x, y):
    model.eval()
    wrong = sum((model(xb).argmax(1) != yb).sum() for xb, yb in zip(x.split(1000), y.split(1000)))
    return wrong.item() / len(x)


def train(variant, data, n_epochs, batch_size, device):
    train_x, train_y, test_x, test_y = data
    torch.manual_seed(0)
    model = ConvPoolCNNC(variant).to(device)
    if variant in ("Baseline", "WeightNorm"):  # the others learn their scale during training
        data_dependent_init(model, train_x[:500])
    lr = 3e-4 if variant == "Baseline" else 3e-3  # tuned per variant, WeightNorm paper Sec. 5.1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)

    history = {"train": [], "test": []}
    for epoch in (bar := tqdm(range(n_epochs), desc=f"{variant:10s}")):
        for group in optimizer.param_groups:
            group["lr"], group["betas"] = adam_settings(epoch, lr, n_epochs)
        model.train()
        wrong = 0
        for idx in torch.randperm(len(train_x), device=device).split(batch_size):
            logits = model(train_x[idx])
            loss = F.cross_entropy(logits, train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            wrong += (logits.argmax(1) != train_y[idx]).sum()
        history["train"].append(wrong.item() / len(train_x))
        history["test"].append(error_rate(model, test_x, test_y))
        bar.set_postfix(train=f"{history['train'][-1]:.3f}", test=f"{history['test'][-1]:.3f}")
    return history


def plot(histories, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    panels = [("train", "Training error", (0, 0.08)), ("test", "Test error", (0.05, 0.2))]
    for ax, (split, title, ylim) in zip(axes, panels):
        for variant, history in histories.items():
            ax.plot(history[split], label=variant)
        ax.set(xlabel="Training epochs", ylabel="Error rate", title=title, ylim=ylim)
        ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = 200
    batch_size = 100
    variants = ["Baseline", "BatchNorm", "LayerNorm", "WeightNorm", "RMSNorm", "pRMSNorm"]

    data = load_cifar10(device)
    histories = {variant: train(variant, data, n_epochs, batch_size, device)
                 for variant in variants}
    plot(histories, "Imgs/rmsnorm.png")
