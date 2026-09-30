import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, residual):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = residual
        self.stride = stride
        self.extra_channels = out_channels - in_channels

    def shortcut(self, x):  # identity, or subsampled and zero-padded when downsampling (Sec. 3.3)
        return F.pad(x[:, :, ::self.stride, ::self.stride], (0, 0, 0, 0, 0, self.extra_channels))

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        if self.residual:
            out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, depth, residual):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        blocks_per_stage = (depth - 2) // 6  # Sec. 4.2
        blocks = []
        in_channels = 16
        for out_channels in (16, 32, 64):
            for _ in range(blocks_per_stage):
                stride = 1 if out_channels == in_channels else 2
                blocks.append(Block(in_channels, out_channels, stride, residual))
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return self.fc(self.blocks(x).mean((2, 3)))


def load_cifar10(device):
    def to_tensors(split):
        images = torch.tensor(split.data).permute(0, 3, 1, 2).float()
        return images.to(device), torch.tensor(split.targets, device=device)

    train_x, train_y = to_tensors(datasets.CIFAR10(".", train=True, download=True))
    test_x, test_y = to_tensors(datasets.CIFAR10(".", train=False, download=True))
    per_pixel_mean = train_x.mean(0)
    return train_x - per_pixel_mean, train_y, test_x - per_pixel_mean, test_y


def augment(images):
    """Pad 4 pixels per side, crop a random 32x32 window, flip half horizontally (Sec. 4.2)."""
    n = len(images)
    padded = F.pad(images, (4, 4, 4, 4)).permute(0, 2, 3, 1)
    offsets = torch.arange(32, device=images.device)
    rows = torch.randint(0, 9, (n, 1, 1), device=images.device) + offsets.view(1, 32, 1)
    cols = torch.randint(0, 9, (n, 1, 1), device=images.device) + offsets.view(1, 1, 32)
    crops = padded[torch.arange(n, device=images.device).view(n, 1, 1), rows, cols]
    crops = crops.permute(0, 3, 1, 2)
    flipped = torch.rand(n, 1, 1, 1, device=images.device) < 0.5
    return torch.where(flipped, crops.flip(3), crops)


@torch.no_grad()
def error_rate(model, x, y):
    model.eval()
    wrong = sum((model(xb).argmax(1) != yb).sum() for xb, yb in zip(x.split(1000), y.split(1000)))
    return wrong.item() / len(x)


def train(model, data, warm_up, n_iterations=64_000, batch_size=128, lr=0.1, momentum=0.9,
          weight_decay=1e-4, milestones=(32_000, 48_000), eval_every=400):
    train_x, train_y, test_x, test_y = data
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01 if warm_up else lr,
                                momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    history = {"iteration": [], "train": [], "test": []}

    train_error = 0
    for iteration in tqdm(range(1, n_iterations + 1)):
        idx = torch.randint(len(train_x), (batch_size,), device=train_x.device)
        model.train()
        logits = model(augment(train_x[idx]))
        loss = F.cross_entropy(logits, train_y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_error = (logits.argmax(1) != train_y[idx]).float().mean()
        train_error += batch_error
        if warm_up and batch_error < 0.8:
            warm_up = False
            optimizer.param_groups[0]["lr"] = lr
        if iteration % eval_every == 0:
            history["iteration"].append(iteration)
            history["train"].append(train_error.item() / eval_every)
            history["test"].append(error_rate(model, test_x, test_y))
            train_error = 0
    return history


def plot(histories, save_path):
    colors = {20: "#b8b338", 32: "#5ee6ee", 44: "#66ee4d", 56: "#ee3326", 110: "black"}
    with plt.rc_context({"font.family": "serif"}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for (residual, depth), history in histories.items():
            ax = axes[int(residual)]
            iterations = [i / 1e4 for i in history["iteration"]]
            ax.plot(iterations, [100 * e for e in history["train"]], "-.", color=colors[depth],
                    linewidth=1)
            ax.plot(iterations, [100 * e for e in history["test"]], color=colors[depth],
                    linewidth=2.5, label=f"{'ResNet' if residual else 'plain'}-{depth}")
        for ax, legend_position in zip(axes, ("lower left", "upper right")):
            ax.set(xlabel="iter. (1e4)", ylabel="error (%)", xlim=(0, 6.4), ylim=(0, 20),
                   xticks=range(7), yticks=[0, 5, 10, 20])
            ax.grid(axis="y", linestyle="--", color="0.4")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc=legend_position, fancybox=False, edgecolor="black")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close(fig)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plain_depths = [20, 32, 44, 56]
    resnet_depths = [20, 32, 44, 56, 110]

    data = load_cifar10(device)
    histories = {}
    for residual, depths in [(False, plain_depths), (True, resnet_depths)]:
        for depth in depths:
            torch.manual_seed(0)
            model = ResNet(depth, residual).to(device)
            histories[(residual, depth)] = train(model, data, warm_up=depth == 110)
    plot(histories, "Imgs/resnet.png")
