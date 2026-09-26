import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms


class HyperNetwork(nn.Module):
    """The two-layer hypernetwork of paper Eq. 2"""

    def __init__(self, embedding_size=64, block_size=16, filter_size=3):
        super().__init__()
        self.embedding_size = embedding_size
        self.block_size = block_size
        self.filter_size = filter_size
        self.to_channel_vectors = nn.Linear(embedding_size, block_size * embedding_size)
        self.to_kernel_slice = nn.Linear(embedding_size, block_size * filter_size * filter_size)

        for layer in (self.to_channel_vectors, self.to_kernel_slice):
            nn.init.trunc_normal_(layer.weight, std=0.01, a=-0.02, b=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, embeddings):
        grid = embeddings.shape[:-1]  # (out_ch // block_size, in_ch // block_size)
        vectors = self.to_channel_vectors(embeddings).view(*grid, self.block_size,
                                                           self.embedding_size)
        return self.to_kernel_slice(vectors).view(*grid, self.block_size, self.block_size,
                                                  self.filter_size, self.filter_size)


class HyperConv3x3(nn.Module):
    """A 3x3 convolution that stores embeddings instead of a kernel"""

    def __init__(self, in_ch, out_ch, stride, hyper):
        super().__init__()
        self.hyper = hyper
        self.stride = stride
        blocks_out = out_ch // hyper.block_size
        blocks_in = in_ch // hyper.block_size
        self.embeddings = nn.Parameter(torch.randn(blocks_out, blocks_in, hyper.embedding_size))

    def kernel(self):
        """The layer's kernel is a grid of basic kernels side by side (paper Eq. 3).  The
        permutation pairs each grid axis with the channels inside its block."""
        tiles = self.hyper(self.embeddings)
        out_ch = tiles.shape[0] * self.hyper.block_size
        in_ch = tiles.shape[1] * self.hyper.block_size
        return tiles.permute(0, 3, 1, 2, 4, 5).reshape(out_ch, in_ch, 3, 3)

    def forward(self, x):
        return F.conv2d(x, self.kernel(), stride=self.stride, padding=1)


def conv3x3(in_ch, out_ch, stride, hyper):
    if hyper is None:
        return nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
    return HyperConv3x3(in_ch, out_ch, stride, hyper)


class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride, hyper):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = conv3x3(in_ch, out_ch, stride, hyper)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, 1, hyper)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
                         if in_ch != out_ch or stride != 1 else None)

    def forward(self, x):
        activated = F.relu(self.bn1(x))
        residual = self.conv2(F.relu(self.bn2(self.conv1(activated))))
        return residual + (self.shortcut(activated) if self.shortcut is not None else x)


class WideResNet(nn.Module):
    """WRN 40-2 (paper Table 1 with N = 6, k = 2)"""

    def __init__(self, use_hypernetwork, blocks_per_group=6, widths=(32, 64, 128)):
        super().__init__()
        self.hyper = HyperNetwork() if use_hypernetwork else None
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        blocks = []
        in_ch = 16
        for group, width in enumerate(widths):
            for index in range(blocks_per_group):
                downsample = group > 0 and index == 0
                blocks.append(ResidualBlock(in_ch, width, 2 if downsample else 1, self.hyper))
                in_ch = width
        self.blocks = nn.Sequential(*blocks)
        self.final_bn = nn.BatchNorm2d(widths[-1])
        self.classifier = nn.Linear(widths[-1], 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        features = F.relu(self.final_bn(self.blocks(self.stem(x))))
        return self.classifier(F.adaptive_avg_pool2d(features, 1).flatten(1))


def sum_of_squares(tensors):
    return sum(tensor.pow(2).sum() for tensor in tensors)


def l2_penalty(model, ordinary_decay, generated_decay):
    stored_kernels = [m.weight for m in model.modules() if isinstance(m, nn.Conv2d)]
    generated_kernels = [m.kernel() for m in model.modules() if isinstance(m, HyperConv3x3)]
    return 0.5 * (ordinary_decay * sum_of_squares(stored_kernels)
                  + generated_decay * sum_of_squares(generated_kernels))


def endless_batches(loader):
    while True:
        yield from loader


def learning_rate(lr_schedule, step):
    for until, rate in lr_schedule:
        if step < until:
            return rate
    return lr_schedule[-1][1]


@torch.no_grad()
def error_rate(model, loader, device):
    model.eval()
    wrong = 0
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(1)
        wrong += (predictions != labels.to(device)).sum().item()
    model.train()
    return 100.0 * wrong / len(loader.dataset)


def cifar10_loaders(batch_size, val_size):
    """Paper A.3.2."""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=2),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), normalize])
    augmented = datasets.CIFAR10("data", train=True, download=True, transform=train_transform)
    unaugmented = datasets.CIFAR10("data", train=True, download=True, transform=eval_transform)
    test_set = datasets.CIFAR10("data", train=False, download=True, transform=eval_transform)
    shuffled = torch.randperm(len(augmented), generator=torch.Generator().manual_seed(0)).tolist()
    train_set = torch.utils.data.Subset(augmented, shuffled[:-val_size])
    val_set = torch.utils.data.Subset(unaugmented, shuffled[-val_size:])
    return (torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, drop_last=True,
                                        num_workers=4, pin_memory=True, persistent_workers=True),
            torch.utils.data.DataLoader(val_set, 512, num_workers=2),
            torch.utils.data.DataLoader(test_set, 512, num_workers=2))


def train(model, optimizer, lr_schedule, loaders, device, name, ordinary_decay=5e-4,
          generated_decay=5e-6, grad_clip=100.0):
    train_loader, val_loader, test_loader = loaders
    batches = endless_batches(train_loader)
    eval_steps, test_errors = [], []
    best_val_error, selected_test_error = 100.0, 100.0
    total_steps = lr_schedule[-1][0]
    for step in tqdm(range(total_steps), desc=name):
        optimizer.param_groups[0]["lr"] = learning_rate(lr_schedule, step)
        images, labels = next(batches)
        loss = (F.cross_entropy(model(images.to(device)), labels.to(device))
                + l2_penalty(model, ordinary_decay, generated_decay))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if (step + 1) % 5000 == 0:
            val_error = error_rate(model, val_loader, device)
            test_error = error_rate(model, test_loader, device)
            if val_error < best_val_error:
                best_val_error, selected_test_error = val_error, test_error
            eval_steps.append(step + 1)
            test_errors.append(test_error)
    return dict(eval_steps=eval_steps, test_errors=test_errors,
                best_val_error=best_val_error, test_error=selected_test_error)


def plot_errors(runs, save_path):
    plt.figure(figsize=(8, 5), dpi=150)
    for run in runs:
        steps_in_thousands = [step / 1000 for step in run["eval_steps"]]
        plt.plot(steps_in_thousands, run["test_errors"],
                 label=f"{run['name']}: {run['test_error']:.2f}% test error, "
                       f"{run['n_parameters'] / 1e6:.3f}M parameters")
    plt.xlabel("training step (thousands)")
    plt.ylabel("CIFAR-10 test error (%)")
    plt.ylim(0, 30)
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Learning rate schedules, paper Tables 8 and 9
    sgd_schedule = ((28_000, 0.1), (56_000, 0.02), (84_000, 4e-3), (112_000, 8e-4),
                    (140_000, 1.6e-4))
    adam_schedule = ((168_000, 2e-3), (336_000, 1e-3), (504_000, 2e-4), (672_000, 5e-5))
    loaders = cifar10_loaders(batch_size=128, val_size=5_000)

    runs = []
    for name, lr_schedule in (("hypernetwork", adam_schedule), ("baseline", sgd_schedule)):
        use_hypernetwork = name == "hypernetwork"
        model = WideResNet(use_hypernetwork).to(device)
        optimizer = (torch.optim.Adam(model.parameters()) if use_hypernetwork else
                     torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True))
        run = train(model, optimizer, lr_schedule, loaders, device, name)
        run["name"] = name
        run["n_parameters"] = sum(p.numel() for p in model.parameters())
        runs.append(run)
    plot_errors(runs, "Imgs/hypernetworks.png")
