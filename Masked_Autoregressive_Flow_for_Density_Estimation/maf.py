import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from keras.datasets.mnist import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_mask(in_deg, out_deg):
    return (in_deg[None, :] <= out_deg[:, None]).float()

def std_normal_logprob(z):
    D = z.shape[1]
    return -0.5 * (z**2).sum(dim=1) - 0.5 * D * math.log(2 * math.pi)

class BatchNorm(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(1, features))
        self.gamma = nn.Parameter(torch.zeros(1, features))
        self.register_buffer("mean", torch.zeros(1, features))
        self.register_buffer("var",  torch.ones(1, features))
        # accumulators for calibration (collecting statistics from full training data)
        self._collecting = False
        self.register_buffer("_sum",   torch.zeros(1, features))
        self.register_buffer("_sum_of_square", torch.zeros(1, features))
        self.register_buffer("_count", torch.zeros(1, 1))

    def begin_calibration(self):
        self._collecting = True
        self._sum.zero_()
        self._sum_of_square.zero_()
        self._count.zero_()

    def end_calibration(self):
        total = self._count.item()
        if total > 0:
            mean = self._sum / total
            var = (self._sum_of_square / total) - mean**2
            var.clamp_(min=self.eps)
            self.mean.copy_(mean)
            self.var.copy_(var)
        self._collecting = False

    def _compute_batch_stats(self, x):
        m = x.mean(0, keepdim=True)
        v = x.var(0, unbiased=False, keepdim=True)
        return m, v

    def forward(self, x):
        if self._collecting:
            self._sum += x.sum(0, keepdim=True)
            self._sum_of_square += (x**2).sum(0, keepdim=True)
            self._count += torch.tensor([[x.size(0)]], dtype=x.dtype, device=x.device)
            m, v = self._compute_batch_stats(x)
        elif self.training:
            m, v = self._compute_batch_stats(x)
        else:
            m, v = self.mean, self.var

        u = (x - m) * torch.rsqrt(v + self.eps) * torch.exp(self.gamma) + self.beta
        ld = (self.gamma - 0.5 * torch.log(v + self.eps)).sum(dim=1).expand(x.size(0))
        return u, ld

    def inverse(self, u):
        m, v = self.mean, self.var
        x = (u - self.beta) * torch.sqrt(v + self.eps) * torch.exp(-self.gamma) + m
        return x

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

class MADE(nn.Module):
    def __init__(self, features, hidden_features, hidden_layers=2):
        super().__init__()
        self.features = features
        self.hidden_layers = hidden_layers

        ordering = torch.arange(features) + 1  # degrees in {1..D}
        self.hidden_degrees = []  # Hidden degrees in {1..D-1}
        for _ in range(hidden_layers):
            deg = torch.randint(1, features, (hidden_features,))  # [1, D-1]
            self.hidden_degrees.append(deg)
        out_deg = torch.cat([ordering - 1, ordering - 1], dim=0)  # for mean and log_scale

        layers = [MaskedLinear(features, hidden_features, create_mask(ordering, self.hidden_degrees[0])), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [MaskedLinear(hidden_features, hidden_features,
                                    create_mask(self.hidden_degrees[i], self.hidden_degrees[i+1])), nn.ReLU()]
        layers.append(MaskedLinear(hidden_features, 2*features, create_mask(self.hidden_degrees[-1], out_deg)))
        self.net = nn.Sequential(*layers)

    def forward(self, y):
        shift, log_scale = self.net(y).chunk(2, dim=1)
        z = (y - shift) * torch.exp(-log_scale)
        ld = -log_scale.sum(dim=1)
        return z, ld

    def inverse(self, z):
        y = torch.zeros_like(z)
        for j in range(self.features):
            shift, log_scale = self.net(y).chunk(2, dim=1)
            y[:, j] = shift[:, j] + z[:, j] * torch.exp(log_scale[:, j].clamp(-10.0, 10.0))
        return y

class ReversePermutation(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.register_buffer("perm", torch.arange(features-1, -1, -1).long())

    def forward(self, x):
        return x[:, self.perm], torch.zeros(x.size(0), device=x.device)

    def inverse(self, x):
        return x[:, self.perm]


class MAF(nn.Module):
    def __init__(self, features, hidden_features, num_layers=10, hidden_layers=2):
        super().__init__()
        self.transforms = nn.ModuleList()
        for i in range(num_layers):
            self.transforms.append(MADE(features, hidden_features, hidden_layers))
            self.transforms.append(BatchNorm(features))
            if i < num_layers - 1:
                self.transforms.append(ReversePermutation(features))

    def forward_and_logdet(self, y):
        z, ld_sum = y, torch.zeros(y.size(0), device=y.device)
        for t in self.transforms:
            z, ld = t(z)
            ld_sum += ld
        return z, ld_sum

    def invert(self, z):
        x = z
        for t in reversed(self.transforms):
            x = t.inverse(x)
        return x

    def begin_calibration(self):
        for t in self.transforms:
            if isinstance(t, BatchNorm):
                t.begin_calibration()

    def end_calibration(self):
        for t in self.transforms:
            if isinstance(t, BatchNorm):
                t.end_calibration()

class LogitTransform1d(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, reverse=False):
        B, D = x.shape
        if not reverse:
            a = self.alpha
            s = a + (1 - 2*a) * x
            s = s.clamp(1e-12, 1-1e-12)
            y = torch.log(s) - torch.log(1 - s)
            ld = (math.log(1 - 2 * a) * D) - (torch.log(s) + torch.log(1 - s)).sum(dim=1)
            return y, ld
        else:
            s = torch.sigmoid(x)
            a = self.alpha
            x = (s - a) / (1 - 2 * a)
            s = s.clamp(1e-12, 1-1e-12)
            ld = -(math.log(1 - 2 * a) * D) + (torch.log(s) + torch.log(1 - s)).sum(dim=1)
            return x, ld


if __name__ == "__main__":
    device = "cuda"

    (trainX, _), (testX, _) = load_data()
    trainX = torch.from_numpy(trainX.astype(np.float32)/255.0).view(-1, 28 * 28)
    train_loader = DataLoader(TensorDataset(trainX), batch_size=100, shuffle=True, drop_last=True)

    model = MAF(28 * 28, 1024, num_layers=10, hidden_layers=2).to(device)
    logit = LogitTransform1d().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(101):
        for (xb,) in tqdm(train_loader):
            xb = xb.to(device)
            u = torch.rand_like(xb)  # dequantization
            x_deq = (xb * 255.0 + u) / 256.0
            y, _ = logit(x_deq, reverse=False)
            z, logdet_flow = model.forward_and_logdet(y)
            log_pz = std_normal_logprob(z)
            loss = -(log_pz + logdet_flow).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        model.begin_calibration()
        for (xb,) in train_loader:
            xb = xb.to(device)
            u = torch.rand_like(xb)
            x_deq = (xb * 255.0 + u) / 256.0
            y, _ = logit(x_deq, reverse=False)
            _ = model.forward_and_logdet(y)
        model.end_calibration()
        model.eval()

    temperature = 0.7
    with torch.no_grad():
        z = temperature * torch.randn(25, 28 * 28, device=device)
        y = model.invert(z)  # logit-space sample
        x, _ = logit(y, reverse=True)  # back to pixel space in [0,1]
        x_imgs = x.view(z.shape[0], 28, 28).cpu().clamp(0, 1).numpy()

    fig, axes = plt.subplots(5, 5, figsize=(5, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_imgs[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig('Imgs/samples_maf.png')
