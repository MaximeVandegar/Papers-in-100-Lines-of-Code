import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from keras.datasets.mnist import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt

def squeeze2d(x, factor=2):
    B, C, H, W = x.size()
    x = x.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x

def unsqueeze2d(x, factor=2):
    B, C, H, W = x.size()
    x = x.view(B, C // (factor * factor), factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor * factor), H * factor, W * factor)
    return x

def checkerboard_mask(H, W, invert=False, device=None):
    y = torch.arange(H, device=device).view(-1, 1)
    x = torch.arange(W, device=device).view(1, -1)
    m = ((y + x) % 2).float()
    if invert:
        m = 1.0 - m
    return m.view(1, 1, H, W)

def channel_mask(C, first_half=True, device=None):
    m = torch.zeros(C, device=device)
    half = C // 2
    if first_half:
        m[:half] = 1.0
    else:
        m[half:] = 1.0
    return m.view(1, C, 1, 1)

class BatchNorm2dNoAffine(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(1, num_features, 1, 1))
        self.register_buffer("running_var",  torch.ones(1, num_features, 1, 1))

    def _compute_stats_and_update(self, x):
        m = x.mean(dim=[0, 2, 3], keepdim=True)
        v = x.var(dim=[0, 2, 3], unbiased=False, keepdim=True)
        with torch.no_grad():
            self.running_mean.mul_(self.momentum).add_((1 - self.momentum) * m)
            self.running_var.mul_(self.momentum).add_((1 - self.momentum) * v)
        return m, v

    def forward(self, x, reverse=False, return_logdet=False):
        B, C, H, W = x.shape
        if self.training:
            mean, var = self._compute_stats_and_update(x)
        else:
            mean, var = self.running_mean, self.running_var

        if reverse:
            y = x * torch.sqrt(var + self.eps) + mean
            if return_logdet:
                ld = (H * W) * 0.5 * torch.sum(torch.log(var + self.eps)).expand(B)
                return y, ld
            return y
        else:
            y = (x - mean) / torch.sqrt(var + self.eps)
            if return_logdet:
                ld = -(H * W) * 0.5 * torch.sum(torch.log(var + self.eps)).expand(B)
                return y, ld
            return y

class ResBlock(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.conv1 = utils.weight_norm(nn.Conv2d(channels, hidden, kernel_size=3, padding=1))
        self.conv2 = utils.weight_norm(nn.Conv2d(hidden, channels, kernel_size=1))
        self.bn1 = BatchNorm2dNoAffine(hidden)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.conv2(h)
        return x + h

class STNet(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1), nn.ReLU(),
            ResBlock(hidden, hidden),
            ResBlock(hidden, hidden),
            ResBlock(hidden, hidden),
            ResBlock(hidden, hidden),
            nn.Conv2d(hidden, hidden, kernel_size=1), nn.ReLU(),
            nn.Conv2d(hidden, 2*channels, kernel_size=3, padding=1)
        )
        self.logscale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        st = self.model(x)
        C = st.size(1) // 2
        s, t = st[:, :C], st[:, C:]
        s = torch.tanh(s) * torch.exp(self.logscale + torch.tensor(1.0, device=s.device))
        return s, t

class Coupling(nn.Module):
    def __init__(self, channels, mask, hidden=64, bn_momentum=0.9, bn_eps=1e-5):
        super().__init__()
        self.register_buffer("mask", mask)
        self.st = STNet(channels, hidden)
        self.bn_out = BatchNorm2dNoAffine(channels, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x, reverse=False):
        if reverse:
            x, ld_bn = self.bn_out(x, reverse=True, return_logdet=True)
            x_id = self.mask * x
            s, t = self.st(x_id)
            y = x_id + (1 - self.mask) * ((x - t) * torch.exp(-s))
            ld_affine = -(((1 - self.mask) * s).sum(dim=(1, 2, 3)))
            return y, (ld_affine + ld_bn)

        x_id = self.mask * x
        s, t = self.st(x_id)
        y_raw = x_id + (1 - self.mask) * (x * torch.exp(s) + t)
        ld_affine = (((1 - self.mask) * s).sum(dim=(1, 2, 3)))
        y, ld_bn = self.bn_out(y_raw, reverse=False, return_logdet=True)
        return y, (ld_affine + ld_bn)

class Squeeze(nn.Module):
    def forward(self, x, reverse=False):
        return (unsqueeze2d(x) if reverse else squeeze2d(x)), torch.zeros(x.size(0), device=x.device)

class RealNVP(nn.Module):
    def __init__(self):
        super().__init__()

        cb_m0_28 = checkerboard_mask(28, 28, invert=False)
        cb_m1_28 = checkerboard_mask(28, 28, invert=True)
        ch_m0_4 = channel_mask(4, first_half=True)
        ch_m1_4 = channel_mask(4, first_half=False)
        cb_m0_14 = checkerboard_mask(14, 14, invert=False)
        cb_m1_14 = checkerboard_mask(14, 14, invert=True)
        ch_m0_16 = channel_mask(16, first_half=True)
        ch_m1_16 = channel_mask(16, first_half=False)
        cb_m0_7 = checkerboard_mask(7, 7, invert=False)
        cb_m1_7 = checkerboard_mask(7, 7, invert=True)

        self.model = nn.ModuleList([
            Coupling(channels=1, mask=cb_m0_28, hidden=32),
            Coupling(channels=1, mask=cb_m1_28, hidden=32),
            Coupling(channels=1, mask=cb_m0_28, hidden=32),
            Squeeze(),
            Coupling(channels=4, mask=ch_m0_4, hidden=64),
            Coupling(channels=4, mask=ch_m1_4, hidden=64),
            Coupling(channels=4, mask=ch_m0_4, hidden=64),
            Coupling(channels=4, mask=cb_m0_14, hidden=64),
            Coupling(channels=4, mask=cb_m1_14, hidden=64),
            Coupling(channels=4, mask=cb_m0_14, hidden=64),
            Squeeze(),
            Coupling(channels=16, mask=ch_m0_16, hidden=128),
            Coupling(channels=16, mask=ch_m1_16, hidden=128),
            Coupling(channels=16, mask=ch_m0_16, hidden=128),
            Coupling(channels=16, mask=cb_m0_7, hidden=128),
            Coupling(channels=16, mask=cb_m1_7, hidden=128),
            Coupling(channels=16, mask=cb_m0_7, hidden=128),
            Coupling(channels=16, mask=cb_m1_7, hidden=128)
        ])

    def forward(self, x, reverse=False):
        logdet = torch.zeros(x.size(0), device=x.device)
        if not reverse:
            for layer in self.model:
                x, ld = layer(x, reverse=False)
                logdet += ld
            return x, logdet
        else:
            for layer in reversed(self.model):
                x, ld = layer(x, reverse=True)
                logdet += ld
            return x, logdet

class LogitTransform(nn.Module):
    def __init__(self, alpha=1e-6):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x, reverse=False):  # y = logit(α + (1 − 2α) x)
        D = x[0].numel()
        if not reverse:
            a = self.alpha
            s = a + (1 - 2 * a) * x
            y = torch.log(s) - torch.log(1 - s)
            ld = (np.log(1 - 2 * a) * D) - (torch.log(s) + torch.log(1 - s)).sum(dim=(1, 2, 3))
            return y, ld
        else:
            y = x
            s = torch.sigmoid(y)
            a = self.alpha
            x = (s - a) / (1 - 2 * a)
            ld = - (np.log(1 - 2 * a) * D) + (torch.log(s) + torch.log(1 - s)).sum(dim=(1, 2, 3))
            return x, ld

def gaussian_log_p(z):
    D = z[0].numel()
    return -0.5 * (z ** 2).sum(dim=(1, 2, 3)) - 0.5 * D * torch.log(torch.tensor(2 * np.pi, device=z.device))

if __name__ == "__main__":
    (trainX, _), (testX, _) = load_data()
    trainX = torch.from_numpy(np.float32(trainX) / 255.0)
    train_loader = DataLoader(TensorDataset(trainX[:, None]),
                              batch_size=128, shuffle=True, drop_last=True)

    n_epochs = 200
    device = "cuda"
    model = RealNVP().to(device)
    logit_transform = LogitTransform(alpha=1e-6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)

    training_loss = []
    for epoch in tqdm(range(n_epochs)):
        total_loss = 0.0

        for (x_batch,) in tqdm(train_loader):
            x_batch = x_batch.to(device)

            u = torch.rand_like(x_batch)  # Uniform dequantization
            x_deq = (x_batch * 255.0 + u) / 256.0
            y, logdet_logit = logit_transform(x_deq, reverse=False)  # Logit transform
            z, logdet_flow = model(y, reverse=False)

            log_pz = gaussian_log_p(z)
            log_px = log_pz + logdet_flow + logdet_logit
            loss = -log_px.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    n_rows, n_cols = 5, 5
    with torch.no_grad():
        z = torch.randn(n_rows * n_cols, 16, 7, 7, device=device)
        y, _ = model(z, reverse=True)  # y is in logit space
        x, _ = logit_transform(y, reverse=True)  # back to [0,1]
        imgs = x.clamp(0, 1).cpu().numpy().squeeze(1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j].imshow(imgs[i * n_cols + j], cmap="gray")
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig('Imgs/generated_samples.png')
