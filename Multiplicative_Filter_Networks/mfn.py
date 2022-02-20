import torch
import torch.nn as nn
import skimage
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision


class SineLayer(nn.Module):

    def __init__(self, w0):
        super(SineLayer, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(self, w0=30, in_dim=2, hidden_dim=256, out_dim=1):
        super(Siren, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, out_dim))

        # Init weights
        with torch.no_grad():
            self.net[0].weight.uniform_(-1. / in_dim, 1. / in_dim)
            self.net[2].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0, np.sqrt(6. / hidden_dim) / w0)
            self.net[4].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0, np.sqrt(6. / hidden_dim) / w0)
            self.net[6].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0, np.sqrt(6. / hidden_dim) / w0)
            self.net[8].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0, np.sqrt(6. / hidden_dim) / w0)

    def forward(self, x):
        return self.net(x)


class GaborFilter(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta=1.0):
        super(GaborFilter, self).__init__()

        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim)

        # Init weights
        self.linear.weight.data *= 128. * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        # torch.norm((x.unsqueeze(1) - mu), dim=2)**2
        norm = (x ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(x))


class GaborNet(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1, k=4):
        super(GaborNet, self).__init__()

        self.k = k
        self.gabon_filters = nn.ModuleList([GaborFilter(in_dim, hidden_dim, alpha=6.0 / k) for _ in range(k)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(k - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])

        for lin in self.linear[:k - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim))

    def forward(self, x):

        # Recursion - Equation 3
        zi = self.gabon_filters[0](x)  # Eq 3.a
        for i in range(self.k - 1):
            zi = self.linear[i](zi) * self.gabon_filters[i + 1](x)  # Eq 3.b

        return self.linear[self.k - 1](zi)  # Eq 3.c


def train(model, optim, nb_epochs=15000):
    psnrs = []
    for _ in tqdm(range(nb_epochs)):
        model_output = model(pixel_coordinates)
        loss = ((model_output - pixel_values) ** 2).mean()
        psnrs.append(20 * np.log10(1.0 / np.sqrt(loss.item())))

        optim.zero_grad()
        loss.backward()
        optim.step()

    return psnrs, model_output


if __name__ == "__main__":
    device = 'cuda'
    siren = Siren().to(device)
    gabor_net = GaborNet().to(device)

    # Target
    img = ((torch.from_numpy(skimage.data.camera()) - 127.5) / 127.5)
    img = torchvision.transforms.Resize(256)(img.unsqueeze(0))[0]
    pixel_values = img.reshape(-1, 1).to(device)

    # Input
    resolution = img.shape[0]
    tmp = torch.linspace(-1, 1, steps=resolution)
    x, y = torch.meshgrid(tmp, tmp)
    pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=13)

    for i, model in enumerate([siren, gabor_net]):
        # Training
        optim = torch.optim.Adam(lr=1e-4 if (i == 0) else 1e-2, params=model.parameters())
        psnrs, model_output = train(model, optim, nb_epochs=1000)

        axes[i + 1].imshow(model_output.cpu().view(resolution, resolution).detach().numpy(), cmap='gray')
        axes[i + 1].set_title('SIREN' if (i == 0) else 'GaborNet', fontsize=13)
        axes[4].plot(psnrs, label='SIREN' if (i == 0) else 'GaborNet', c='green' if (i == 0) else 'purple')
        axes[4].set_xlabel('Iterations', fontsize=14)
        axes[4].set_ylabel('PSNR', fontsize=14)
        axes[4].legend(fontsize=13)

    for i in range(4):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    axes[3].axis('off')
    plt.savefig('Multiplicative_Filter_Networks.png')
    plt.close()
