import torch
import torch.nn as nn
import skimage
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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
            self.net[2].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[4].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[6].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[8].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1):
        super(MLP, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


def train(model, model_optimizer, nb_epochs=15000):
    psnr = []
    for _ in tqdm(range(nb_epochs)):
        model_output = model(pixel_coordinates)
        loss = ((model_output - pixel_values) ** 2).mean()
        psnr.append(20 * np.log10(1.0 / np.sqrt(loss.item())))

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

    return psnr, model_output


if __name__ == "__main__":
    device = 'cuda'

    siren = Siren().to(device)
    mlp = MLP().to(device)

    # Target
    img = ((torch.from_numpy(skimage.data.camera()) - 127.5) / 127.5)
    pixel_values = img.reshape(-1, 1).to(device)

    # Input
    resolution = img.shape[0]
    tmp = torch.linspace(-1, 1, steps=resolution)
    x, y = torch.meshgrid(tmp, tmp)
    pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=13)

    for i, model in enumerate([mlp, siren]):
        # Training
        optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
        psnr, model_output = train(model, optim, nb_epochs=15000)

        axes[i + 1].imshow(model_output.cpu().view(resolution, resolution).detach().numpy(), cmap='gray')
        axes[i + 1].set_title('ReLU' if (i == 0) else 'SIREN', fontsize=13)
        axes[4].plot(psnr, label='ReLU' if (i == 0) else 'SIREN', c='C0' if (i == 0) else 'mediumseagreen')
        axes[4].set_xlabel('Iterations', fontsize=14)
        axes[4].set_ylabel('PSNR', fontsize=14)
        axes[4].legend(fontsize=13)

    for i in range(4):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    axes[3].axis('off')
    plt.savefig('Imgs/Siren.png')
    plt.close()
