import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy.io

# Load and prepare training set
img_size = (32, 32)
img_data = scipy.io.loadmat('train_32x32.mat')["X"].T
trainX = torch.tensor(img_data, dtype=torch.float) / 255.


def sample_bach(batch_size, device):
    indices = torch.randperm(trainX.shape[0])[:batch_size]
    return trainX[indices].to(device)


def sample_latent(batch_size, device):
    return torch.randn((batch_size, 256, 1, 1), device=device)


class GeneratorZ(nn.Module):

    def __init__(self):
        super(GeneratorZ, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(32, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False),
            nn.BatchNorm2d(64, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(128, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(512, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(512, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False))

    def forward(self, x):
        z = self.network(x)
        mu, sigma = z[:, :256, :, :], z[:, 256:, :, :]
        return mu, sigma

    def sample(self, x):
        mu, log_sigma = self.forward(x)
        sigma = torch.exp(log_sigma)
        return torch.randn(sigma.shape, device=x.device) * sigma + mu


class GeneratorX(nn.Module):

    def __init__(self):
        super(GeneratorX, self).__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=1, padding=0, bias=False), nn.BatchNorm2d(256, momentum=0.05),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=0, bias=False), nn.BatchNorm2d(128, momentum=0.05),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=1, padding=0, bias=False), nn.BatchNorm2d(64, momentum=0.05),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=0, bias=False), nn.BatchNorm2d(32, momentum=0.05),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose2d(32, 32, 5, stride=1, padding=0, bias=False), nn.BatchNorm2d(32, momentum=0.05),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(32, momentum=0.05),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator_x = nn.Sequential(
            nn.Dropout(0.2), nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.2), nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False),
            nn.BatchNorm2d(64, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.2), nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(128, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.2), nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.2), nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(512, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), )

        self.discriminator_z = nn.Sequential(
            nn.Dropout(0.2), nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.2), nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.discriminator_xz = nn.Sequential(
            nn.Dropout(0.2), nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.2), nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.2), nn.Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.Sigmoid())

    def forward(self, x, z):
        return self.discriminator_xz(torch.cat((self.discriminator_z(z), self.discriminator_x(x)), dim=1))


def Ali(g_x, g_z, d, optimizers, nb_epochs, batch_size=100, device='cpu'):
    for _ in tqdm(range(nb_epochs)):
        # Draw M samples from the dataset and the prior
        x = sample_bach(batch_size, device)
        z = sample_latent(batch_size, device)

        # Sample from the conditionals
        x_hat = g_x(z)
        z_hat = g_z.sample(x)

        # Compute discriminator predictions
        pho_q = d(x, z_hat)
        pho_p = d(x_hat, z)

        # Compute discriminator loss
        L_d = torch.nn.BCELoss()(pho_q.reshape(batch_size), torch.ones(batch_size, device=device)) + torch.nn.BCELoss()(
            pho_p.reshape(batch_size), torch.zeros(batch_size, device=device))

        optimizers[2].zero_grad()
        L_d.backward()
        optimizers[2].step()

        # Draw M samples from the dataset and the prior
        x = sample_bach(batch_size, device)
        z = sample_latent(batch_size, device)

        # Sample from the conditionals
        x_hat = g_x(z)
        z_hat = g_z.sample(x)

        # Compute discriminator predictions
        pho_q = d(x, z_hat)
        pho_p = d(x_hat, z)

        # Compute generator loss
        L_g = torch.nn.BCELoss()(pho_p.reshape(batch_size), torch.ones(batch_size, device=device)) + torch.nn.BCELoss()(
            pho_q.reshape(batch_size), torch.zeros(batch_size, device=device))

        optimizers[0].zero_grad()
        optimizers[1].zero_grad()
        L_g.backward()
        optimizers[0].step()
        optimizers[1].step()


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.ConvTranspose2d):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()


if __name__ == "__main__":
    device = 'cuda'

    gz = GeneratorZ().to(device)
    gx = GeneratorX().to(device)
    d = Discriminator().to(device)

    gx.apply(init_weights)
    gz.apply(init_weights)
    d.apply(init_weights)

    optimizers = [optim.Adam(gz.parameters(), lr=0.0001, betas=(0.5, 0.999)),
                  optim.Adam(gx.parameters(), lr=0.0001, betas=(0.5, 0.999)),
                  optim.Adam(d.parameters(), lr=0.0001, betas=(0.5, 0.999))]

    Ali(gx, gz, d, optimizers, 73_000, device=device)

    NB_IMAGES = 8 ** 2
    z = sample_latent(NB_IMAGES, device)
    x_hat = gx(z)
    plt.figure(figsize=(12, 12))
    for i in range(NB_IMAGES):
        plt.subplot(8, 8, 1 + i)
        plt.axis('off')
        plt.imshow(x_hat[i].data.cpu().numpy().T)
    plt.savefig("Img/ali.png")
