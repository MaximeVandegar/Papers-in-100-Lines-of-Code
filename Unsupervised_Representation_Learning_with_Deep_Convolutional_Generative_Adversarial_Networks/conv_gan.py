import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Dataset:

    def __init__(self, data_path):
        self.data_path = data_path
        self.files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        self.len = len(self.files)
        self.transform = transforms.Compose(
            [transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.transform(Image.open(f'{self.data_path}/' + self.files[index]).convert('RGB'))


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, kernel_size=(4, 4), stride=(1, 1), bias=False), nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(3), nn.Tanh(), )

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False), nn.Sigmoid())

    def forward(self, x):
        return self.network(x)


def sample_noise(batch_size, device):
    return torch.randn((batch_size, 100, 1, 1), device=device)


def train(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader, nb_epochs=5, k=1):

    for _ in range(nb_epochs):
        for x in tqdm(dataloader):
            # Train the discriminator

            batch_size = x.shape[0]
            for _ in range(k):
                # Sample a minibatch of m noise samples
                z = sample_noise(batch_size, device)
                # Sample a minibatch of m examples from the data generating distribution
                x = x.to(device)

                # Update the discriminator by ascending its stochastic gradient
                f_loss = torch.nn.BCELoss()(discriminator(generator(z)).reshape(batch_size),
                                            torch.zeros(batch_size, device=device))
                r_loss = torch.nn.BCELoss()(discriminator(x).reshape(batch_size), torch.ones(batch_size, device=device))
                loss = (r_loss + f_loss) / 2
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()

            # Train the generator

            # Sample a minibatch of m noise samples
            z = sample_noise(batch_size, device)
            # Update the generator by descending its stochastic gradient
            loss = torch.nn.BCELoss()(discriminator(generator(z)).reshape(batch_size),
                                      torch.ones(batch_size, device=device))
            generator_optimizer.zero_grad()
            loss.backward()
            generator_optimizer.step()


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.ConvTranspose2d):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()


if __name__ == "__main__":
    device = 'cuda:0'
    batch_size = 128
    data = DataLoader(Dataset('data'), batch_size=128, shuffle=True)

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    discriminator.apply(init_weights)
    generator.apply(init_weights)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    train(generator, discriminator, optimizer_g, optimizer_d, data, nb_epochs=5)

    NB_IMAGES = 8 ** 2
    img = generator(sample_noise(NB_IMAGES, device))
    all_images = np.zeros((64 * 8, 64 * 8, 3))
    for i in range(int(np.sqrt(NB_IMAGES))):
        for j in range(int(np.sqrt(NB_IMAGES))):
            all_images[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, :] = img[i * int(
                np.sqrt(NB_IMAGES)) + j].data.cpu().transpose(0, 1).transpose(1, 2).numpy() / 2 + .5
    plt.figure(figsize=(16, 16))
    plt.imshow(all_images)
    plt.gca().axis('off')
    plt.savefig('Imgs/generated_bedrooms_after_five_epoch.png', dpi=300)
