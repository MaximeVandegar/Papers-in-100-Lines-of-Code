import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from os import listdir
from os.path import isfile, join
import torchvision.transforms as transforms


class Dataset():

    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        self.len = len(self.files)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        transform_list = []
        transform_list += [transforms.Resize(64)]
        transform_list += [transforms.CenterCrop(64)]
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform = transforms.Compose(transform_list)
        return transform(Image.open(f'{self.data_path}/' + self.files[index]).convert('RGB'))


class Generator(nn.Module):

    def __init__(self, noise_dim=100, out_channel=3):
        super(Generator, self).__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Tanh())

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False))

    def forward(self, x):
        return self.network(x)


def sample_noise(batch_size, device):
    return torch.randn((batch_size, 100, 1, 1), device=device)


def train(generator, critic, generator_optimizer, critic_optimizer, dataloader, nb_epochs=500_000, c=0.01, ncritic=5):
    training_loss = {'generative': [], 'critic': []}
    dataset_iter = iter(dataloader)

    for epoch in tqdm(range(nb_epochs)):

        k = (20 * ncritic) if ((epoch < 25) or (epoch % 500 == 0)) else ncritic
        for _ in range(k):

            # Sample a batch from the real data
            try:
                x = next(dataset_iter).to(device)
            except:
                dataset_iter = iter(dataloader)
                x = next(dataset_iter).to(device)

            # Sample a batch of prior samples
            batch_size = x.shape[0]
            z = sample_noise(batch_size, device)

            critic_optimizer.zero_grad()
            loss = -(critic(x) - critic(generator(z).detach())).mean()
            loss.backward()
            critic_optimizer.step()
            training_loss['critic'].append(loss.item())

            with torch.no_grad():
                for param in critic.parameters():
                    param.data.clamp_(-c, c)

        # Train the generator

        # Sample a batch of prior samples
        z = sample_noise(batch_size, device)

        # Update the generator by descending its stochastic gradient
        loss = -critic(generator(z)).mean(0)

        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()
        training_loss['generative'].append(loss.item())
    return training_loss


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.ConvTranspose2d):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()


def moving_average(data, window_size):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


if __name__ == "__main__":
    device = 'cuda'
    batch_size = 64

    discriminator = Discriminator().to(device)
    generator = Generator(out_channel=3).to(device)
    discriminator.apply(init_weights)
    generator.apply(init_weights)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=0.00005)
    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=0.00005)

    data = DataLoader(Dataset(data_path='data'), batch_size=64, shuffle=True, num_workers=0)

    loss = train(generator, discriminator, optimizer_g, optimizer_d, data, nb_epochs=500_000)
    loss_critic = moving_average(loss["critic"], window_size=1000)
    plt.plot(-loss_critic)
    plt.xlabel("Discriminator iterations", fontsize=13)
    plt.ylabel("Wasserstein estimate", fontsize=13)
    plt.savefig("Imgs/wgan_loss.png")
    plt.close()

    NB_IMAGES = 8 ** 2
    generator.eval()
    img = generator(torch.randn(NB_IMAGES, 100, 1, 1, device=device))
    plt.figure(figsize=(12, 12))
    for i in range(NB_IMAGES):
        plt.subplot(8, 8, 1 + i)
        plt.axis('off')
        plt.imshow(img[i].data.cpu().transpose(0, 1).transpose(1, 2).numpy() / 2 + .5)
    plt.savefig("Imgs/generated_images.png")
    plt.close()
