import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.autograd import Variable
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class DownResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size):
        super(DownResBlock, self).__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, bias=True),
            nn.AvgPool2d(2))
        self.network = nn.Sequential(
            nn.InstanceNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, input_dim, kernel_size=filter_size, padding=filter_size // 2),
            nn.InstanceNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, output_dim, kernel_size=filter_size, padding=filter_size // 2),
            nn.AvgPool2d(2))

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        output = self.network(inputs)
        return shortcut + output


class UpResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size):
        super(UpResBlock, self).__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(input_dim, 4 * output_dim, kernel_size=1, stride=1, bias=True),
            nn.PixelShuffle(2))
        self.network = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(input_dim, output_dim, kernel_size=filter_size, padding=filter_size // 2),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=filter_size, padding=filter_size // 2))

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        output = self.network(inputs)
        return shortcut + output


class Generator(nn.Module):
    def __init__(self, dim=64):
        super(Generator, self).__init__()
        self.dim = dim
        self.input_layer = nn.Linear(128, 4 * 4 * 8 * dim)
        self.model = nn.Sequential(UpResBlock(8 * dim, 8 * dim, 3),
                                   UpResBlock(8 * dim, 4 * dim, 3),
                                   UpResBlock(4 * dim, 2 * dim, 3),
                                   UpResBlock(2 * dim, 1 * dim, 3),
                                   nn.BatchNorm2d(1 * dim),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(1 * dim, 3, kernel_size=3, padding=1),
                                   nn.Tanh())

    def forward(self, noise):
        output = self.input_layer(noise)
        output = output.view(-1, 8 * self.dim, 4, 4)
        return self.model(output).view(-1, 3, 64, 64)


class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()
        self.dim = dim

        self.input_conv = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.model = nn.Sequential(DownResBlock(dim, 2 * dim, 3),
                                   DownResBlock(2 * dim, 4 * dim, 3),
                                   DownResBlock(4 * dim, 8 * dim, 3),
                                   DownResBlock(8 * dim, 8 * dim, 3),)
        self.output_linear = nn.Linear(4 * 4 * 8 * dim, 1)

    def forward(self, inputs):
        output = self.input_conv(inputs)
        output = self.model(output)
        output = output.view(-1, 4 * 4 * 8 * self.dim)
        return self.output_linear(output).view(-1)


class Dataset():

    def __init__(self, data_path):
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


def sample_noise(batch_size, device):
    return torch.randn((batch_size, 128), device=device)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def train(generator, critic, generator_optimizer, critic_optimizer, dataloader, nb_epochs, ncritic=5, lambda_gp=10.):
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
            batch_size = x.shape[0]
            # Sample a batch of prior samples
            z = sample_noise(batch_size, device)

            critic_optimizer.zero_grad()
            x_tilde = generator(z).detach()
            eps = torch.rand((x_tilde.shape[0], 1, 1, 1), device=device)
            x_hat = Variable(eps * x + (1 - eps) * x_tilde, requires_grad=True)
            loss = -(critic(x).squeeze(0) - critic(x_tilde).squeeze(0)).mean()

            gradients = torch.autograd.grad(outputs=critic(x_hat).squeeze(0), inputs=x_hat, grad_outputs=torch.ones(
                (x_hat.shape[0]), device=x_hat.device), create_graph=True, retain_graph=True)[0]
            gradient_penalty = ((gradients.reshape(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
            loss = loss + lambda_gp * gradient_penalty
            loss.backward()
            critic_optimizer.step()

            training_loss['critic'].append(loss.item())

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


if __name__ == "__main__":
    device = 'cuda'

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0., 0.9))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0., 0.9))

    data = DataLoader(Dataset('../data'), batch_size=64, shuffle=True, num_workers=0)
    loss = train(generator, discriminator, optimizer_g, optimizer_d, data, 25_000)

    loss_critic = moving_average(loss["critic"], window_size=1000)
    plt.plot(-np.array(loss["critic"]))
    plt.plot(-loss_critic)
    plt.xlabel("Discriminator iterations", fontsize=13)
    plt.ylabel("Negative critic loss", fontsize=13)
    plt.savefig("Imgs/wgan_loss.png")
    plt.close()

    generator.eval()
    NB_IMAGES = 8 ** 2
    img = generator(sample_noise(NB_IMAGES, device))
    plt.figure(figsize=(12, 12))
    for i in range(NB_IMAGES):
        plt.subplot(8, 8, 1 + i)
        plt.axis('off')
        plt.imshow(img[i].data.cpu().transpose(0, 1).transpose(1, 2).numpy() / 2 + .5)
    plt.savefig("Imgs/generated_samples.png")
    plt.close()
