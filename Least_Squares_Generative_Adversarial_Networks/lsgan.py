import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm

torch.manual_seed(8080)


def get_minibatch(batch_size, n_mixture=8):
    thetas = torch.linspace(0, 2 * np.pi, n_mixture + 1)[:n_mixture]
    mu_x, mu_y = torch.sin(thetas).reshape(-1, 1), torch.cos(thetas).reshape(-1, 1)
    idx = torch.randint(n_mixture, (batch_size,))  # Sample randomly a mixture component
    m = MultivariateNormal(torch.cat(([mu_x[idx], mu_y[idx]]), dim=1), torch.eye(2) * 0.01 ** 2)
    return m.sample()


def sample_noise(size, dim=256):
    return torch.randn((size, dim))


class Generator(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=128, output_dim=2):
        super(Generator, self).__init__()

        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, output_dim))

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=128, output_dim=1):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.network(x)


def train(generator, discriminator, generator_optimizer, discriminator_optimizer, nb_epochs, k=1,
          batch_size=100, mse_loss=nn.MSELoss(), save_every=5000):
    training_loss = {'generative': [], 'discriminator': []}
    for epoch in tqdm(range(nb_epochs)):

        #### Train the disciminator ####
        for _ in range(k):
            # Sample a minibatch of m noise samples
            z = sample_noise(batch_size).to(device)
            # Sample a minibatch of m examples from the data generating distribution
            x = get_minibatch(batch_size).to(device)

            # Update the discriminator by ascending its stochastic gradient
            pred_fake = discriminator(generator(z))
            pred_real = discriminator(x)
            loss_fake = mse_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_real = mse_loss(pred_real, torch.ones_like(pred_real))
            loss = .5 * (loss_fake + loss_real)
            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()
            training_loss['discriminator'].append(loss.item())

        #### Train the generator ####

        # Sample a minibatch of m noise samples
        z = sample_noise(batch_size).to(device)
        # Update the generator by descending its stochastic gradient
        d_score = discriminator(generator(z))
        loss = .5 * mse_loss(d_score, torch.ones_like(d_score))
        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()
        training_loss['generative'].append(loss.item())

        if (epoch % save_every) == 0:
            torch.save(generator.cpu(), f'generator_epoch_{epoch}')
            generator.to(device)

    return training_loss


def show_distribution(ax, data, epoch, fontsize=17):
    """
    Inspired from https://github.com/xudonmao/LSGAN/blob/master/stability_comparison/mixture_gaussian/ls.py
    """
    ax = sns.kdeplot(data[:, 0], data[:, 1], shade=True, cmap="Greens", n_levels=30,
                     clip=[[-4, 4]] * 2, ax=ax)
    ax.set_facecolor(sns.color_palette("Greens", n_colors=256)[0])
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('Step ' + str(epoch // 1000) + 'k', fontsize=fontsize)


if __name__ == 'main':
    device = 'cuda'
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4)
    optimizer_g = optim.Adam(generator.parameters(), lr=1e-3)
    loss = train(generator, discriminator, optimizer_g, optimizer_d, 50001, batch_size=512)

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))
    for i, epoch in enumerate([0, 5000, 15000, 25000, 40000]):
        g = torch.load(f'generator_epoch_{epoch}')
        z = sample_noise(10000)
        data = g(z).data.numpy()

        show_distribution(axes[i], data, epoch)
    plt.savefig('Imgs/lsgan.png')
