import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.datasets.mnist import load_data

# load (and normalize) mnist dataset
(trainX, trainy), (testX, testy) = load_data()
trainX = np.float32(trainX) / 255.


def get_minibatch(batch_size, device):
    indices = torch.randperm(trainX.shape[0])[:batch_size]
    return torch.tensor(trainX[indices], dtype=torch.float).reshape(batch_size, -1).to(
        device), torch.nn.functional.one_hot(torch.tensor(trainy[indices], dtype=torch.long), num_classes=10).to(
        device).type(torch.float)


def sample_noise(size, device, dim=100):
    return torch.rand((size, dim), device=device)


class Generator(nn.Module):

    def __init__(self, latent_dim=100, context_dim=10, output_dim=28 * 28):
        super(Generator, self).__init__()

        self.hidden1_z = nn.Sequential(nn.Linear(latent_dim, 200), nn.Dropout(p=0.5), nn.ReLU(), )
        self.hidden1_context = nn.Sequential(nn.Linear(context_dim, 1000), nn.Dropout(p=0.5), nn.ReLU(), )
        self.hidden2 = nn.Sequential(nn.Linear(1200, 1200), nn.Dropout(p=0.5), nn.ReLU(), )
        self.out_layer = nn.Sequential(nn.Linear(1200, output_dim), nn.Sigmoid(), )

    def forward(self, noise, context):
        h = torch.cat((self.hidden1_z(noise), self.hidden1_context(context)), dim=1)
        h = self.hidden2(h)
        return self.out_layer(h)


class Discriminator(nn.Module):

    def __init__(self, input_dim=28 * 28, context_dim=10):
        super(Discriminator, self).__init__()

        self.hidden1_x = nn.Sequential(nn.Linear(input_dim, 240), nn.Dropout(p=0.5), nn.LeakyReLU(), )
        self.hidden1_context = nn.Sequential(nn.Linear(context_dim, 50), nn.Dropout(p=0.5), nn.LeakyReLU(), )
        self.hidden2 = nn.Sequential(nn.Linear(290, 240), nn.Dropout(p=0.5), nn.LeakyReLU(), )
        self.out_layer = nn.Sequential(nn.Linear(240, 1), nn.Sigmoid(), )

    def forward(self, x, context):
        h = torch.cat((self.hidden1_x(x), self.hidden1_context(context)), dim=1)
        h = self.hidden2(h)
        return self.out_layer(h)


def train(generator, discriminator, generator_optimizer, discriminator_optimizer, schedulers, nb_epochs, k=1,
          batch_size=100):
    training_loss = {'generative': [], 'discriminator': []}
    for epoch in tqdm(range(nb_epochs)):

        ### Train the disciminator
        for _ in range(k):
            # Sample a minibatch of m noise samples
            z = sample_noise(batch_size, device)
            # Sample a minibatch of m examples from the data generating distribution
            x, label = get_minibatch(batch_size, device)

            # Update the discriminator by ascending its stochastic gradient
            f_loss = torch.nn.BCELoss()(discriminator(generator(z, label), label).reshape(batch_size),
                                        torch.zeros(batch_size, device=device))
            r_loss = torch.nn.BCELoss()(discriminator(x, label).reshape(batch_size),
                                        torch.ones(batch_size, device=device))
            loss = (r_loss + f_loss) / 2
            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()
            training_loss['discriminator'].append(loss.item())

        ### Train the generator
        # Sample a minibatch of m noise samples
        z = sample_noise(batch_size, device)
        _, label = get_minibatch(batch_size, device)
        # Update the generator by descending its stochastic gradient
        loss = torch.nn.BCELoss()(discriminator(generator(z, label), label).reshape(batch_size),
                                  torch.ones(batch_size, device=device))
        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()
        training_loss['generative'].append(loss.item())

        for scheduler in schedulers:
            scheduler.step()

    return training_loss


if __name__ == "__main__":

    device = 'cuda:0'
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    optimizer_d = optim.SGD(discriminator.parameters(), lr=0.1, momentum=0.5)
    optimizer_g = optim.SGD(generator.parameters(), lr=0.1, momentum=0.5)
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer_d, 1 / 1.00004),
                  torch.optim.lr_scheduler.ExponentialLR(optimizer_g, 1 / 1.00004)]

    loss = train(generator, discriminator, optimizer_g, optimizer_d, schedulers, 287828, batch_size=100)

    plt.figure(figsize=(12, 12))
    NB_IMAGES = 10
    for i in range(10):
        z = sample_noise(NB_IMAGES, device)
        context = torch.nn.functional.one_hot(torch.ones(NB_IMAGES, dtype=torch.long) * i, num_classes=10).to(
            device).type(torch.float)
        x = generator(z, context)
        for j in range(NB_IMAGES):
            plt.subplot(10, 10, 10 * i + 1 + j)
            plt.axis('off')
            plt.imshow(x[j].data.cpu().numpy().reshape(28, 28), cmap='gray')
    plt.savefig('Imgs/cgan.png')
