import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.datasets.mnist import load_data
from torch.utils.data import DataLoader, TensorDataset


def generate_training_dataset(N=100):
    (trainX, trainy), (testX, testy) = load_data()
    trainX = np.float32(trainX) / 255.
    testX = np.float32(testX) / 255.

    trainX_supervised = []
    trainX_unsupervised = []
    trainy_supervised = []
    trainy_unsupervised = []
    for i in range(10):
        idx = np.where(trainy == i)[0]
        np.random.shuffle(idx)

        idx_sup, idx_unsup = idx[:(N // 10)], idx[(N // 10):]
        trainy_supervised.append(trainy[idx_sup])
        trainy_unsupervised.append(trainy[idx_unsup])
        trainX_supervised.append(trainX[idx_sup])
        trainX_unsupervised.append(trainX[idx_unsup])

    trainy_supervised = np.concatenate(trainy_supervised)
    trainy_unsupervised = np.concatenate(trainy_unsupervised)
    trainX_supervised = np.concatenate(trainX_supervised)
    trainX_unsupervised = np.concatenate(trainX_unsupervised)
    return trainy_supervised, trainy_unsupervised, trainX_supervised, trainX_unsupervised, testX, testy


class GaussianNoiseLayer(nn.Module):

    def __init__(self, sigma):
        super(GaussianNoiseLayer, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.train:
            noise = torch.randn(x.shape, device=x.device) * self.sigma
            return x + noise
        else:
            return x


class Generator(nn.Module):

    def __init__(self, latent_dim=100, output_dim=28 * 28):
        super(Generator, self).__init__()

        self.network = nn.Sequential(nn.Linear(latent_dim, 500), nn.Softplus(), nn.BatchNorm1d(500),
                                     nn.Linear(500, 500), nn.Softplus(), nn.BatchNorm1d(500),
                                     nn.utils.weight_norm(nn.Linear(500, output_dim)), nn.Sigmoid())

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.network_head = nn.Sequential(GaussianNoiseLayer(sigma=0.3), nn.utils.weight_norm(nn.Linear(28 ** 2, 1000)),
                                          nn.ReLU(),
                                          GaussianNoiseLayer(sigma=0.5), nn.utils.weight_norm(nn.Linear(1000, 500)),
                                          nn.ReLU(),
                                          GaussianNoiseLayer(sigma=0.5), nn.utils.weight_norm(nn.Linear(500, 250)),
                                          nn.ReLU(),
                                          GaussianNoiseLayer(sigma=0.5), nn.utils.weight_norm(nn.Linear(250, 250)),
                                          nn.ReLU(),
                                          GaussianNoiseLayer(sigma=0.5), nn.utils.weight_norm(nn.Linear(250, 250)),
                                          nn.ReLU())
        self.network_tail = nn.Sequential(GaussianNoiseLayer(sigma=0.5), nn.utils.weight_norm(nn.Linear(250, 10)))

    def forward(self, x):
        features = self.network_head(x)
        score = self.network_tail(features)
        return features, score


def train(g, d, g_optimizer, d_optimizer, sup_data_loader, unsup_data_loader1, unsup_data_loader2, testX, testy,
          nb_epochs=180000, latent_dim=100, lambda_=1.0, device='cpu'):
    testing_accuracy = []
    for epoch in tqdm(range(nb_epochs)):
        ### Train discriminator
        supervised_data, target = iter(sup_data_loader).next()
        unsupervised_data = iter(unsup_data_loader1).next().to(device)
        noise = torch.rand((supervised_data.shape[0], latent_dim), device=device)
        fake_data = g(noise)
        # Supervised loss
        log_prob = torch.nn.functional.log_softmax(d(supervised_data.to(device))[1], dim=1)
        supervised_loss = torch.nn.functional.nll_loss(log_prob, target.to(device))
        # Unsupervised loss
        _, prob_before_softmax_unsupervised = d(unsupervised_data)
        _, prob_before_softmax_fake = d(fake_data)
        unsupervised_loss = .5 * torch.nn.functional.softplus(torch.logsumexp(prob_before_softmax_fake, dim=1)).mean() \
                            - .5 * torch.logsumexp(prob_before_softmax_unsupervised, dim=1).mean() \
                            + .5 * torch.nn.functional.softplus(
            torch.logsumexp(prob_before_softmax_unsupervised, dim=1)).mean()

        loss = supervised_loss + lambda_ * unsupervised_loss
        d_optimizer.zero_grad()
        loss.backward()
        d_optimizer.step()

        ### Train generator
        unsupervised_data = iter(unsup_data_loader2).next().to(device)
        noise = torch.rand((unsupervised_data.shape[0], latent_dim), device=device)
        fake_data = g(noise)
        # Feature matching loss
        features_gen, _ = d(fake_data)
        features_real, _ = d(unsupervised_data)
        loss = torch.nn.functional.mse_loss(features_gen, features_real)

        g_optimizer.zero_grad()
        loss.backward()
        g_optimizer.step()

        # Testing
        d.train(mode=False)
        _, log_prob = d(torch.from_numpy(testX.reshape(-1, 28 * 28)).to(device))
        testing_accuracy.append(
            (log_prob.argmax(-1) == torch.from_numpy(testy).to(device)).sum().item() / testy.shape[0])
        d.train(mode=True)

    return testing_accuracy


if __name__ == "__main__":
    device = 'cuda'

    trainy_sup, trainy_unsup, trainX_sup, trainX_unsup, testX, testy = generate_training_dataset(100)
    sup_data_loader = DataLoader(TensorDataset(torch.from_numpy(trainX_sup.reshape(-1, 28 * 28)), torch.from_numpy(
        trainy_sup.reshape(-1))), batch_size=100, shuffle=True)
    unsup_data_loader1 = DataLoader(torch.from_numpy(trainX_unsup.reshape(-1, 28 * 28)), batch_size=100, shuffle=True)
    unsup_data_loader2 = DataLoader(torch.from_numpy(trainX_unsup.reshape(-1, 28 * 28)), batch_size=100, shuffle=True)

    g = Generator().to(device)
    d = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(g.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(d.parameters(), lr=0.001, betas=(0.5, 0.999))
    testing_accuracy = train(g, d, optimizer_g, optimizer_d, sup_data_loader, unsup_data_loader1, unsup_data_loader2,
                             testX, testy, nb_epochs=50_000, latent_dim=100, lambda_=1.0, device=device)
    plt.plot(testing_accuracy)
    plt.ylabel('Testing accuracy', fontsize=13)
    plt.xlabel('Epochs', fontsize=13)
    plt.title('100 labeled examples', fontsize=13)
    plt.savefig('Imgs/permutation_invariant_MNIST.png')
