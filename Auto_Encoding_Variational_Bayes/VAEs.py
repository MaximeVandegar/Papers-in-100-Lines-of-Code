import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
import scipy.io

# Load and prepare training set
img_size = (28, 20)
img_data = scipy.io.loadmat('Data/frey_rawface.mat')["ff"]
img_data = img_data.T.reshape((-1, img_size[0], img_size[1]))
trainX = torch.tensor(img_data[:int(0.8 * img_data.shape[0])], dtype=torch.float)/255.

def get_minibatch(batch_size, device='cpu'):
    indices = torch.randperm(trainX.shape[0])[:batch_size]
    return trainX[indices].reshape(batch_size, -1).to(device)

class Model(nn.Module):
    def __init__(self, data_dim=2, context_dim=2, hidden_dim=200, constrain_mean=False):
        super(Model, self).__init__()
        '''
        Model p(y|x) as N(mu, sigma) where mu and sigma are Neural Networks
        '''

        self.h = nn.Sequential(
                 nn.Linear(context_dim, hidden_dim),
                 nn.Tanh(),
                 )
        self.log_var = nn.Sequential(nn.Linear(hidden_dim, data_dim),)

        if constrain_mean:
            self.mu = nn.Sequential(nn.Linear(hidden_dim, data_dim), nn.Sigmoid())
        else:
            self.mu = nn.Sequential(nn.Linear(hidden_dim, data_dim), )

    def get_mean_and_log_var(self, x):
        h = self.h(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def forward(self, epsilon, x):
        '''
        Sample y ~ p(y|x) using the reparametrization trick
        '''
        mu, log_var = self.get_mean_and_log_var(x)
        sigma = torch.sqrt(torch.exp(log_var))
        return epsilon * sigma + mu

    def compute_log_density(self, y, x):
        '''
        Compute log p(y|x)
        '''
        mu, log_var = self.get_mean_and_log_var(x)
        log_density = -.5 * (torch.log(2 * torch.tensor(np.pi)) + log_var + (((y-mu)**2)/(torch.exp(log_var) + 1e-10))).sum(dim=1)
        return log_density

    def compute_KL(self, x):
        '''
        Assume that p(x) is a normal gaussian distribution; N(0, 1)
        '''
        mu, log_var = self.get_mean_and_log_var(x)
        return -.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)
    
def AVEB(encoder, decoder, encoder_optimizer, decoder_optimizer, nb_epochs, M=100, L=1, latent_dim=2):
    losses = []
    for epoch in tqdm(range(nb_epochs)):
        x = get_minibatch(M, device=device)
        epsilon = torch.normal(torch.zeros(M * L, latent_dim), torch.ones(latent_dim)).to(device)

        # Compute the loss
        z = encoder(epsilon, x)
        log_likelihoods = decoder.compute_log_density(x, z)
        kl_divergence = encoder.compute_KL(x)
        loss = (kl_divergence - log_likelihoods.view(-1, L).mean(dim=1)).mean()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        losses.append(loss.item())
    return losses

if __name__ == "__main__": 
    device = 'cuda:0'
    encoder = Model(data_dim=2, context_dim=img_size[0]*img_size[1], hidden_dim=200).to(device)
    decoder = Model(data_dim=img_size[0]*img_size[1], context_dim=2, hidden_dim=200, constrain_mean=True).to(device)
    encoder_optimizer = torch.optim.Adagrad(encoder.parameters(), lr=0.01, weight_decay=0.5)
    decoder_optimizer = torch.optim.Adagrad(decoder.parameters(), lr=0.01)

    loss = AVEB(encoder, decoder, encoder_optimizer, decoder_optimizer, 10**6)
    
    plt.figure(figsize=(4, 4))
    plt.plot(100*np.arange(len(loss)), -np.array(loss), c='r', label='AEVD (train)')
    plt.xscale('log')
    plt.xlim([10**5, 10**8])
    plt.ylim(0, 1600)
    plt.title(r'Frey Face, $N_z = 2$', fontsize=15)
    plt.ylabel(r'$\mathcal{L}$', fontsize=15)
    plt.legend(fontsize=12)
    plt.savefig('Imgs/Training_loss.png', bbox_inches="tight")
    plt.show()
    
    grid_size = 10
    xx, yy = norm.ppf(np.meshgrid(np.linspace(0.1, .9, grid_size), np.linspace(0.1, .9, grid_size)))

    fig = plt.figure(figsize=(10, 14), constrained_layout=False)
    grid = fig.add_gridspec(grid_size, grid_size, wspace=0, hspace=0)

    for i in range(grid_size):
        for j in range(grid_size):
            img = decoder.get_mean_and_log_var(torch.tensor([[xx[i, j], yy[i, j]]], device=device, dtype=torch.float))
            ax = fig.add_subplot(grid[i, j])
            ax.imshow(np.clip(img[0].data.cpu().numpy().reshape(img_size[0], img_size[1]), 0, 1), cmap='gray', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig('Imgs/Learned_data_manifold.png', bbox_inches="tight") 
    plt.show()
