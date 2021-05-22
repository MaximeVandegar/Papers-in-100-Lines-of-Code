import torch
import torch.nn as nn
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt
torch.manual_seed(1)


class MLP(nn.Module):
    def __init__(self, input_dim=13, output_dim=1, hidden_dim=256):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SELU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.SELU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.SELU(),
                                    nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        log_ratio = self.layers(x)
        classifier_output = log_ratio.sigmoid()
        return classifier_output, log_ratio


class UniformPrior:

    @staticmethod
    def log_prob(x_batch):
        uniform = Uniform(torch.zeros(x_batch.shape[0], 5) + torch.tensor([-3.]),
                          torch.zeros(x_batch.shape[0], 5) + torch.tensor([3.]))
        return uniform.log_prob(x_batch).sum(1)

    @staticmethod
    def sample(size):
        uniform = Uniform(torch.zeros(size, 5) + torch.tensor([-3.]), torch.zeros(size, 5) + torch.tensor([3.]))
        return uniform.sample()


class MultivariateNormalTransitionDistribution:
    # Model p(y|x) as a multivariate normal gaussian distribution with mean x

    @staticmethod
    def log_prob(y_batch, x_batch):
        # Returns log p(y|x)
        m = MultivariateNormal(x_batch, torch.eye(x_batch.shape[1]))
        return m.log_prob(y_batch)

    @staticmethod
    def sample(x_batch):
        # Returns y ~ p(y|x)
        m = MultivariateNormal(x_batch, torch.eye(x_batch.shape[1]))
        y = m.sample()
        return y


class SLCPSimulator:

    @staticmethod
    def get_ground_truth_parameters():
        return torch.tensor([0.7, -2.9, -1.0, -0.9, 0.6])

    @staticmethod
    def simulate(theta, eps=1e-6):
        means = theta[:, :2]
        s1 = torch.pow(theta[:, 2], 2)
        s2 = torch.pow(theta[:, 3], 2)
        pho = torch.tanh(theta[:, 4])

        cov = torch.zeros(theta.shape[0], 2, 2) + eps
        cov[:, 0, 0] = torch.pow(s1, 2)
        cov[:, 0, 1] = pho * s1 * s2
        cov[:, 1, 0] = pho * s1 * s2
        cov[:, 1, 1] = torch.pow(s2, 2)
        normal = MultivariateNormal(means, cov)

        x = torch.zeros(theta.shape[0], 8)
        x[:, :2] = normal.sample()
        x[:, 2:4] = normal.sample()
        x[:, 4:6] = normal.sample()
        x[:, 6:] = normal.sample()
        return x


def algorithm1(simulator, prior, criterion=nn.BCELoss(), batch_size=256, nb_epochs=int(1e6 / 256), device='cpu'):
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training_loss = []

    for _ in tqdm(range(nb_epochs)):
        theta = prior.sample(batch_size)
        theta_prime = prior.sample(batch_size)
        x = simulator.simulate(theta)

        nn_input = torch.cat((torch.cat((theta, theta_prime)), torch.cat((x, x))), dim=1).to(device)
        target = torch.zeros(2 * batch_size, device=device)
        target[:batch_size] = 1.
        classifier_output, log_ratio = model(nn_input)
        loss = criterion(classifier_output.squeeze(-1), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())
    return training_loss, model


def likelihood_free_metropolis_hastings(prior, transition_distribution, ratio_estimator, observation_x,
                                        T=5000, eps=1e-15, thinning=10, num_chains=10):
    """
    Algorithm 2 from Appendix A + thinning & multiple chains
    :param T: number of samples per chain
    """
    theta_t = prior.sample(num_chains)
    samples = torch.empty((T * thinning, num_chains, theta_t.shape[1]))
    samples[0] = theta_t

    for t in tqdm(range(1, T * thinning)):
        theta_prime = transition_distribution.sample(theta_t)

        _, log_ratio = ratio_estimator(
            torch.cat((theta_t, observation_x.repeat(num_chains, observation_x.shape[0])), dim=1))
        _, log_ratio_prime = ratio_estimator(
            torch.cat((theta_prime, observation_x.repeat(num_chains, observation_x.shape[0])), dim=1))
        log_prior = prior.log_prob(theta_t)
        log_prior_prime = prior.log_prob(theta_prime)
        lambda_ = log_ratio_prime.squeeze() + log_prior_prime - (log_ratio.squeeze() + log_prior)
        q_theta_given_theta_prime = torch.exp(transition_distribution.log_prob(theta_t, theta_prime))
        q_theta_prime_given_theta = torch.exp(transition_distribution.log_prob(theta_prime, theta_t))
        pho = torch.exp(lambda_) * q_theta_given_theta_prime / (q_theta_prime_given_theta + eps)
        pho[pho > 1] = 1

        # Update theta with probability pho
        r = torch.rand(num_chains)
        update_condition = r < pho
        theta_t[update_condition] = theta_prime[update_condition]
        samples[t] = theta_t

    return samples[::thinning, :, :].reshape(-1, theta_t.shape[1])


def make_plot(samples, savepath, theta_star, fig_size=(8, 8)):
    fig = plt.figure(figsize=fig_size)
    for i in range(samples.shape[1]):
        for j in range(i + 1):
            ax = plt.subplot(samples.shape[1], samples.shape[1], i * samples.shape[1] + j + 1)
            if i == j:
                ax.hist(samples[:, i], bins=50, histtype='step', color='k')
                ax.axvline(theta_star[i])
            else:
                ax.scatter(samples[:, j], samples[:, i], c='k', alpha=0.015, s=.2)
                ax.set_ylim([-3.5, 3.5])
                ax.axvline(theta_star[j]); ax.axhline(theta_star[i])
            if i < samples.shape[1] - 1:
                ax.set_xticks([])
            ax.set_xlim([-3.5, 3.5])
            ax.set_yticks([])

    plt.savefig(savepath); plt.close()


simulator = SLCPSimulator()
loss, ratio_estimator = algorithm1(simulator, UniformPrior())
gt_parameters = simulator.get_ground_truth_parameters()
observation = simulator.simulate(gt_parameters.unsqueeze(0))
samples = likelihood_free_metropolis_hastings(UniformPrior(), MultivariateNormalTransitionDistribution(),
                                              ratio_estimator, observation, T=2000).data.cpu().numpy()
make_plot(samples, 'Imgs/posteriors_from_the_tractable_benchmark.png', gt_parameters)
