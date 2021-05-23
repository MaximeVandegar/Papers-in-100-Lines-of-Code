import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from UMNN import UMNNMAFFlow  # Normalizing Flow
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
torch.manual_seed(1)


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


class MultivariateNormalDistribution:
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


def sample_from_mcmc(prior, likelihood_function, observation_x, T=5000, thinning=10, num_chains=10,
                     transition_distribution=MultivariateNormalDistribution(), eps=1e-15, device='cpu'):
    theta_t = prior.sample(num_chains)
    samples = torch.empty((T * thinning, num_chains, theta_t.shape[1]))
    samples[0] = theta_t

    for t in tqdm(range(1, T * thinning), desc="Sampling from MCMC"):
        theta_prime = transition_distribution.sample(theta_t)

        p_x_given_theta = likelihood_function(observation_x.repeat(num_chains, 1).to(device),
                                              theta_t.to(device)).cpu()
        p_x_given_theta_prime = likelihood_function(observation_x.repeat(num_chains, 1).to(device),
                                                    theta_prime.to(device)).cpu()

        density_prior = torch.exp(prior.log_prob(theta_t))
        density_prior_prime = torch.exp(prior.log_prob(theta_prime))

        q_theta_given_theta_prime = torch.exp(transition_distribution.log_prob(theta_t, theta_prime))
        q_theta_prime_given_theta = torch.exp(transition_distribution.log_prob(theta_prime, theta_t))

        acceptance_probability = ((density_prior_prime * p_x_given_theta_prime * q_theta_prime_given_theta) / (
                density_prior * p_x_given_theta * q_theta_given_theta_prime + eps))
        acceptance_probability[acceptance_probability > 1] = 1

        # Update theta with some probability
        r = torch.rand(num_chains)
        update_condition = r < acceptance_probability
        theta_t[update_condition] = theta_prime[update_condition]
        samples[t] = theta_t

    return samples[::thinning, :, :].reshape(-1, theta_t.shape[1])


def train(estimator, dataset, theta_dim=5, batch_size=100, nb_epochs=1000):  # @Todo : improve. E.g. early stopping, ...

    optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)
    for _ in tqdm(range(nb_epochs), desc="Training"):
        batch_idx = torch.randperm(dataset.shape[0])[:batch_size]
        theta = dataset[batch_idx, :theta_dim].to(device)
        x = dataset[batch_idx, theta_dim:].to(device)
        ll, z = estimator.compute_ll(x, context=theta)

        optimizer.zero_grad()
        loss = - torch.mean(ll)
        loss.backward()
        optimizer.step()


def snl(x_0, estimator, prior, simulator, R=10, N=1000, num_chains=10, device='cpu'):
    likelihood_estimator = lambda x, theta: torch.exp(estimator.compute_ll(x, context=theta)[0])

    dataset = torch.tensor([])
    for r in range(R):
        theta_n = sample_from_mcmc(prior, likelihood_estimator, x_0, T=int(N / num_chains), device=device,
                                   num_chains=num_chains) if r > 0 else prior.sample(N)
        x_n = simulator.simulate(theta_n)
        dataset = torch.cat((dataset, torch.cat((theta_n, x_n), dim=1)))
        train(estimator, dataset, theta_dim=theta_n.shape[1], nb_epochs=1000)


def make_plot(samples, save_path, theta_star, fig_size=(8, 8)):

    plt.figure(figsize=fig_size)
    labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$', r'$\theta_5$']
    for i in range(samples.shape[1]):
        for j in range(i, samples.shape[1]):
            ax = plt.subplot(samples.shape[1], samples.shape[1], i * samples.shape[1] + j + 1)
            if i == j:
                ax.hist(samples[:, i], bins=25, histtype='stepfilled', color='k')
                ax.axvline(theta_star[i], c='r')
                plt.xlabel(labels[i], fontsize=17)
            else:
                ax.scatter(samples[:, j], samples[:, i], c='k', alpha=0.2, s=.2)
                plt.scatter(theta_star[j], theta_star[i], c='r', marker='*', s=100)
                ax.set_ylim([-3.5, 3.5])

            ax.set_xlim([-3.5, 3.5])
            ax.set_yticks([]); ax.set_xticks([])
    plt.savefig(save_path); plt.close()


if __name__ == "__main__":
    device = 'cuda'
    simulator = SLCPSimulator()
    prior_distribution = UniformPrior()
    gt_parameters = simulator.get_ground_truth_parameters()
    observation = simulator.simulate(gt_parameters.unsqueeze(0)).squeeze(0)
    model = UMNNMAFFlow(nb_flow=6, nb_in=8, cond_in=5, hidden_derivative=[75, 75, 75], hidden_embedding=[75, 75, 75],
                        embedding_s=10, nb_steps=20, device=device)

    snl(observation, model, prior_distribution, simulator, device=device)
    likelihood_estimator = lambda x, theta: torch.exp(model.compute_ll(x, context=theta)[0])
    posterior_samples = sample_from_mcmc(prior_distribution, likelihood_estimator, observation, T=500, thinning=30,
                                         num_chains=10, device=device)
    make_plot(posterior_samples.data.cpu().numpy(), "Imgs/posterior_samples.png", gt_parameters.data.numpy())
