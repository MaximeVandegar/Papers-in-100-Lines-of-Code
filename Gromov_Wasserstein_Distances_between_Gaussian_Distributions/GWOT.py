import torch
import matplotlib.pyplot as plt


def symsqrt(matrix):
    """
    Compute the square root of a positive definite matrix.
    Retrieved from https://github.com/pytorch/pytorch/issues/25481#issuecomment-544465798.
    """
    _, s, v = matrix.svd()
    # truncate small components
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    # compose the square root matrix
    square_root = (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)
    return square_root


def TW2(x, m0, m1, cov0, cov1):
    """
    Eq. 1.5
    """
    a = torch.inverse(symsqrt(cov0))
    b = symsqrt(torch.mm(symsqrt(cov0), torch.mm(cov1, symsqrt(cov0))))
    c = torch.inverse(symsqrt(cov0))
    return m1 + torch.mm(torch.mm(a, torch.mm(b, c)), (x - m0).unsqueeze(-1)).squeeze(-1)


class Distribution:

    def __init__(self, mean: torch.tensor, cov: torch.tensor):
        self.mean = mean
        self.cov = cov
        self.m = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)

    def sample(self, size):
        return self.m.sample(size)


if __name__ == '__main__':
    target_means = [[0., 0.], [0., 0.], [0., 0.]]
    target_covariances = [[[4.1, 4.],
                           [4., 4.1]],
                          [[6.1, -2.],
                           [-2., 3.1]],
                          [[15, .1],
                           [.1, 1.]]]
    init_means = [[0., 0.], [0., 0.], [0., 0.]]
    init_covariances = [[[1.1, .2],
                         [.2, 1.1]],
                        [[2.1, 2.],
                         [2., 2.1]],
                        [[.5, .2],
                         [.2, 4.5]]]

    cmap = plt.cm.get_cmap('viridis', 500)
    colors = []
    for i in range(500):
        colors.append(cmap(i))

    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    fig_index = 0
    for target_mean, target_cov, init_mean, init_cov in zip(target_means, target_covariances, init_means,
                                                            init_covariances):

        target_distribution = Distribution(torch.tensor(target_mean), torch.tensor(target_cov))
        init_distribution = Distribution(torch.tensor(init_mean), torch.tensor(init_cov))

        xt = target_distribution.sample([500])
        x0 = init_distribution.sample([500])
        mask = torch.argsort(x0[:, 1])

        axes[fig_index, 0].scatter(xt[:, 0].numpy(), xt[:, 1].numpy(), c='k', alpha=.3, s=10)
        axes[fig_index, 0].scatter(x0[mask, 0].numpy(), x0[mask, 1].numpy(), c=colors, s=10)
        axes[fig_index, 0].set_xlim([-10, 10])
        axes[fig_index, 0].set_ylim([-10, 10])

        data = []
        for x in x0:
            data.append(TW2(x, init_distribution.mean, target_distribution.mean, init_distribution.cov,
                            target_distribution.cov).reshape(1, 2))
        data = torch.cat(data, dim=0).numpy()
        axes[fig_index, 1].scatter(xt[:, 0].numpy(), xt[:, 1].numpy(), c='k', alpha=.3, s=10)
        axes[fig_index, 1].scatter(data[mask, 0], data[mask, 1], c=colors, s=10)
        axes[fig_index, 1].set_ylim([-10, 10])
        axes[fig_index, 1].set_xlim([-10, 10])
        fig_index += 1
    plt.savefig('Imgs/TW2.png')
    plt.show()
