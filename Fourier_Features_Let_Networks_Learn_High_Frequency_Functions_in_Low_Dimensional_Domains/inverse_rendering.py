import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


@torch.no_grad()
def test(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)

        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img.clip(0, 1))
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()


class NerfModel(nn.Module):
    def __init__(self, NB_features=256, hidden_dim=128):
        super(NerfModel, self).__init__()

        self.net = nn.Sequential(nn.Linear(NB_features * 2, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 4))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.B = torch.randn((3, NB_features), device=device) * 2 * np.pi * 6.05

    def positional_encoding(self, x):
        return torch.cat((torch.cos(x @ self.B), torch.sin(x @ self.B)), dim=-1)

    def forward(self, o):
        emb_x = self.positional_encoding(o)
        out = self.net(emb_x)
        c, sigma = self.sigmoid(out[:, :-1]), self.relu(out[:, -1])
        return c, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
    colors, sigma = nerf_model(x.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    # Regularization for white background
    weight_sum = weights.sum(-1).sum(-1)
    return c + 1 - weight_sum.unsqueeze(-1)


def train(nerf_model, optimizer, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(5e4),
          nb_bins=192, H=400, W=400):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):

        batch = next(iter(data_loader))
        ray_origins = batch[:, :3].to(device)
        ray_directions = batch[:, 3:6].to(device)
        ground_truth_px_values = batch[:, 6:].to(device)

        regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
        loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())

        if ((epoch % 5000) == 0) and (epoch > 0):
            for img_index in range(200):
                test(hn, hf, testing_dataset, img_index=img_index, nb_bins=nb_bins, H=H, W=W)

    return training_loss


if __name__ == 'main':
    device = 'cuda'
    training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
    model = NerfModel(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    train(model, model_optimizer, data_loader, device=device, hn=2, hf=6, nb_bins=192, H=400, W=400)
