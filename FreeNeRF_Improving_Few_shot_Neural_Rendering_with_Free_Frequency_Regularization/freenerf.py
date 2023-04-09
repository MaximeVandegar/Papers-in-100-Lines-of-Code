import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt


@torch.no_grad()
def test(hn, hf, dataset, step, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, step, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{img_index}_v1.png', bbox_inches='tight')
    plt.close()


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=16, embedding_dim_direction=4, hidden_dim=128, T=40_000):
        super(NerfModel, self).__init__()

        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), )
        self.block3 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2),
                                    nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(), )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()
        self.T = T

    def positional_encoding(self, x, L, step, is_pos=False):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        out = torch.cat(out, dim=1)

        Lmax = 2 * 3 * L + 3
        if is_pos:
            out[:, int(step / self.T * Lmax) + 3:] = 0.
        return out

    def forward(self, o, d, step):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos, step, is_pos=True)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction, step, is_pos=False)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], torch.nn.functional.softplus(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)
        return c, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, step, hn=0, hf=0.5, nb_bins=192):
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
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3), step)
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)


def sample_batch(data, batch_size, device):
    idx = torch.randperm(data.shape[0])[:batch_size]
    return torch.from_numpy(data[idx]).to(device)


def train(nerf_model, optimizer, training_data, nb_epochs, batch_size, device='cpu', hn=0, hf=1, nb_bins=192):
    training_loss = []
    for step in tqdm(range(nb_epochs)):
        batch = sample_batch(training_data, batch_size, device)
        rays_o = batch[:, :3].to(device)
        rays_d = batch[:, 3:6].to(device)
        ground_truth_px_values = batch[:, 6:].to(device)

        regenerated_px_values = render_rays(nerf_model, rays_o, rays_d, step, hn=hn, hf=hf, nb_bins=nb_bins)
        loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
        training_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return training_loss


if __name__ == 'main':
    device = 'cuda'
    training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
    nb_epochs = 80_000

    img0 = training_dataset[26 * 400 * 400:(26 + 1) * 400 * 400]
    img2 = training_dataset[86 * 400 * 400:(86 + 1) * 400 * 400]
    img3 = training_dataset[2 * 400 * 400:(2 + 1) * 400 * 400]
    img4 = training_dataset[55 * 400 * 400:(55 + 1) * 400 * 400]
    img5 = training_dataset[75 * 400 * 400:(75 + 1) * 400 * 400]
    img6 = training_dataset[93 * 400 * 400:(93 + 1) * 400 * 400]
    img7 = training_dataset[16 * 400 * 400:(16 + 1) * 400 * 400]
    img8 = training_dataset[73 * 400 * 400:(73 + 1) * 400 * 400]

    training_data = np.concatenate((img0, img2, img3, img4, img5, img6, img7, img8))
    model = NerfModel(hidden_dim=256, T=nb_epochs // 2).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    train(model, model_optimizer, training_data, nb_epochs, 1024, device=device, hn=2, hf=6, nb_bins=192)
    for img_index in range(200):
        test(2, 6, testing_dataset, nb_epochs, img_index=img_index, nb_bins=192, H=400, W=400)
