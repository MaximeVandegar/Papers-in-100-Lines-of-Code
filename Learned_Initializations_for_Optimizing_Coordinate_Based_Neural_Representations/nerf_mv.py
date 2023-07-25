import os
import json
import torch
import copy
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from typing import Callable
from imageio.v3 import imread
import matplotlib.pyplot as plt

class NerfModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super(NerfModel, self).__init__()

        self.net = nn.Sequential(nn.Linear(120, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 4))

    def forward(self, o):
        emb_x = torch.cat([torch.cat([torch.sin(o * (2 ** i)), torch.cos(o * (2 ** i))],
                                     dim=-1) for i in torch.linspace(0, 8, 20)], axis=-1)
        h = self.net(emb_x)
        c, sigma = torch.sigmoid(h[:, :3]), torch.relu(h[:, -1])
        return c, sigma

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat(
        (torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
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
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(
                           ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    colors, sigma = nerf_model(x.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(
        2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)

def load_data(data_path, json_path, train=True, N=25, H=128, W=128):
    gt_pixels = []
    rays_d = []
    rays_o = []

    with open(json_path, "r") as f:
        data = json.load(f)

    scenes = [data_path + f for f in sorted(data['train' if train else 'test'])]
    for scene_path in tqdm(scenes):
        transforms = scene_path + "/transforms.json"
        if os.path.isfile(transforms):
            with open(transforms, "r") as f:
                data = json.load(f)

            scene_gt_pixels = torch.zeros((N, H, W, 3))
            scene_rays_d = torch.zeros((N, H, W, 3))
            scene_rays_o = torch.zeros((N, H, W, 3))
            for view_idx in range(N):
                view = data["frames"][view_idx]
                img = torch.from_numpy(imread(
                    scene_path + f"/{view['file_path'].split('/')[-1]}.png") / 255.)
                c2w = torch.tensor(view["transform_matrix"])

                focal_length = W / 2. / torch.tan(
                    torch.tensor(data["camera_angle_x"]) / 2.)
                u, v = torch.meshgrid(torch.arange(W), torch.arange(H))
                dirs = torch.stack((v - W / 2, -(u - H / 2),
                                    - torch.ones_like(u) * focal_length), axis=-1)
                dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
                scene_rays_d[view_idx] = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
                scene_rays_o[view_idx] = torch.zeros_like(scene_rays_d[view_idx]) + c2w[:3, 3]
                scene_gt_pixels[view_idx] = img[..., :3] * img[..., -1:] + 1 - img[..., -1:]
            rays_d.append(scene_rays_d)
            rays_o.append(scene_rays_o)
            gt_pixels.append(scene_gt_pixels)
    return rays_o, rays_d, gt_pixels

@torch.no_grad()
def sample_task(rays_o, rays_d, gt_pixels):
    scene_idx = torch.randint(0, len(rays_o), (1,))
    o, d, gt = rays_o[scene_idx], rays_d[scene_idx], gt_pixels[scene_idx]
    return torch.cat([o.reshape(-1, 3), d.reshape(-1, 3), gt.reshape(-1, 3)], dim=-1)

def perform_k_training_steps(nerf_model, task, k, optimizer, batch_size=128,
                             device='cpu', hn=2., hf=6., nb_bins=128):
    for _ in (range(k)):
        indices = torch.randint(task.shape[0], size=[batch_size])
        batch = task[indices]
        ray_origins = batch[:, :3].to(device)
        ray_directions = batch[:, 3:6].to(device)
        ground_truth_px_values = batch[:, 6:].to(device)

        regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions,
                                            hn=hn, hf=hf, nb_bins=nb_bins)
        loss = nn.functional.mse_loss(ground_truth_px_values, regenerated_px_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return nerf_model.parameters()

def reptile(meta_model, meta_optim, nb_iterations: int, device: str, sample_task: Callable,
            perform_k_training_steps: Callable, k=32):

    for epoch in tqdm(range(nb_iterations)):
        task = sample_task()
        nerf_model =copy.deepcopy(meta_model)
        optimizer = torch.optim.SGD(nerf_model.parameters(), 0.5)
        phi_tilde = perform_k_training_steps(nerf_model, task, k, optimizer, device=device)

        # Update phi
        meta_optim.zero_grad()
        with torch.no_grad():
            for p, g in zip(meta_model.parameters(), phi_tilde):
                p.grad = p - g
        meta_optim.step()

if __name__ == "__main__":
    device = 'cuda'
    meta_model = NerfModel(hidden_dim=256).to(device)
    meta_optim = torch.optim.Adam(meta_model.parameters(), lr=5e-5)
    rays_o, rays_d, gt_pixels = load_data("data/cars/", "data/car_splits.json",
                                          train=True, N=25, H=128, W=128)
    reptile(meta_model, meta_optim, 100_000, device,
            lambda: sample_task(rays_o, rays_d, gt_pixels), perform_k_training_steps, 32)

    # Testing
    rays_o, rays_d, gt_pixels = load_data("data/cars/", "data/car_splits.json",
                                          train=False, N=25, H=128, W=128)
    plt.figure(figsize=(12, 12), dpi=300)
    for test_img in range(10):
        nerf_model = copy.deepcopy(meta_model)
        optimizer = torch.optim.SGD(nerf_model.parameters(), 0.5)

        test_data = torch.cat([rays_o[test_img][0].reshape(-1, 3),
                               rays_d[test_img][0].reshape(-1, 3),
                               gt_pixels[test_img][0].reshape(-1, 3)], dim=-1).to(device)

        training_loss = perform_k_training_steps(nerf_model, test_data, 1000,
                                                 optimizer, batch_size=128,
                                                 device=device)
        plt.subplot(10, 10, test_img * 10 + 1)
        plt.axis('off')
        plt.imshow(gt_pixels[test_img][0])
        if test_img == 0: plt.title('Input image')
        with torch.no_grad():
            for idx, i in enumerate(range(1, 25, 3)):
                img = render_rays(nerf_model,
                                  rays_o[test_img][i].to(device).reshape(-1, 3),
                                  rays_d[test_img][i].to(device).reshape(-1, 3),
                                  hn=2., hf=6., nb_bins=128)

                plt.subplot(10, 10, test_img * 10 + 3 + idx)
                plt.axis('off')
                plt.imshow(img.reshape(128, 128, 3).data.cpu().numpy().clip(0, 1))
                if (test_img == 0) and (idx == 3): plt.title('Novel views')
    plt.savefig(f'meta_mv.png', bbox_inches='tight')
    plt.show()