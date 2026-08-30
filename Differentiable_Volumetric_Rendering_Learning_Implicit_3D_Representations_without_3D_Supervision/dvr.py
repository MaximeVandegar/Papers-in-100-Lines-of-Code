import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18


class NMRDataset(Dataset):
    def __init__(self, data_path, json_path, train=True, n_views=24, H=64, W=64):
        self.n_views, self.H, self.W = n_views, H, W
        with open(json_path, "r") as f:
            split = json.load(f)
        scenes = [os.path.join(data_path, f)
                  for f in sorted(split["train" if train else "test"])]

        gt_pixels, masks, c2ws, intrinsics = [], [], [], []
        for scene_path in tqdm(scenes, desc="loading scenes"):
            cam = np.load(os.path.join(scene_path, "cameras.npz"))
            s_px = torch.zeros((n_views, H, W, 3))
            s_mask = torch.zeros((n_views, H, W))
            s_c2w = torch.zeros((n_views, 4, 4))
            s_K = torch.zeros((n_views, 4, 4))
            for v in range(n_views):
                img = np.array(Image.open(
                    os.path.join(scene_path, "image", f"{v:04d}.png")).convert("RGB"))
                m = np.array(Image.open(
                    os.path.join(scene_path, "mask", f"{v:04d}.png")).convert("L"))
                s_px[v] = torch.from_numpy(img).float() / 255.0
                s_mask[v] = torch.from_numpy(m).float() / 255.0
                s_c2w[v] = torch.from_numpy(cam[f"world_mat_inv_{v}"]).float()
                s_K[v] = torch.from_numpy(cam[f"camera_mat_{v}"]).float()
            gt_pixels.append(s_px)
            masks.append(s_mask)
            c2ws.append(s_c2w)
            intrinsics.append(s_K)

        self.gt_pixels = torch.stack(gt_pixels)  # [B, N, H, W, 3]
        self.masks = torch.stack(masks)  # [B, N, H, W]
        self.c2ws = torch.stack(c2ws)  # [B, N, 4, 4]
        self.intrinsics = torch.stack(intrinsics)  # [B, N, 4, 4]

    def __len__(self):
        return self.gt_pixels.shape[0]

    def __getitem__(self, i):
        src, tgt = random.sample(range(self.n_views), 2)
        return {"src_img": self.gt_pixels[i, src].permute(2, 0, 1),
                "tgt_img": self.gt_pixels[i, tgt].permute(2, 0, 1),
                "tgt_mask": self.masks[i, tgt],
                "tgt_c2w": self.c2ws[i, tgt],
                "tgt_cam": self.intrinsics[i, tgt]}


def intrinsics_to_fxfycxcy(camera_mat, H, W):
    s = float(camera_mat[0, 0])
    return s * W / 2.0, s * H / 2.0, W / 2.0, H / 2.0


def get_rays(c2w, K, H, W, device):
    fx, fy, cx, cy = K
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32),
                            indexing="ij")
    dirs = torch.stack([(xs + 0.5 - cx) / fx, (ys + 0.5 - cy) / fy,
                        torch.ones_like(xs)], dim=-1)  # camera frame, +z forward
    rays_d = F.normalize(dirs @ c2w[:3, :3].t(), dim=-1)
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def positional_encoding(x, n_freqs=6):
    freqs = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)
    enc = torch.cat([torch.sin(x[..., None] * freqs), torch.cos(x[..., None] * freqs)], -1)
    return torch.cat([x, enc.reshape(*x.shape[:-1], -1)], dim=-1)


class ImageEncoder(nn.Module):
    def __init__(self, latent=256):
        super().__init__()
        net = resnet18(weights="IMAGENET1K_V1")
        net.fc = nn.Linear(512, latent)
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class OccupancyAndTexture(nn.Module):

    def __init__(self, latent=256, n_freqs=6, hidden=256, radius=0.5):
        super().__init__()
        self.n_freqs, self.radius = n_freqs, radius

        self.occ_net = nn.Sequential(nn.Linear(3 + latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))
        self.tex_net = nn.Sequential(nn.Linear(3 + 3 * 2 * n_freqs + latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 3))
        # Zeroing the last layer makes occ_net output 0, so occ() returns exactly `radius - |p|`:
        # the signed distance from p to a sphere of radius 0.5 centred on the origin.
        nn.init.zeros_(self.occ_net[-1].weight)
        nn.init.zeros_(self.occ_net[-1].bias)

    def cat(self, x, z):
        return torch.cat([x, z.expand(*x.shape[:-1], z.shape[-1])], dim=-1)

    def occ(self, p, z):
        return self.occ_net(self.cat(p, z)).squeeze(-1) + self.radius - p.norm(dim=-1)

    def rgb(self, p, z):
        return torch.sigmoid(self.tex_net(self.cat(positional_encoding(p, self.n_freqs), z)))


def marching_steps(step, init=16, milestones=(50_000, 100_000, 250_000)):
    """March coarsely at first and double the resolution at each milestone."""
    return init * 2 ** sum(step >= m for m in milestones)


@torch.no_grad()
def find_surface(model, z, rays_o, rays_d, near, far, n_steps=128, n_secant=8):
    """March the ray, keep the first outside->inside crossing, refine it by secant."""
    t = torch.linspace(near, far, n_steps, device=rays_o.device)
    pts = rays_o[:, None, :] + t[None, :, None] * rays_d[:, None, :]
    f = model.occ(pts, z)  # [R, S]

    crossing = (f[:, :-1] < 0) & (f[:, 1:] > 0)
    hit = crossing.any(dim=1)
    first = torch.argmax(crossing.float(), dim=1)  # [R]
    t_lo, t_hi = t[first], t[first + 1]
    f_lo = torch.gather(f, 1, first[:, None]).squeeze(1)
    f_hi = torch.gather(f, 1, (first + 1)[:, None]).squeeze(1)

    for _ in range(n_secant):
        t_mid = t_lo - f_lo * (t_hi - t_lo) / (f_hi - f_lo).clamp(min=1e-6)
        f_mid = model.occ(rays_o + t_mid[:, None] * rays_d, z)
        neg = f_mid < 0
        t_lo, f_lo = torch.where(neg, t_mid, t_lo), torch.where(neg, f_mid, f_lo)
        t_hi, f_hi = torch.where(neg, t_hi, t_mid), torch.where(neg, f_hi, f_mid)
    return hit, t_lo - f_lo * (t_hi - t_lo) / (f_hi - f_lo).clamp(min=1e-6)


def attach_gradient(model, z, p, rays_d):
    """The ray march is not differentiable, so the surface point has no gradient. The hack
    below: everything is detached except f(p), so autograd differentiates that one term and
    returns Eq. 6, from implicit differentiation of f(p). f(p) is ~0, so p barely moves."""
    p = p.detach().requires_grad_(True)
    grad_p = torch.autograd.grad(model.occ(p, z).sum(), p, create_graph=False)[0]
    denom = (grad_p * rays_d).sum(-1).detach()
    denom = torch.where(denom.abs() < 1e-4, torch.full_like(denom, 1e-4), denom)
    return p.detach() - rays_d * (model.occ(p.detach(), z) / denom)[:, None]


def normal_loss(model, z, p, noise=0.01):
    """Two nearby points should have nearly the same normal; penalising the difference is
    what keeps the surface smooth. The normal is grad f normalised."""
    neighbour = p + noise * torch.randn_like(p)
    points = torch.cat([p, neighbour], dim=0).requires_grad_(True)
    grad_f = torch.autograd.grad(model.occ(points, z).sum(), points, create_graph=True)[0]
    normal, normal_neighbour = F.normalize(grad_f, dim=-1).chunk(2, dim=0)
    return (normal - normal_neighbour).norm(dim=-1).sum()


def sample_along_ray(rays_o, rays_d, near, far):
    depth = near + (far - near) * torch.rand(rays_o.shape[0], 1, device=rays_o.device)
    return rays_o + depth * rays_d


def compute_losses(model, z, rays_o, rays_d, target, mask, near, far, n_steps=128,
                   lambda_mask=1.0, lambda_normal=0.05):
    """Foreground and background pixels get different losses. Background: the ray must be
    empty. Foreground: if the ray hit the surface, match the colour there; if it missed,
    push occupancy up so that a surface appears."""
    hit, t_hit = find_surface(model, z, rays_o, rays_d, near, far, n_steps=n_steps)
    n_rays, terms = rays_o.shape[0], {}

    foreground_hit = hit & (mask > 0.5)
    if foreground_hit.any():
        p = rays_o[foreground_hit] + t_hit[foreground_hit][:, None] * rays_d[foreground_hit]
        p = attach_gradient(model, z, p, rays_d[foreground_hit])
        terms["rgb"] = (model.rgb(p, z) - target[foreground_hit]).abs().sum() / n_rays
        terms["normal"] = lambda_normal * normal_loss(model, z, p.detach()) / n_rays

    background = mask < 0.5
    if background.any():
        p = sample_along_ray(rays_o[background], rays_d[background], near, far)
        logits = model.occ(p, z)
        terms["freespace"] = lambda_mask * F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction="sum") / n_rays

    foreground_missed = (~hit) & (mask > 0.5)
    if foreground_missed.any():
        p = sample_along_ray(rays_o[foreground_missed], rays_d[foreground_missed], near, far)
        logits = model.occ(p, z)
        terms["occupancy"] = lambda_mask * F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits), reduction="sum") / n_rays
    return terms


@torch.no_grad()
def render_image(model, z, c2w, K, H, W, near, far, n_steps=1024, point_budget=2 ** 19):
    rays_o, rays_d = get_rays(c2w, K, H, W, z.device)
    img = torch.ones((H * W, 3), device=z.device)
    chunk = max(64, point_budget // n_steps)
    for i in range(0, rays_o.shape[0], chunk):
        o, d = rays_o[i:i + chunk], rays_d[i:i + chunk]
        hit, t_hit = find_surface(model, z, o, d, near, far, n_steps=n_steps)
        if hit.any():
            p = o[hit] + t_hit[hit][:, None] * d[hit]
            img[i:i + chunk][hit] = model.rgb(p, z)
    return img.reshape(H, W, 3)


@torch.no_grad()
def render_novel_view_grid(encoder, model, dataset, near, far, device,
                           save_path, num_scenes=10, res=256, n_steps=1024):
    view_idx = [1, 2, 4, 7, 10, 13, 16, 19, 22, 23]
    fig, axes = plt.subplots(num_scenes, len(view_idx), dpi=100,
                             figsize=(2.2 * len(view_idx), 2.2 * num_scenes), squeeze=False)
    for s in range(num_scenes):
        src = view_idx[s % len(view_idx)]
        z = encoder(dataset.gt_pixels[s, src].permute(2, 0, 1)[None].to(device))
        axes[s, 0].axis("off")
        axes[s, 0].imshow(dataset.gt_pixels[s, src].numpy().clip(0, 1))
        if s == 0:
            axes[s, 0].set_title("Input image", fontsize=25)
        for col, v in enumerate([i for i in view_idx if i != src], start=1):
            K = intrinsics_to_fxfycxcy(dataset.intrinsics[s, v], res, res)
            img = render_image(model, z, dataset.c2ws[s, v].to(device), K, res, res,
                               near, far, n_steps=n_steps)
            axes[s, col].axis("off")
            axes[s, col].imshow(img.cpu().numpy().clip(0, 1))
            if s == 0 and col == len(view_idx) // 2:
                axes[s, col].set_title("Novel views", fontsize=25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data_root = "NMR_Dataset/02958343/"
    split_json = "car_splits.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = W = 64
    near, far = 2.0, 3.5
    batch_size, n_rays = 64, 1024

    train_set = NMRDataset(data_root, split_json, train=True, H=H, W=W)
    test_set = NMRDataset(data_root, split_json, train=False, H=H, W=W)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=True, pin_memory=torch.cuda.is_available())

    encoder = ImageEncoder().to(device)
    model = OccupancyAndTexture().to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150_000, 225_000],
                                                     gamma=0.5)

    train_iter = iter(train_loader)
    for step in tqdm(range(1, 300_001)):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        latents = encoder(batch["src_img"].to(device))
        n_steps = marching_steps(step)
        loss = torch.zeros((), device=device)
        for b in range(batch_size):
            K = intrinsics_to_fxfycxcy(batch["tgt_cam"][b], H, W)
            rays_o, rays_d = get_rays(batch["tgt_c2w"][b].to(device), K, H, W, device)
            idx = torch.randint(0, H * W, (n_rays,), device=device)
            target = batch["tgt_img"][b].permute(1, 2, 0).reshape(-1, 3).to(device)[idx]
            mask = batch["tgt_mask"][b].reshape(-1).to(device)[idx]
            terms = compute_losses(model, latents[b:b + 1], rays_o[idx], rays_d[idx],
                                   target, mask, near, far, n_steps=n_steps)
            loss = loss + sum(terms.values())
        loss = loss / batch_size

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    encoder.eval()
    model.eval()
    render_novel_view_grid(encoder, model, test_set, near, far, device, save_path="Imgs/dvr.png")
