import os
import json
import math
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class NMRDataset(Dataset):
    def __init__(self, data_path, json_path, train=True, n_views=24, H=64, W=64):
        self.n_views, self.H, self.W = n_views, H, W
        with open(json_path, "r") as f:
            split = json.load(f)
        scenes = [os.path.join(data_path, f)
                  for f in sorted(split["train" if train else "test"])]

        gt_pixels, c2ws, intrinsics = [], [], []
        for scene_path in tqdm(scenes, desc="loading scenes"):
            cam = np.load(os.path.join(scene_path, "cameras.npz"))
            s_px = torch.zeros((n_views, H, W, 3))
            s_c2w = torch.zeros((n_views, 4, 4))
            s_K = torch.zeros((n_views, 4, 4))
            for v in range(n_views):
                img = np.array(Image.open(
                    os.path.join(scene_path, "image", f"{v:04d}.png")).convert("RGB"))
                s_px[v] = torch.from_numpy(img).float() / 255.0
                s_c2w[v] = torch.from_numpy(cam[f"world_mat_inv_{v}"]).float()
                s_K[v] = torch.from_numpy(cam[f"camera_mat_{v}"]).float()
            gt_pixels.append(s_px)
            c2ws.append(s_c2w)
            intrinsics.append(s_K)

        self.gt_pixels = torch.stack(gt_pixels)  # [B, N, H, W, 3]
        self.c2ws = torch.stack(c2ws)  # [B, N, 4, 4]
        self.intrinsics = torch.stack(intrinsics)  # [B, N, 4, 4]

    def __len__(self):
        return self.gt_pixels.shape[0]

    def __getitem__(self, i):
        v = random.randrange(self.n_views)
        return {"scene_idx": i, "img": self.gt_pixels[i, v].permute(2, 0, 1),
                "c2w": self.c2ws[i, v], "K": self.intrinsics[i, v]}


def intrinsics_to_fxfycxcy(K, H, W):
    s = float(K[0, 0])
    return s * W / 2.0, s * H / 2.0, W / 2.0, H / 2.0


def plucker_rays(c2w, K, H, W, device):  # Paper Eq. 4–5
    fx, fy, cx, cy = K
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32),
                            indexing="ij")
    dirs_cam = torch.stack([(xs + 0.5 - cx) / fx, (ys + 0.5 - cy) / fy,
                            torch.ones_like(xs)], dim=-1)
    R, o = c2w[:3, :3], c2w[:3, 3]
    d = F.normalize(dirs_cam @ R.t(), dim=-1)  # [H, W, 3]
    m = torch.cross(o.expand_as(d), d, dim=-1)  # [H, W, 3]
    return torch.cat([d, m], dim=-1).reshape(-1, 6)  # [H*W, 6]


def lfn_forward(rays, weights):
    """Functional LFN: 6-layer MLP, LayerNorm (no affine) before each non-final layer (paper §2.2).
    rays:  [B, N, 6]
    weights: list of (W, b) with W: [B, out, in] and b: [B, out], one tuple per layer.
    """
    x = rays
    n = len(weights)
    for i, (W, b) in enumerate(weights):
        x = torch.einsum("bni,boi->bno", x, W) + b.unsqueeze(1)
        if i < n - 1:
            x = F.layer_norm(x, x.shape[-1:])
            x = F.relu(x)
    return torch.sigmoid(x)


class HyperNet(nn.Module):
    LFN_SPECS = [(6, 256), (256, 256), (256, 256), (256, 256), (256, 256), (256, 3)]

    def __init__(self, latent_dim=256, hidden=256):
        super().__init__()
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.heads = nn.ModuleList(nn.Linear(hidden, fan_in * fan_out + fan_out)
                                   for fan_in, fan_out in self.LFN_SPECS)
        # Init so that at z = 0 the generated LFN is a normally-initialized MLP: the bias
        # carries nn.Linear's own init, the weight is shrunk so z only perturbs it.
        with torch.no_grad():
            for head, (fan_in, _) in zip(self.heads, self.LFN_SPECS):
                bound = 1.0 / math.sqrt(fan_in)
                head.bias.uniform_(-bound, bound)
                head.weight.mul_(1e-2)

    def forward(self, z):
        h = self.latent_encoder(z)
        out = []
        for head, (fan_in, fan_out) in zip(self.heads, self.LFN_SPECS):
            wb = head(h)  # [B, fan_in * fan_out + fan_out]
            W = wb[..., :fan_in * fan_out].reshape(*wb.shape[:-1], fan_out, fan_in)
            b = wb[..., fan_in * fan_out:]
            out.append((W, b))
        return out


def invert_latent(hypernet, src_rays, src_rgb, n_iters=200, lr=1e-2, lam_lat=1e2):
    """Test-time auto-decoding (paper Eq. 12). Freeze Ψ, optimize z from zero."""
    for p in hypernet.parameters():
        p.requires_grad_(False)
    z = torch.zeros(1, 256, device=src_rays.device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(n_iters):
        weights = hypernet(z)
        pred = lfn_forward(src_rays.unsqueeze(0), weights)[0]
        loss = ((pred - src_rgb) ** 2).sum() + lam_lat * (z ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in hypernet.parameters():
        p.requires_grad_(True)
    return z.detach()


@torch.no_grad()
def render_view(hypernet, z, c2w, K, H, W, chunk=4096):
    rays = plucker_rays(c2w, K, H, W, z.device)
    weights = hypernet(z)
    out = [lfn_forward(rays[i:i + chunk].unsqueeze(0), weights)[0]
           for i in range(0, rays.shape[0], chunk)]
    return torch.cat(out, 0).reshape(H, W, 3)


def render_novel_view_grid(hypernet, dataset, H, W, device, save_path, num_scenes=10,
                           n_invert_iters=200, lr_invert=1e-2, lam_lat=1e2):
    hypernet.eval()
    view_idx = [1, 2, 4, 7, 10, 13, 16, 19, 22, 23]
    fig, axes = plt.subplots(num_scenes, len(view_idx),  dpi=300,
                             figsize=(2.2 * len(view_idx), 2.2 * num_scenes), squeeze=False)
    for s in range(num_scenes):
        src = view_idx[s]
        src_K = intrinsics_to_fxfycxcy(dataset.intrinsics[s, src], H, W)
        src_c2w = dataset.c2ws[s, src].to(device)
        src_rays = plucker_rays(src_c2w, src_K, H, W, device)
        src_rgb = dataset.gt_pixels[s, src].to(device).reshape(-1, 3)
        z = invert_latent(hypernet, src_rays, src_rgb, n_invert_iters, lr_invert, lam_lat)
        axes[s, 0].axis("off")
        axes[s, 0].imshow(dataset.gt_pixels[s, src].numpy().clip(0, 1))
        if s == 0:
            axes[s, 0].set_title("Input image", fontsize=25)
        for col, v in enumerate([i for i in view_idx if i != src], start=1):
            tgt_K = intrinsics_to_fxfycxcy(dataset.intrinsics[s, v], H, W)
            img = render_view(hypernet, z, dataset.c2ws[s, v].to(device), tgt_K, H, W)
            axes[s, col].axis("off")
            axes[s, col].imshow(img.cpu().numpy().clip(0, 1))
            if s == 0 and col == len(view_idx) // 2:
                axes[s, col].set_title("Novel views", fontsize=25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    hypernet.train()


if __name__ == "__main__":
    data_root = "NMR_Dataset/02958343/"
    split_json = "car_splits.json"
    device = "cuda"
    H = W = 64
    batch_size, n_rays = 32, 512
    lam_lat = 1e2

    train_set = NMRDataset(data_root, split_json, train=True, H=H, W=W)
    test_set = NMRDataset(data_root, split_json, train=False, H=H, W=W)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=True, pin_memory=torch.cuda.is_available())

    latents = nn.Embedding(len(train_set), 256).to(device)
    nn.init.zeros_(latents.weight)
    hypernet = HyperNet().to(device)
    optimizer = torch.optim.Adam(list(hypernet.parameters()) + list(latents.parameters()), lr=1e-4)

    train_iter = iter(train_loader)
    for step in tqdm(range(1, 200_001)):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        z = latents(batch["scene_idx"].to(device))  # [B, 256]
        weights = hypernet(z)
        B = z.shape[0]

        rays_b, rgb_b = [], []
        for b in range(B):
            K = intrinsics_to_fxfycxcy(batch["K"][b], H, W)
            rays = plucker_rays(batch["c2w"][b].to(device), K, H, W, device)
            rgb = batch["img"][b].permute(1, 2, 0).reshape(-1, 3).to(device)
            idx = torch.randint(0, H * W, (n_rays,), device=device)
            rays_b.append(rays[idx])
            rgb_b.append(rgb[idx])
        rays_b = torch.stack(rays_b)  # [B, n_rays, 6]
        rgb_b = torch.stack(rgb_b)   # [B, n_rays, 3]

        pred = lfn_forward(rays_b, weights)  # [B, n_rays, 3]
        recon = ((pred - rgb_b) ** 2).sum() * (H * W) / (B * n_rays)
        reg = (z ** 2).sum() / B
        loss = recon + lam_lat * reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    render_novel_view_grid(hypernet, test_set, H, W, device, save_path="Imgs/lfn.png")
