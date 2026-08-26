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
from torchvision.models import resnet34


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
        self.c2ws = torch.stack(c2ws)   # [B, N, 4, 4]
        self.intrinsics = torch.stack(intrinsics)  # [B, N, 4, 4]

    def __len__(self):
        return self.gt_pixels.shape[0]

    def __getitem__(self, i):
        src, tgt = random.sample(range(self.n_views), 2)
        return {"src_img": self.gt_pixels[i, src].permute(2, 0, 1),
                "tgt_img": self.gt_pixels[i, tgt].permute(2, 0, 1),
                "source_c2w": self.c2ws[i, src], "target_c2w": self.c2ws[i, tgt],
                "source_cam": self.intrinsics[i, src], "target_cam": self.intrinsics[i, tgt]}


def intrinsics_to_fxfycxcy(camera_mat, H, W):
    s = float(camera_mat[0, 0])
    return s * W / 2.0, s * H / 2.0, W / 2.0, H / 2.0


class ImageEncoder(nn.Module):
    """ResNet34 backbone giving a multi-scale feature map, one vector per pixel."""
    def __init__(self):
        super().__init__()
        net = resnet34(weights="IMAGENET1K_V1")
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu)  # 64 ch at 32x32
        self.layer1 = net.layer1  # 64 ch at 32x32
        self.layer2 = net.layer2  # 128 ch at 16x16
        self.layer3 = net.layer3  # 256 ch at 8x8
        self.feat_dim = 64 + 64 + 128 + 256
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        H, W = x.shape[-2:]
        x = (x - self.mean) / self.std
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        feats = [F.interpolate(f, size=(H, W), mode="bilinear", align_corners=True)
                 for f in (f0, f1, f2, f3)]
        return torch.cat(feats, dim=1)  # [B, feat_dim, H, W]


def positional_encoding(x, n_freqs=6):
    freqs = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)
    xb = x[..., None] * freqs  # [..., 3, n_freqs]
    enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
    return torch.cat([x, enc.reshape(*x.shape[:-1], -1)], dim=-1)


class PixelNeRF(nn.Module):
    def __init__(self, feat_dim, n_freqs=6, hidden=512, n_blocks=5):
        super().__init__()
        self.n_freqs = n_freqs
        self.inp = nn.Linear(3 + 3 * 2 * n_freqs + 3, hidden)
        self.feat_proj = nn.ModuleList(nn.Linear(feat_dim, hidden) for _ in range(n_blocks))
        self.blocks = nn.ModuleList(nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden),
                                                  nn.ReLU(), nn.Linear(hidden, hidden)
                                                  ) for _ in range(n_blocks))
        self.out = nn.Linear(hidden, 4)

    def forward(self, pts, dirs, feat):
        h = F.relu(self.inp(torch.cat([positional_encoding(pts, self.n_freqs), dirs], -1)))
        for proj, block in zip(self.feat_proj, self.blocks):
            h = h + proj(feat)
            h = h + block(h)
        h = self.out(F.relu(h))
        return torch.sigmoid(h[..., :3]), F.relu(h[..., 3])


def get_rays(c2w, K, H, W, device):
    fx, fy, cx, cy = K
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32),
                            indexing="ij")
    dirs = torch.stack([(xs + 0.5 - cx) / fx, (ys + 0.5 - cy) / fy, torch.ones_like(xs)], dim=-1)
    rays_d = dirs @ c2w[:3, :3].t()
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def render_rays(model, feat, src_w2c, src_K, rays_o, rays_d, near, far, n_samples=96):
    device = rays_o.device
    R = rays_o.shape[0]
    t = torch.linspace(near, far, n_samples, device=device).expand(R, n_samples).clone()
    mid = 0.5 * (t[:, 1:] + t[:, :-1])
    lower = torch.cat([t[:, :1], mid], -1)
    upper = torch.cat([mid, t[:, -1:]], -1)
    t = lower + (upper - lower) * torch.rand_like(t)
    pts_w = rays_o[:, None, :] + t[..., None] * rays_d[:, None, :]  # [R, n, 3]

    # Everything is reasoned about in the source camera frame (pixelNeRF Sec. 4.1).
    Rsrc, tsrc = src_w2c[:3, :3], src_w2c[:3, 3]
    pts_c = pts_w @ Rsrc.t() + tsrc
    dirs_c = F.normalize(rays_d @ Rsrc.t(), dim=-1)[:, None, :].expand_as(pts_c)

    fx, fy, cx, cy = src_K
    H, W = feat.shape[-2:]
    z = pts_c[..., 2].clamp(min=1e-4)
    u = fx * pts_c[..., 0] / z + cx
    v = fy * pts_c[..., 1] / z + cy
    # For each sample point, find where it lands in the source image. grid_sample takes the pixel
    # coordinate rescaled to [-1, 1], running from the first pixel centre to the last.
    grid = torch.stack([2 * (u - 0.5) / (W - 1) - 1,
                        2 * (v - 0.5) / (H - 1) - 1], dim=-1)  # [R, n, 2]
    sampled = F.grid_sample(feat, grid[None], align_corners=True, padding_mode="border")
    sampled = sampled[0].permute(1, 2, 0)  # [R, n, C]

    color, sigma = model(pts_c, dirs_c, sampled)
    delta = torch.cat([t[:, 1:] - t[:, :-1], torch.full_like(t[:, :1], 1e10)], -1)
    alpha = 1 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]),
                                     1 - alpha + 1e-10], -1), -1)[:, :-1]
    weights = trans * alpha
    rgb = (weights[..., None] * color).sum(1)
    return rgb + (1 - weights.sum(1, keepdim=True))  # white background


@torch.no_grad()
def render_image(model, feat, src_w2c, src_K, c2w, tgt_K, H, W, near, far, chunk=2048):
    rays_o, rays_d = get_rays(c2w, tgt_K, H, W, feat.device)
    out = [render_rays(model, feat, src_w2c, src_K,
                       rays_o[i:i + chunk], rays_d[i:i + chunk], near, far)
           for i in range(0, rays_o.shape[0], chunk)]
    return torch.cat(out, 0).reshape(H, W, 3)


@torch.no_grad()
def render_novel_view_grid(encoder, model, dataset, H, W, near, far, device,
                           save_path="Imgs/pixelnerf.png", num_scenes=10):
    view_idx = [1, 2, 4, 7, 10, 13, 16, 19, 22, 23]
    fig, axes = plt.subplots(num_scenes, len(view_idx), dpi=300, squeeze=False,
                             figsize=(2.2 * len(view_idx), 2.2 * num_scenes))
    for s in range(num_scenes):
        src = view_idx[s]
        feat = encoder(dataset.gt_pixels[s, src].permute(2, 0, 1)[None].to(device))
        src_w2c = torch.inverse(dataset.c2ws[s, src]).to(device)
        src_K = intrinsics_to_fxfycxcy(dataset.intrinsics[s, src], H, W)
        axes[s, 0].axis("off")
        axes[s, 0].imshow(dataset.gt_pixels[s, src].numpy().clip(0, 1))
        if s == 0:
            axes[s, 0].set_title("Input image", fontsize=25)
        for col, v in enumerate([i for i in view_idx if i != src], start=1):
            tgt_K = intrinsics_to_fxfycxcy(dataset.intrinsics[s, v], H, W)
            img = render_image(model, feat, src_w2c, src_K, dataset.c2ws[s, v].to(device),
                               tgt_K, H, W, near, far)
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
    batch_size, n_rays = 4, 128

    train_set = NMRDataset(data_root, split_json, train=True, H=H, W=W)
    test_set = NMRDataset(data_root, split_json, train=False, H=H, W=W)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=True, pin_memory=torch.cuda.is_available())
    encoder = ImageEncoder().to(device)
    model = PixelNeRF(encoder.feat_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=1e-4)

    train_iter = iter(train_loader)
    for step in tqdm(range(1, 400_001)):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        feats = encoder(batch["src_img"].to(device))  # [B, C, H, W]
        loss = 0.0
        for b in range(batch_size):
            src_w2c = torch.inverse(batch["source_c2w"][b]).to(device)
            src_K = intrinsics_to_fxfycxcy(batch["source_cam"][b], H, W)
            tgt_K = intrinsics_to_fxfycxcy(batch["target_cam"][b], H, W)
            rays_o, rays_d = get_rays(batch["target_c2w"][b].to(device), tgt_K, H, W, device)
            idx = torch.randint(0, H * W, (n_rays,), device=device)
            rgb = render_rays(model, feats[b:b + 1], src_w2c, src_K,
                              rays_o[idx], rays_d[idx], near, far)
            tgt = batch["tgt_img"][b].permute(1, 2, 0).reshape(-1, 3).to(device)[idx]
            loss = loss + F.mse_loss(rgb, tgt)
        loss = loss / batch_size

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    encoder.eval()
    model.eval()
    render_novel_view_grid(encoder, model, test_set, H, W, near, far, device,
                           save_path="Imgs/pixelnerf.png")
