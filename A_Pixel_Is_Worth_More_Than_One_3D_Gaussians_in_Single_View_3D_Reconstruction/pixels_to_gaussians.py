import os
import json
import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from unet import SplatterImageUNet
from gaussian_splatting import render, quat_xyzw_to_rotmat


class NMRDataset(Dataset):
    def __init__(self, data_path, json_path, train=True, n_views=24, H=64, W=64):
        self.data_path = data_path
        self.n_views = n_views
        self.H = H
        self.W = W

        with open(json_path, "r") as f:
            split = json.load(f)

        self.scenes = [
            os.path.join(data_path, f) for f in sorted(split["train" if train else "test"])]

        gt_pixels = []
        c2ws = []
        intrinsics = []
        for scene_path in tqdm(self.scenes, desc="loading scenes"):
            cam_data = np.load(os.path.join(scene_path, "cameras.npz"))

            scene_gt_pixels = torch.zeros((n_views, H, W, 3), dtype=torch.float32)
            scene_c2ws = torch.zeros((n_views, 4, 4), dtype=torch.float32)
            scene_intrinsics = torch.zeros((n_views, 4, 4), dtype=torch.float32)

            for view_idx in range(n_views):
                img = np.array(Image.open(
                    os.path.join(scene_path, "image", f"{view_idx:04d}.png")).convert("RGB"))
                c2w = cam_data[f"world_mat_inv_{view_idx}"]
                K = cam_data[f"camera_mat_{view_idx}"]

                scene_gt_pixels[view_idx] = torch.from_numpy(img).float() / 255.0
                scene_c2ws[view_idx] = torch.from_numpy(c2w).float()
                scene_intrinsics[view_idx] = torch.from_numpy(K).float()

            gt_pixels.append(scene_gt_pixels)
            c2ws.append(scene_c2ws)
            intrinsics.append(scene_intrinsics)

        self.gt_pixels = torch.stack(gt_pixels, dim=0)  # [B, N, H, W, 3]
        self.c2ws = torch.stack(c2ws, dim=0)  # [B, N, 4, 4]
        self.intrinsics = torch.stack(intrinsics, dim=0)  # [B, N, 4, 4]

    def __len__(self):
        return self.gt_pixels.shape[0]

    def __getitem__(self, scene_idx):
        scene_imgs = self.gt_pixels[scene_idx]
        scene_c2ws = self.c2ws[scene_idx]
        scene_intr = self.intrinsics[scene_idx]

        src_idx, tgt_idx = sample_source_target_indices(self.n_views)
        return {"src_img": scene_imgs[src_idx].permute(2, 0, 1),
                "tgt_img": scene_imgs[tgt_idx].permute(2, 0, 1),
                "source_c2w": scene_c2ws[src_idx],
                "target_c2w": scene_c2ws[tgt_idx],
                "source_cam": scene_intr[src_idx],
                "target_cam": scene_intr[tgt_idx],
                "meta": torch.tensor([scene_idx, src_idx, tgt_idx], dtype=torch.long)}


def inv_sigmoid_scalar(y, eps=1e-6):
    y = max(eps, min(1.0 - eps, float(y)))
    return math.log(y / (1.0 - y))


def intrinsics_to_fxfycxcy(camera_mat, H, W):
    s = float(camera_mat[0, 0])
    fx = s * W / 2.0
    fy = s * H / 2.0
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy


def sample_source_target_indices(n_views):
    src_idx = random.randrange(n_views)
    tgt_idx = random.randrange(n_views - 1)
    if tgt_idx >= src_idx:
        tgt_idx += 1
    return src_idx, tgt_idx


def build_pixel_grid_from_intrinsics(H, W, fx, fy, cx, cy, device, dtype):
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                            torch.arange(W, device=device, dtype=dtype),
                            indexing="ij")
    u = (xs + 0.5 - cx) / fx
    v = (ys + 0.5 - cy) / fy
    return torch.stack([u, v, torch.ones_like(u)], dim=0)


class ChildGaussianMLPs(nn.Module):
    def __init__(self, n_children=3, parent_dim=15, hidden_dim=24):
        super().__init__()
        self.n_children = n_children
        in_dim = parent_dim + 1 + 3  # parent Gaussian params + distance (1) + direction (3)

        self.mlp_offset = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, 3 * n_children))
        self.mlp_scale = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, 3 * n_children))
        self.mlp_quat = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, 4 * n_children))
        self.mlp_color = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, 3 * n_children))
        self.mlp_opacity = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, 1 * n_children))

    def forward(self, x):
        k = self.n_children
        offset = self.mlp_offset(x).reshape(-1, k, 3)
        scale = self.mlp_scale(x).reshape(-1, k, 3)
        quat = self.mlp_quat(x).reshape(-1, k, 4)
        color = self.mlp_color(x).reshape(-1, k, 3)
        opacity = self.mlp_opacity(x).reshape(-1, k, 1)
        return offset, scale, quat, color, opacity


def decode_parent_gaussians(raw, source_c2w, fx, fy, cx, cy, znear, zfar):
    device = raw.device
    dtype = raw.dtype
    _, _, H, W = raw.shape

    opacity_raw = raw[:, 0:1]
    delta = raw[:, 1:4]
    depth_raw = raw[:, 4:5]
    scale_raw = raw[:, 5:8]
    quat_raw = raw[:, 8:12]
    color_raw = raw[:, 12:15]

    depth = (zfar - znear) * torch.sigmoid(depth_raw.clamp(-10, 10)) + znear
    u = build_pixel_grid_from_intrinsics(H, W, fx, fy, cx, cy, device, dtype).unsqueeze(0)

    Rcw = source_c2w[:3, :3]
    tcw = source_c2w[:3, 3]

    mean_cam = torch.empty((1, 3, H, W), device=device, dtype=dtype)
    mean_cam[:, 0:1] = u[:, 0:1] * depth + delta[:, 0:1]
    mean_cam[:, 1:2] = u[:, 1:2] * depth + delta[:, 1:2]
    mean_cam[:, 2:3] = depth + delta[:, 2:3]

    color = torch.sigmoid(color_raw)
    quat_xyzw = torch.nn.functional.normalize(quat_raw, dim=1, eps=1e-6)

    pos_cam = mean_cam[0].permute(1, 2, 0).reshape(-1, 3)
    opacity_raw_flat = opacity_raw[0].reshape(-1)
    scale_raw_flat = scale_raw[0].permute(1, 2, 0).reshape(-1, 3)
    quat_flat = quat_xyzw[0].permute(1, 2, 0).reshape(-1, 4)
    color_flat = color[0].permute(1, 2, 0).reshape(-1, 3)

    # To world space
    pos_world = (Rcw @ pos_cam.t()).t() + tcw.unsqueeze(0)
    # Covariance in world space
    scales = torch.exp(scale_raw_flat.clamp(-10.0, 3.0)).clamp_min(1e-6)
    R_local = quat_xyzw_to_rotmat(quat_flat)
    S = torch.diag_embed(scales)
    sigma_cam = R_local @ S @ S @ R_local.transpose(1, 2)
    sigma_world = Rcw.unsqueeze(0) @ sigma_cam @ Rcw.t().unsqueeze(0)

    parent_raw_flat = raw[0].permute(1, 2, 0).reshape(-1, 15)
    return pos_world, color_flat, opacity_raw_flat, sigma_world, parent_raw_flat


def decode_child_gaussians(child_mlps, parent_raw_flat, parent_pos_world,
                           target_cam_center, source_c2w):
    Rcw = source_c2w[:3, :3]

    cam_to_parent = parent_pos_world - target_cam_center.unsqueeze(0)  # [N, 3]
    dist_to_cam = torch.sqrt((cam_to_parent ** 2).sum(dim=-1, keepdim=True) + 1e-8)
    dir_from_cam = cam_to_parent / dist_to_cam  # [N, 3]

    mlp_input = torch.cat([parent_raw_flat, dist_to_cam, dir_from_cam], dim=-1)  # [N, 19]
    offset, scale_raw, quat_raw, color_raw, opacity_raw = child_mlps(mlp_input)  # [N, k, dim]
    child_pos = parent_pos_world.unsqueeze(1) + offset  # [N, k, 3]

    child_color = torch.sigmoid(color_raw)  # [N, k, 3]
    child_quat = torch.nn.functional.normalize(quat_raw, dim=-1, eps=1e-6)  # [N, k, 4]
    child_scales = torch.exp(scale_raw.clamp(-10.0, 3.0)).clamp_min(1e-6)   # [N, k, 3]

    # Covariance  Σ = R S² Rᵀ  then rotate to world
    R_local = quat_xyzw_to_rotmat(child_quat.reshape(-1, 4))  # [N*k, 3, 3]
    S = torch.diag_embed(child_scales.reshape(-1, 3))  # [N*k, 3, 3]
    sigma_cam = R_local @ S @ S @ R_local.transpose(1, 2)
    sigma_world = Rcw.unsqueeze(0) @ sigma_cam @ Rcw.t().unsqueeze(0)

    child_pos_flat = child_pos.reshape(-1, 3)
    child_color_flat = child_color.reshape(-1, 3)
    child_opacity_flat = opacity_raw.reshape(-1)
    return child_pos_flat, child_color_flat, child_opacity_flat, sigma_world


@torch.no_grad()
def init_output_head(model, znear=0.8, zfar=1.8, *, opacity_bias=-3.5, depth0=1.2,
                     scale_bias=0.02, opacity_gain=1.0, xyz_gain=0.1, depth_gain=1.0,
                     scale_gain=0.1, quat_gain=1.0, rgb_gain=5.0):
    weight = model.head.weight
    bias = model.head.bias
    weight.zero_()
    bias.zero_()

    depth01 = (depth0 - znear) / (zfar - znear)
    depth_bias = inv_sigmoid_scalar(depth01)

    torch.nn.init.xavier_uniform_(weight[0], gain=opacity_gain)  # Opacity
    torch.nn.init.constant_(bias[0], opacity_bias)
    torch.nn.init.xavier_uniform_(weight[1:4], gain=xyz_gain)  # xyz
    torch.nn.init.xavier_uniform_(weight[4], gain=depth_gain)  # Depth
    torch.nn.init.constant_(bias[4], depth_bias)
    torch.nn.init.xavier_uniform_(weight[5:8], gain=scale_gain)  # Scale
    torch.nn.init.constant_(bias[5:8], math.log(scale_bias))
    torch.nn.init.xavier_uniform_(weight[8:12], gain=quat_gain)  # Quat
    torch.nn.init.constant_(bias[11], 1.0)
    torch.nn.init.xavier_uniform_(weight[12:15], gain=rgb_gain)  # RGB


@torch.no_grad()
def init_child_mlps(child_mlps, opacity_bias=-3.5, scale_bias=0.02, xyz_gain=0.1,
                    scale_gain=0.1, quat_gain=1.0, rgb_gain=5.0, opacity_gain=1.0):
    k = child_mlps.n_children

    weight = child_mlps.mlp_offset[-1].weight  # Offset
    bias = child_mlps.mlp_offset[-1].bias
    torch.nn.init.xavier_uniform_(weight, gain=xyz_gain)
    torch.nn.init.constant_(bias, 0.0)
    weight = child_mlps.mlp_scale[-1].weight  # Scale
    bias = child_mlps.mlp_scale[-1].bias
    torch.nn.init.xavier_uniform_(weight, gain=scale_gain)
    torch.nn.init.constant_(bias, math.log(scale_bias))
    weight = child_mlps.mlp_quat[-1].weight  # Quat (xyzw)
    bias = child_mlps.mlp_quat[-1].bias
    torch.nn.init.xavier_uniform_(weight, gain=quat_gain)
    torch.nn.init.constant_(bias, 0.0)
    for i in range(k):
        bias[i * 4 + 3] = 1.0
    weight = child_mlps.mlp_color[-1].weight  # Color
    bias = child_mlps.mlp_color[-1].bias
    torch.nn.init.xavier_uniform_(weight, gain=rgb_gain)
    torch.nn.init.constant_(bias, 0.0)
    weight = child_mlps.mlp_opacity[-1].weight  # Opacity
    bias = child_mlps.mlp_opacity[-1].bias
    torch.nn.init.xavier_uniform_(weight, gain=opacity_gain)
    torch.nn.init.constant_(bias, opacity_bias)


@torch.no_grad()
def render_novel_view_grid(model, child_mlps, gt_pixels, c2ws, intrinsics, H, W, znear, zfar,
                           device, save_path="pixels_to_gaussians_v2.png", num_test_scenes=10,
                           title_fontsize=25, use_children=True):

    novel_view_indices = [1, 2, 4, 7, 10, 13, 16, 19, 22, 23]
    ncols = len(novel_view_indices)
    fig, axes = plt.subplots(num_test_scenes, ncols, figsize=(2.2 * ncols, 2.2 * num_test_scenes),
                             dpi=300, squeeze=False)

    for scene_idx in range(num_test_scenes):
        scene_imgs = gt_pixels[scene_idx]
        scene_c2ws = c2ws[scene_idx]
        scene_intr = intrinsics[scene_idx]

        src_view_idx = novel_view_indices[scene_idx]
        tgt_view_indices = [v for v in novel_view_indices if v != src_view_idx]
        src_img = scene_imgs[src_view_idx].permute(2, 0, 1).unsqueeze(0).to(device)
        source_c2w = scene_c2ws[src_view_idx].to(device)
        source_cam = scene_intr[src_view_idx]

        fx_src, fy_src, cx_src, cy_src = intrinsics_to_fxfycxcy(source_cam, H, W)
        pos, color, opacity_raw, sigma, parent_raw_flat = decode_parent_gaussians(
            raw=model(src_img), source_c2w=source_c2w, fx=fx_src, fy=fy_src,
            cx=cx_src, cy=cy_src, znear=znear, zfar=zfar)

        ax = axes[scene_idx, 0]
        ax.axis("off")
        ax.imshow(scene_imgs[src_view_idx].cpu().numpy().clip(0, 1))
        if scene_idx == 0:
            ax.set_title("Input image", fontsize=title_fontsize)
        for col_idx, view_idx in enumerate(tgt_view_indices, start=1):
            target_c2w = scene_c2ws[view_idx].to(device)
            target_cam = scene_intr[view_idx]
            fx_t, fy_t, cx_t, cy_t = intrinsics_to_fxfycxcy(target_cam, H, W)

            if use_children and child_mlps is not None:  # Combine parent + child Gaussians
                target_cam_center = target_c2w[:3, 3]
                c_pos, c_color, c_opa, c_sigma = decode_child_gaussians(child_mlps, parent_raw_flat,
                                                                        pos, target_cam_center,
                                                                        source_c2w)
                all_pos = torch.cat([pos, c_pos], dim=0)
                all_color = torch.cat([color, c_color], dim=0)
                all_opacity = torch.cat([opacity_raw, c_opa], dim=0)
                all_sigma = torch.cat([sigma, c_sigma], dim=0)
            else:
                all_pos, all_color, all_opacity, all_sigma = pos, color, opacity_raw, sigma

            pred = render(pos=all_pos, color=all_color, opacity_raw=all_opacity, sigma=all_sigma,
                          c2w=target_c2w, H=H, W=W, fx=fx_t, fy=fy_t, cx=cx_t, cy=cy_t)

            ax = axes[scene_idx, col_idx]
            ax.axis("off")
            ax.imshow(pred.cpu().numpy().clip(0, 1))
            if scene_idx == 0 and col_idx == (ncols // 2):
                ax.set_title("Novel views", fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data_root = "NMR_Dataset/02958343/"
    split_json = "car_splits.json"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    H = W = 64

    znear = 2.0
    zfar = 3.5
    depth0 = 2.7
    batch_size = 8
    train_steps = 800_001
    stage1_steps = 10_000
    n_children = 3
    lr_main = 1e-4

    train_dataset = NMRDataset(data_path=data_root, json_path=split_json, train=True, H=H, W=W)
    test_dataset = NMRDataset(data_path=data_root, json_path=split_json, train=False, H=H, W=W)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available(), drop_last=True)
    model = SplatterImageUNet().to(device)
    init_output_head(model, znear=znear, zfar=zfar, depth0=depth0)
    child_mlps = ChildGaussianMLPs(n_children=n_children).to(device)
    init_child_mlps(child_mlps)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(child_mlps.parameters()),
                                 lr=lr_main)

    train_iter = iter(train_loader)
    for step in tqdm(range(1, train_steps + 1)):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        use_children = step > stage1_steps
        batch_src_imgs = batch["src_img"].to(device, non_blocking=True)
        batch_tgt_imgs = batch["tgt_img"].to(device, non_blocking=True)
        batch_source_c2w = batch["source_c2w"].to(device, non_blocking=True)
        batch_target_c2w = batch["target_c2w"].to(device, non_blocking=True)
        batch_source_cam = batch["source_cam"]
        batch_target_cam = batch["target_cam"]

        optimizer.zero_grad(set_to_none=True)
        raw_batch = model(batch_src_imgs)  # [B, 15, H, W]

        n_scenes = raw_batch.shape[0]
        batch_loss = 0.0
        for b in range(n_scenes):
            fx_src, fy_src, cx_src, cy_src = intrinsics_to_fxfycxcy(batch_source_cam[b], H, W)
            fx_t, fy_t, cx_t, cy_t = intrinsics_to_fxfycxcy(batch_target_cam[b], H, W)

            raw_b = raw_batch[b:b + 1]
            pos, color, opacity_raw, sigma, parent_raw_flat = decode_parent_gaussians(
                raw=raw_b, source_c2w=batch_source_c2w[b], fx=fx_src, fy=fy_src, cx=cx_src,
                cy=cy_src, znear=znear, zfar=zfar)

            if use_children:
                target_cam_center = batch_target_c2w[b, :3, 3]
                c_pos, c_color, c_opa, c_sigma = decode_child_gaussians(
                    child_mlps, parent_raw_flat, pos, target_cam_center, batch_source_c2w[b])

                all_pos = torch.cat([pos, c_pos], dim=0)
                all_color = torch.cat([color, c_color], dim=0)
                all_opacity = torch.cat([opacity_raw, c_opa], dim=0)
                all_sigma = torch.cat([sigma, c_sigma], dim=0)
            else:
                all_pos, all_color, all_opacity, all_sigma = pos, color, opacity_raw, sigma

            pred = render(pos=all_pos, color=all_color, opacity_raw=all_opacity, sigma=all_sigma,
                          c2w=batch_target_c2w[b], H=H, W=W, fx=fx_t, fy=fy_t, cx=cx_t, cy=cy_t
                          ).permute(2, 0, 1).unsqueeze(0)
            tgt = batch_tgt_imgs[b:b + 1]
            loss_b = torch.nn.functional.mse_loss(pred, tgt)
            batch_loss = batch_loss + loss_b
        batch_loss = batch_loss / n_scenes
        batch_loss.backward()
        optimizer.step()

    render_novel_view_grid(model=model, child_mlps=child_mlps, gt_pixels=test_dataset.gt_pixels,
                           c2ws=test_dataset.c2ws, intrinsics=test_dataset.intrinsics, H=H, W=W,
                           znear=znear, zfar=zfar, device=device, num_test_scenes=10,
                           save_path="Imgs/pixels_to_gaussians.png", use_children=True)
