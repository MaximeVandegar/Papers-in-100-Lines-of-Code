import torch
from tqdm import tqdm
import numpy as np
from PIL import Image

SH_C0 = 0.28209479177387814
SH_C1_x = 0.4886025119029199
SH_C1_y = 0.4886025119029199
SH_C1_z = 0.4886025119029199
SH_C2_xy = 1.0925484305920792
SH_C2_xz = 1.0925484305920792
SH_C2_yz = 1.0925484305920792
SH_C2_zz = 0.31539156525252005
SH_C2_xx_yy = 0.5462742152960396
SH_C3_yxx_yyy = 0.5900435899266435
SH_C3_xyz = 2.890611442640554
SH_C3_yzz_yxx_yyy = 0.4570457994644658
SH_C3_zzz_zxx_zyy = 0.3731763325901154
SH_C3_xzz_xxx_xyy = 0.4570457994644658
SH_C3_zxx_zyy = 1.445305721320277
SH_C3_xxx_xyy = 0.5900435899266435

def evaluate_sh(f_dc, f_rest, points, c2w):

    sh = torch.empty((points.shape[0], 16, 3),
                     device=points.device, dtype=points.dtype)
    sh[:, 0] = f_dc
    sh[:, 1:, 0] = f_rest[:, :15]  # R
    sh[:, 1:, 1] = f_rest[:, 15:30]  # G
    sh[:, 1:, 2] = f_rest[:, 30:45]  # B

    view_dir = points - c2w[:3, 3].unsqueeze(0)  # [N, 3]
    view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)
    x, y, z = view_dir[:, 0], view_dir[:, 1], view_dir[:, 2]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z

    Y0 = torch.full_like(x, SH_C0)  # [N]
    Y1 = - SH_C1_y * y
    Y2 = SH_C1_z * z
    Y3 = - SH_C1_x * x
    Y4 = SH_C2_xy * xy
    Y5 = SH_C2_yz * yz
    Y6 = SH_C2_zz * (3 * zz - 1)
    Y7 = SH_C2_xz * xz
    Y8 = SH_C2_xx_yy * (xx - yy)
    Y9 = SH_C3_yxx_yyy * y * (3 * xx - yy)
    Y10 = SH_C3_xyz * x * y * z
    Y11 = SH_C3_yzz_yxx_yyy * y * (4 * zz - xx - yy)
    Y12 = SH_C3_zzz_zxx_zyy * z * (2 * zz - 3 * xx - 3 * yy)
    Y13 = SH_C3_xzz_xxx_xyy * x * (4 * zz - xx - yy)
    Y14 = SH_C3_zxx_zyy * z * (xx - yy)
    Y15 = SH_C3_xxx_xyy * x * (xx - 3 * yy)
    Y = torch.stack([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12, Y13, Y14, Y15],
                    dim=1)  # [N, 16]
    return torch.sigmoid((sh * Y.unsqueeze(2)).sum(dim=1))

def project_points(pc, c2w, fx, fy, cx, cy):
    w2c = torch.eye(4, device=pc.device)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c[:3, :3] = R.t()
    w2c[:3, 3] = -R.t() @ t

    PC = ((w2c @ torch.concatenate(
        [pc, torch.ones_like(pc[:, :1])], dim=1).t()).t())[:, :3]
    x, y, z = PC[:, 0], PC[:, 1], PC[:, 2]  # Camera space

    uv = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)
    return uv, x, y, z

def inv2x2(M, eps=1e-12):
    a = M[:, 0, 0]
    b = M[:, 0, 1]
    c = M[:, 1, 0]
    d = M[:, 1, 1]
    det = a * d - b * c
    safe_det = torch.clamp(det, min=eps)
    inv = torch.empty_like(M)
    inv[:, 0, 0] = d / safe_det
    inv[:, 0, 1] = -b / safe_det
    inv[:, 1, 0] = -c / safe_det
    inv[:, 1, 1] = a / safe_det
    return inv

def build_sigma_from_params(scale_raw, q_raw):
    scale = torch.exp(scale_raw).clamp_min(1e-6)
    q = q_raw / (q_raw.norm(dim=-1, keepdim=True) + 1e-9)
    R = quat_to_rotmat(q)
    S = torch.diag_embed(scale)
    return R @ S @ S @ R.transpose(1, 2)

def quat_to_rotmat(quat):
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)
    ], dim=-1).reshape(quat.shape[:-1] + (3, 3))
    return R

def scale_intrinsics(H, W, H_src, W_src, fx, fy, cx, cy):
    scale_x = W / W_src
    scale_y = H / H_src
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y
    return fx_scaled, fy_scaled, cx_scaled, cy_scaled

@torch.no_grad()
def render(pos, color, opacity_raw, sigma, c2w, H, W, fx, fy, cx, cy,
           near=2e-3, far=100, pix_guard=64, T=16, min_conis=1e-6,
           chi_square_clip=9.21, alpha_max=0.99, alpha_cutoff=1/255.):

    uv, x, y, z = project_points(pos, c2w, fx, fy, cx, cy)
    in_guard = (uv[:, 0] > -pix_guard) & (uv[:, 0] < W + pix_guard) & (
        uv[:, 1] > -pix_guard) & (uv[:, 1] < H + pix_guard) & (z > near) & (z < far)

    uv = uv[in_guard]
    pos = pos[in_guard]
    color = color[in_guard]
    opacity = torch.sigmoid(opacity_raw[in_guard]).clamp(0, 0.999)
    z = z[in_guard]
    x = x[in_guard]
    y = y[in_guard]
    sigma = sigma[in_guard]
    idx = torch.nonzero(in_guard, as_tuple=False).squeeze(1)

    # Project the covariance
    Rcw = c2w[:3, :3]
    Rwc = Rcw.t()
    invz = 1 / z.clamp_min(1e-6)
    invz2 = invz * invz
    J = torch.zeros((pos.shape[0], 2, 3), device=pos.device, dtype=pos.dtype)
    J[:, 0, 0] = fx * invz
    J[:, 1, 1] = fy * invz
    J[:, 0, 2] = -fx * x * invz2
    J[:, 1, 2] = -fy * y * invz2
    tmp = Rwc.unsqueeze(0) @ sigma @ Rwc.t().unsqueeze(0)  # Eq. 5
    sigma_camera = J @ tmp @ J.transpose(1, 2)
    sigma_camera = 0.5 * (sigma_camera + sigma_camera.transpose(1, 2))  # Enforce symmetry
    # Ensure positive definiteness
    evals, evecs = torch.linalg.eigh(sigma_camera)
    evals = torch.clamp(evals, min=1e-6, max=1e4)
    sigma_camera = evecs @ torch.diag_embed(evals) @ evecs.transpose(1, 2)

    keep = torch.isfinite(
        sigma_camera.reshape(sigma.shape[0], -1)).all(dim=-1)
    uv = uv[keep]
    color = color[keep]
    opacity = opacity[keep]
    z = z[keep]
    sigma_camera = sigma_camera[keep]
    idx = idx[keep]

    # Global depth sorting
    order = torch.argsort(z, descending=False)
    uv = uv[order]
    u = uv[:, 0]
    v = uv[:, 1]
    color = color[order]
    opacity = opacity[order]
    sigma_camera = sigma_camera[order]
    evals = evals[order]
    idx = idx[order]

    inverse_covariance = inv2x2(sigma_camera)
    inverse_covariance[:, 0, 0] = torch.clamp(inverse_covariance[:, 0, 0], min=min_conis)
    inverse_covariance[:, 1, 1] = torch.clamp(inverse_covariance[:, 1, 1], min=min_conis)

    # Tiling
    A = inverse_covariance[:, 0, 0]
    B = inverse_covariance[:, 0, 1]
    C = inverse_covariance[:, 1, 1]
    t = 2.0 * torch.log(255.0 * opacity)  # Eq. 11
    B2_minus_AC = (B ** 2 - A * C)  # Eq. 16
    xd_arg = torch.sqrt(- B * B * t / (B2_minus_AC * A))
    xd_arg[B < 0] = - xd_arg[B < 0]  # For Eq. 15 to be == 0, xd should be <0 when B<0
    yd_arg = torch.sqrt(- B * B * t / (B2_minus_AC * C))   # Symmetry of Eq. 16
    yd_arg[B < 0] = - yd_arg[B < 0]

    # Substituting xd_args into Equation 15 and adding µ
    vmax = v + (B * xd_arg + torch.sqrt((B ** 2 - A * C) * xd_arg ** 2 + t * C)) / C
    vmin = v + (-B * xd_arg - torch.sqrt((B ** 2 - A * C) * xd_arg ** 2 + t * C)) / C
    umax = u + (B * yd_arg + torch.sqrt((B ** 2 - A * C) * yd_arg ** 2 + t * A)) / A
    umin = u + (-B * yd_arg - torch.sqrt((B ** 2 - A * C) * yd_arg ** 2 + t * A)) / A
    umin = torch.floor(umin).to(torch.int64)
    umax = torch.floor(umax).to(torch.int64)
    vmin = torch.floor(vmin).to(torch.int64)
    vmax = torch.floor(vmax).to(torch.int64)

    on_screen = (umax >= 0) & (umin < W) & (vmax >= 0) & (vmin < H)
    if not on_screen.any():
        raise Exception("All projected points are off-screen")
    u, v = u[on_screen], v[on_screen]
    color = color[on_screen]
    opacity = opacity[on_screen]
    sigma_camera = sigma_camera[on_screen]
    inverse_covariance = inverse_covariance[on_screen]
    umin, umax = umin[on_screen], umax[on_screen]
    vmin, vmax = vmin[on_screen], vmax[on_screen]
    idx = idx[on_screen]
    umin = umin.clamp(0, W - 1)
    umax = umax.clamp(0, W - 1)
    vmin = vmin.clamp(0, H - 1)
    vmax = vmax.clamp(0, H - 1)

    # Tile index for each AABB
    umin_tile = (umin // T).to(torch.int64)  # [N]
    umax_tile = (umax // T).to(torch.int64)  # [N]
    vmin_tile = (vmin // T).to(torch.int64)  # [N]
    vmax_tile = (vmax // T).to(torch.int64)  # [N]

    # Number of tiles each gaussian intersects
    n_u = umax_tile - umin_tile + 1  # [N]
    n_v = vmax_tile - vmin_tile + 1  # [N]

    # Max number of tiles
    max_u = int(n_u.max().item())
    max_v = int(n_v.max().item())

    nb_gaussians = umin_tile.shape[0]
    span_indices_u = torch.arange(max_u, device=pos.device, dtype=torch.int64)  # [max_u]
    span_indices_v = torch.arange(max_v, device=pos.device, dtype=torch.int64)  # [max_v]
    tile_u = (umin_tile[:, None, None] + span_indices_u[None, :, None]
              ).expand(nb_gaussians, max_u, max_v)  # [N, max_u, max_v]
    tile_v = (vmin_tile[:, None, None] + span_indices_v[None, None, :]
              ).expand(nb_gaussians, max_u, max_v)  # [N, max_u, max_v]
    mask = (span_indices_u[None, :, None] < n_u[:, None, None]
            ) & (span_indices_v[None, None, :] < n_v[:, None, None])  # [N, max_u, max_v]
    flat_tile_u = tile_u[mask]  # [0, 0, 1, 1, 2, ...]
    flat_tile_v = tile_v[mask]  # [0, 1, 0, 1, 2]

    nb_tiles_per_gaussian = n_u * n_v  # [N]
    gaussian_ids = torch.repeat_interleave(
        torch.arange(nb_gaussians, device=pos.device, dtype=torch.int64),
        nb_tiles_per_gaussian)  # [0, 0, 0, 0, 1 ...]
    nb_tiles_u = (W + T - 1) // T
    flat_tile_id = flat_tile_v * nb_tiles_u + flat_tile_u  # [0, 0, 0, 0, 1 ...]

    idx_z_order = torch.arange(nb_gaussians, device=pos.device, dtype=torch.int64)
    M = nb_gaussians + 1
    comp = flat_tile_id * M + idx_z_order[gaussian_ids]
    comp_sorted, perm = torch.sort(comp)
    gaussian_ids = gaussian_ids[perm]
    tile_ids_1d = torch.div(comp_sorted, M, rounding_mode='floor')

    # tile_ids_1d [0, 0, 0, 1, 1, 2, 2, 2, 2]
    # nb_gaussian_per_tile [3, 2, 4]
    # start [0, 3, 5]
    # end [3, 5, 9]
    unique_tile_ids, nb_gaussian_per_tile = torch.unique_consecutive(tile_ids_1d, return_counts=True)
    start = torch.zeros_like(unique_tile_ids)
    start[1:] = torch.cumsum(nb_gaussian_per_tile[:-1], dim=0)
    end = start + nb_gaussian_per_tile

    final_image = torch.zeros((H * W, 3), device=pos.device, dtype=pos.dtype)
    # Iterate over tiles
    for tile_id, s0, s1 in zip(unique_tile_ids.tolist(), start.tolist(), end.tolist()):

        current_gaussian_ids = gaussian_ids[s0:s1]

        txi = tile_id % nb_tiles_u
        tyi = tile_id // nb_tiles_u
        x0, y0 = txi * T, tyi * T
        x1, y1 = min((txi + 1) * T, W), min((tyi + 1) * T, H)
        if x0 >= x1 or y0 >= y1:
            continue

        xs = torch.arange(x0, x1, device=pos.device, dtype=pos.dtype)
        ys = torch.arange(y0, y1, device=pos.device, dtype=pos.dtype)
        pu, pv = torch.meshgrid(xs, ys, indexing='xy')
        px_u = pu.reshape(-1)  # [T * T]
        px_v = pv.reshape(-1)
        pixel_idx_1d = (px_v * W + px_u).to(torch.int64)

        gaussian_i_u = u[current_gaussian_ids]  # [N]
        gaussian_i_v = v[current_gaussian_ids]  # [N]
        gaussian_i_color = color[current_gaussian_ids]  # [N, 3]
        gaussian_i_opacity = opacity[current_gaussian_ids]  # [N]
        gaussian_i_inverse_covariance = inverse_covariance[current_gaussian_ids]  # [N, 2, 2]

        du = px_u.unsqueeze(0) - gaussian_i_u.unsqueeze(-1)  # [N, T * T]
        dv = px_v.unsqueeze(0) - gaussian_i_v.unsqueeze(-1)  # [N, T * T]
        A11 = gaussian_i_inverse_covariance[:, 0, 0].unsqueeze(-1)  # [N, 1]
        A12 = gaussian_i_inverse_covariance[:, 0, 1].unsqueeze(-1)
        A22 = gaussian_i_inverse_covariance[:, 1, 1].unsqueeze(-1)
        q = A11 * du * du + 2 * A12 * du * dv + A22 * dv * dv   # [N, T * T]

        inside = q <= chi_square_clip
        g = torch.exp(-0.5 * torch.clamp(q, max=chi_square_clip))  # [N, T * T]
        g = torch.where(inside, g, torch.zeros_like(g))
        alpha_i = (gaussian_i_opacity.unsqueeze(-1) * g).clamp_max(alpha_max)  # [N, T * T]
        alpha_i = torch.where(alpha_i >= alpha_cutoff, alpha_i, torch.zeros_like(alpha_i))
        one_minus_alpha_i = 1 - alpha_i  # [N, T * T]

        T_i = torch.cumprod(one_minus_alpha_i, dim=0)
        T_i = torch.concatenate([
            torch.ones((1, alpha_i.shape[-1]), device=pos.device, dtype=pos.dtype),
            T_i[:-1]], dim=0)
        alive = (T_i > 1e-4).float()
        w = alpha_i * T_i * alive  # [N, T * T]

        final_image[pixel_idx_1d] = (w.unsqueeze(-1) * gaussian_i_color.unsqueeze(1)).sum(dim=0)
    return final_image.reshape((H, W, 3)).clamp(0, 1)

if __name__ == "__main__":

    pos = torch.load('trained_gaussians/kitchen/pos_7000.pt').cuda()
    opacity_raw = torch.load('trained_gaussians/kitchen/opacity_raw_7000.pt').cuda()
    f_dc = torch.load('trained_gaussians/kitchen/f_dc_7000.pt').cuda()
    f_rest = torch.load('trained_gaussians/kitchen/f_rest_7000.pt').cuda()
    scale_raw = torch.load('trained_gaussians/kitchen/scale_raw_7000.pt').cuda()
    q_raw = torch.load('trained_gaussians/kitchen/q_rot_7000.pt').cuda()

    cam_parameters = np.load('out_colmap/kitchen/cam_meta.npy', allow_pickle=True).item()
    orbit_c2ws = torch.load('camera_trajectories/kitchen_orbit.pt').cuda()

    sigma = build_sigma_from_params(scale_raw, q_raw)

    with torch.no_grad():
        for i, c2w_i in tqdm(enumerate(orbit_c2ws)):

            c2w = c2w_i
            H = cam_parameters['height'] // 2
            W = cam_parameters['width'] // 2
            H_src = cam_parameters['height']
            W_src = cam_parameters['width']
            fx, fy = cam_parameters['fx'], cam_parameters['fy']
            cx, cy = W_src / 2, H_src / 2
            fx, fy, cx, cy = scale_intrinsics(H, W, H_src, W_src, fx, fy, cx, cy)

            color = evaluate_sh(f_dc, f_rest, pos, c2w)
            img = render(pos, color, opacity_raw, sigma, c2w, H, W, fx, fy, cx, cy)

            Image.fromarray((img.cpu().detach().numpy() * 255).astype(np.uint8)).save(f'novel_views/frame_{i:04d}.png')
