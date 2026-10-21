import os
import glob
import math
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def parse_backbone(path):
    """Read the N, CA and C atoms of each residue out of a PDB file."""
    residues = {}
    with open(path, "r") as pdb_file:
        for line in pdb_file:
            # ATOM lines hold coordinates; an atom in two places is written twice, A and B; keep A
            if line[:6] != "ATOM  " or line[16] not in " A":
                continue
            atom_name = line[12:16].strip()
            if atom_name in ("N", "CA", "C"):  # the backbone, ignoring side chains
                residue_key = line[21:27]
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                residues.setdefault(residue_key, {})[atom_name] = [x, y, z]
    # Unresolved atoms are missing from the file, so keep only complete residues
    backbone = [[r["N"], r["CA"], r["C"]] for r in residues.values() if len(r) == 3]
    return np.array(backbone).reshape(-1, 3)


def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def longest_bonded_segment(atoms, expected_c_to_n, tolerance=0.1):
    """Residues next to each other in the file are not always bonded, usually because some
    are missing, so keep the longest segment that the C to N bond still links together."""
    residues = atoms.reshape(-1, 3, 3)
    c_to_n_distance = np.linalg.norm(residues[1:, 0] - residues[:-1, 2], axis=-1)
    idx_after_gap = np.flatnonzero(np.abs(c_to_n_distance - expected_c_to_n) >= tolerance) + 1
    bonded_segments = np.split(residues, idx_after_gap)  # each segment now has no gap inside it
    longest = max(bonded_segments, key=len)  # longest segment with no gap
    return longest.reshape(-1, 3)  # back to a flat list of atoms


def vector_rejection(vectors, unit_axis):
    """The part of each vector orthogonal to unit_axis."""
    return vectors - (vectors * unit_axis).sum(-1, keepdims=True) * unit_axis


def backbone_to_angles(atoms):
    """A backbone can be described by six angles per residue instead of coordinates; each
    atom contributes a bond angle and a dihedral, both fixed by the three bonds before it."""
    bonds = atoms[1:] - atoms[:-1]  # [n_atoms - 1, 3]
    # Four atoms, three bonds: atom A --first--> atom B --second--> atom C --third--> atom D
    first, second, third = bonds[:-2], bonds[1:-1], bonds[2:]  # [n_atoms - 3, 3] each

    # The angle at atom C, between the second and third bonds
    direction_to_previous_atom, direction_to_next_atom = normalize(-second), normalize(third)
    cosines = (direction_to_previous_atom * direction_to_next_atom).sum(-1)
    bond_angles = np.arccos(np.clip(cosines, -1.0, 1.0))  # [n_atoms - 3]

    # Split the first and third bonds into components parallel and orthogonal to the second;
    # the dihedral is the angle between the orthogonal components
    axis = normalize(second)
    perp_first = vector_rejection(-first, axis)
    perp_third = vector_rejection(third, axis)
    perp_first_turned = np.cross(axis, perp_first)  # a quarter turn (90 deg) about the axis
    scaled_cosine = (perp_first * perp_third).sum(-1)  # |perp_first| * |perp_third| * cos
    scaled_sine = (perp_first_turned * perp_third).sum(-1)  # |perp_first| * |perp_third| * sin
    dihedrals = np.arctan2(scaled_sine, scaled_cosine)

    per_atom = np.stack([bond_angles, dihedrals], axis=-1)  # [n_atoms - 3, 2]
    return per_atom.reshape(-1, 6)  # three atoms per residue, two angles each


def place_next_atom(previous_three, bond_length, bond_angle, dihedral):
    """Place the next atom from the previous three and the angles relative to them."""
    atom_a, atom_b, atom_c = previous_three

    # An orthonormal frame at atom C
    bond_axis = normalize(atom_c - atom_b)  # along the B to C bond
    normal = normalize(np.cross(atom_b - atom_a, bond_axis))  # perpendicular to the A, B, C plane
    in_plane = np.cross(normal, bond_axis)  # in the A, B, C plane
    frame = np.stack([bond_axis, in_plane, normal], axis=-1)

    # Spherical to Cartesian in that frame: bond_length is the radius, bond_angle the polar
    # angle from -bond_axis, dihedral the azimuth from in_plane
    offset_in_frame = np.stack([-bond_length * np.cos(bond_angle),
                                bond_length * np.sin(bond_angle) * np.cos(dihedral),
                                bond_length * np.sin(bond_angle) * np.sin(dihedral)], axis=-1)
    return atom_c + np.einsum("ij,j->i", frame, offset_in_frame)  # to world axes


def angles_to_backbone(angles, bond_lengths, first_bond_angle):
    """Convert a backbone from its angle representation to its coordinate representation.
    bond_lengths[i] is the bond arriving at atom i of a residue (0 = N, 1 = CA, 2 = C)."""
    angles_per_atom = angles.reshape(-1, 2)

    # --- Residue 0 (N - CA - C) ---
    # Angles fix shape but not absolute position or orientation: atom 0's position, the
    # direction 0 to 1 and the plane 0-1-2 are arbitrary, a frame the angles are invariant to
    atoms = np.zeros((len(angles_per_atom) + 3, 3))  # atom 0 (N) at the origin
    # Atom 1 (CA) one bond along x (arbitrary); another direction would only rotate the result
    atoms[1] = [bond_lengths[1], 0.0, 0.0]
    # Atom 2 (C) one bond away from atom 1, at first_bond_angle between vectors
    # atom 1 -> atom 0 (-x) and atom 1 -> atom 2; z = 0 picks their plane (arbitrary)
    atoms[2] = atoms[1] + bond_lengths[2] * np.array(
        [-math.cos(first_bond_angle), math.sin(first_bond_angle), 0.0])

    for k, (bond_angle, dihedral) in enumerate(angles_per_atom, start=3):
        atoms[k] = place_next_atom(atoms[k - 3:k], bond_lengths[k % 3], bond_angle, dihedral)
    return atoms


def wrap_to_circle(angles):
    """Python's modulo lands in [0, 2pi), so shift by pi before and after to reach [-pi, pi)."""
    return (angles + math.pi) % (2 * math.pi) - math.pi


def circular_mean(angles):
    """E.g. +pi and -pi are close but average to 0, so take the mean unit vector's direction."""
    return np.arctan2(np.sin(angles).mean(axis=0), np.cos(angles).mean(axis=0))


def load_backbone_angles(data_dir, c_to_n_bond_length, min_residues=40):
    """Read every protein backbone in data_dir into its angle representation, dropping chains
    shorter than min_residues, then split 80/10/10 into train, valid and test."""
    min_atoms = 3 * min_residues
    paths = sorted(glob.glob(os.path.join(data_dir, "*")))
    chains = []
    for path in tqdm(paths, desc="parsing CATH"):
        atoms = longest_bonded_segment(parse_backbone(path), c_to_n_bond_length)
        if len(atoms) - 3 >= min_atoms:  # the angle representation drops residue 0 (3 atoms)
            chains.append(backbone_to_angles(atoms))
    random.Random(0).shuffle(chains)
    n_train, n_valid = int(0.8 * len(chains)), int(0.1 * len(chains))
    return chains[:n_train], chains[n_train:n_train + n_valid], chains[n_train + n_valid:]


class BackboneAngles(Dataset):
    """Angles centered on the training set's circular mean, cropped at random to a fixed
    length and padded, with a mask marking the real residues."""

    def __init__(self, chains, angle_mean, max_residues):
        self.max_residues = max_residues
        self.centered_chains = [wrap_to_circle(chain - angle_mean) for chain in chains]

    def __len__(self):
        return len(self.centered_chains)

    def __getitem__(self, index):
        chain = self.centered_chains[index]
        if len(chain) > self.max_residues:
            start = random.randint(0, len(chain) - self.max_residues)
            chain = chain[start:start + self.max_residues]
        angles = torch.zeros(self.max_residues, chain.shape[1])
        mask = torch.zeros(self.max_residues, dtype=torch.bool)
        angles[:len(chain)] = torch.from_numpy(chain).float()
        mask[:len(chain)] = True
        return angles, mask


def cosine_alpha_bars(n_steps, device, offset=0.008):
    """The cosine noise schedule standard in diffusion models."""
    fractions = torch.linspace(0, n_steps, n_steps + 1, device=device) / n_steps
    squared_cosines = torch.cos((fractions + offset) / (1 + offset) * math.pi / 2) ** 2
    betas = (1 - squared_cosines[1:] / squared_cosines[:-1]).clamp(max=0.999)
    alpha_bars = torch.cumprod(1 - betas, dim=0)
    previous_alpha_bars = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
    return alpha_bars, previous_alpha_bars


class FourierTimeEmbedding(nn.Module):
    """Random Fourier features: the frequencies are drawn once and never trained."""

    def __init__(self, dim, scale=16.0):
        super().__init__()
        self.register_buffer("frequencies", torch.randn(dim // 2) * scale)

    def forward(self, timesteps):
        phases = 2 * math.pi * timesteps[:, None].float() * self.frequencies[None]
        return torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1)


class RelativeSelfAttention(nn.Module):
    """Self-attention with a learned key per separation between two residues, so position
    enters only through how far apart residues are, never through where they sit."""

    def __init__(self, dim, n_heads, max_residues):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        # The signed separation j - i between two residues has 2 * max_residues - 1 possible values
        self.learned_keys = nn.Embedding(2 * max_residues - 1, self.head_dim)
        positions = torch.arange(max_residues)
        # Stores, for every pair (i, j), the separation j - i, shifted because indexing starts at 0
        key_indices = positions - positions[:, None] + max_residues - 1
        self.register_buffer("key_indices", key_indices)

    def forward(self, hidden, mask):
        n_batch, length, dim = hidden.shape
        qkv = self.qkv(hidden).reshape(n_batch, length, 3, self.n_heads, self.head_dim)
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4)
        separation_keys = self.learned_keys(self.key_indices[:length, :length])
        logits = queries @ keys.transpose(-2, -1) + torch.einsum(
            "bhid,ijd->bhij", queries, separation_keys)
        logits = logits / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attended = logits.softmax(dim=-1) @ values
        return self.projection(attended.transpose(1, 2).reshape(n_batch, length, dim))


class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, max_residues, dropout=0.1):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.attention = RelativeSelfAttention(dim, n_heads, max_residues)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(),
                                 nn.Linear(mlp_dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        hidden = hidden + self.dropout(self.attention(self.norm_attention(hidden), mask))
        return hidden + self.dropout(self.mlp(self.norm_mlp(hidden)))


class AngleDenoiser(nn.Module):
    """Predicts the noise that was added to each residue's six angles."""

    def __init__(self, dim, mlp_dim, n_layers, n_heads, max_residues, n_angles_per_residue):
        super().__init__()
        self.embed = nn.Linear(n_angles_per_residue, dim)
        self.time_embedding = FourierTimeEmbedding(dim)
        self.blocks = nn.ModuleList(
            EncoderBlock(dim, n_heads, mlp_dim, max_residues) for _ in range(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim),
                                  nn.Linear(dim, n_angles_per_residue))

    def forward(self, angles, timesteps, mask):
        hidden = self.embed(angles) + self.time_embedding(timesteps)[:, None]
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.head(self.norm(hidden))


def diffusion_loss(model, angles, mask, alpha_bars, transition=0.1 * math.pi):
    timesteps = torch.randint(0, alpha_bars.shape[0], (angles.shape[0],), device=angles.device)
    noise = torch.randn_like(angles)
    sqrt_alpha_bar = alpha_bars[timesteps].sqrt()[:, None, None]
    sqrt_one_minus_alpha_bar = (1 - alpha_bars[timesteps]).sqrt()[:, None, None]
    # Angles repeat every 2pi: 179 and -179 differ by 2, not 358, hence wrap_to_circle (Sec. 3.2)
    noised = wrap_to_circle(sqrt_alpha_bar * angles + sqrt_one_minus_alpha_bar * noise)
    error = wrap_to_circle(model(noised, timesteps, mask) - noise)
    per_angle = F.smooth_l1_loss(error, torch.zeros_like(error), beta=transition,  # Sec. 3.2
                                 reduction="none")
    return (per_angle * mask[..., None]).sum() / (mask.sum() * angles.shape[-1])


@torch.no_grad()
def sample_backbones(model, lengths, max_residues, n_angles_per_residue, alpha_bars,
                     previous_alpha_bars, device):
    alphas = alpha_bars / previous_alpha_bars
    posterior_variance = (1 - alphas) * (1 - previous_alpha_bars) / (1 - alpha_bars)
    mask = torch.zeros(len(lengths), max_residues, dtype=torch.bool, device=device)
    for row, length in enumerate(lengths):
        mask[row, :length] = True
    angles = wrap_to_circle(torch.randn(len(lengths), max_residues, n_angles_per_residue,
                                        device=device))

    for step in tqdm(range(alpha_bars.shape[0] - 1, -1, -1), desc="sampling"):
        timesteps = torch.full((len(lengths),), step, device=device, dtype=torch.long)
        predicted_noise = model(angles, timesteps, mask)
        scaled_noise = (1 - alphas[step]) / (1 - alpha_bars[step]).sqrt() * predicted_noise
        denoised = (angles - scaled_noise) / alphas[step].sqrt()  # the posterior mean
        if step > 0:
            denoised = denoised + posterior_variance[step].sqrt() * torch.randn_like(angles)
        angles = wrap_to_circle(denoised)
    return angles, mask


def phi_psi_pairs(angles, real_residue_mask, phi_column=5, psi_column=1):
    """The (phi, psi) of each residue, flattened over all chains, e.g. for a Ramachandran plot.
    The six columns are theta2, psi, theta3, omega, theta1, phi."""
    real_pair_mask = real_residue_mask[:, :-1] & real_residue_mask[:, 1:]
    # phi spans C-N-CA-C and psi N-CA-C-N, so psi reaches into the next residue and lands a
    # row later: pair phi from row i with psi from row i + 1
    return (wrap_to_circle(angles[:, :-1, phi_column])[real_pair_mask],
            wrap_to_circle(angles[:, 1:, psi_column])[real_pair_mask])


def plot_results(traces, natural_phi_psi, generated_phi_psi, save_path, n_columns=6):
    n_trace_rows = math.ceil(len(traces) / n_columns)
    figure = plt.figure(figsize=(3.0 * n_columns, 3.4 * n_trace_rows + 5.0))
    grid = figure.add_gridspec(n_trace_rows + 1, n_columns)

    for panel, trace in enumerate(traces):  # One 3D panel per trace
        axes = figure.add_subplot(grid[panel // n_columns, panel % n_columns], projection="3d")
        axes.plot(trace[:, 0], trace[:, 1], trace[:, 2], lw=2.5, color="#3b6ea5")
        axes.set_axis_off()
        axes.set_box_aspect((1, 1, 1))
        if panel == n_columns // 2:  # one title, over the middle of the top row
            axes.set_title("Generated backbones", fontsize=26)

    half = n_columns // 2  # The two Ramachandran scatters
    for column, (name, (phi, psi)) in enumerate([("Natural", natural_phi_psi),
                                                 ("Generated", generated_phi_psi)]):
        axes = figure.add_subplot(grid[-1, column * half:(column + 1) * half])
        axes.scatter(phi, psi, s=0.6, alpha=0.15, color="#3b6ea5", edgecolors="none")
        axes.set_xlim(-math.pi, math.pi)
        axes.set_ylim(-math.pi, math.pi)
        axes.set_xlabel("phi", fontsize=16)
        axes.set_ylabel("psi", fontsize=16)
        axes.set_title(f"{name} Ramachandran", fontsize=22)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    data_dir = "data/dompdb"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c_to_n, n_to_ca, ca_to_c = 1.329, 1.458, 1.525
    bond_lengths = [c_to_n, n_to_ca, ca_to_c]  # indexed by k % 3, the position of atom k
    first_bond_angle = 1.941  # N to CA to C
    n_angles_per_residue, max_residues, batch_size, n_diffusion_steps = 6, 128, 64, 1000
    shortest_sampled_length, n_samples_per_length, n_structures_shown = 50, 10, 12
    # Schedule spans 10,000 epochs, early stop at 1,488 (Sec. 3.3)
    n_epochs, schedule_epochs = 1_488, 10_000

    train_chains, _, test_chains = load_backbone_angles(data_dir, c_to_n)
    angle_mean = circular_mean(np.concatenate(train_chains, axis=0))
    train_set = BackboneAngles(train_chains, angle_mean, max_residues)
    test_set = BackboneAngles(test_chains, angle_mean, max_residues)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                              drop_last=True, pin_memory=torch.cuda.is_available())

    model = AngleDenoiser(dim=384, mlp_dim=768, n_layers=12, n_heads=12,
                          max_residues=max_residues,
                          n_angles_per_residue=n_angles_per_residue).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
    alpha_bars, previous_alpha_bars = cosine_alpha_bars(n_diffusion_steps, device)

    total_steps = schedule_epochs * len(train_loader)
    warmup_steps = total_steps // 10
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(
        step / max(1, warmup_steps),
        max(0.0, (total_steps - step) / (total_steps - warmup_steps))))

    for epoch in tqdm(range(n_epochs)):
        for angles, mask in train_loader:
            loss = diffusion_loss(model, angles.to(device), mask.to(device), alpha_bars)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    model.eval()
    lengths = [n for n in range(shortest_sampled_length, max_residues)
               for _ in range(n_samples_per_length)]
    sampled, sampled_mask = sample_backbones(model, lengths, max_residues, n_angles_per_residue,
                                             alpha_bars, previous_alpha_bars, device)
    generated = wrap_to_circle(sampled.cpu().numpy() + angle_mean)
    generated_mask = sampled_mask.cpu().numpy()

    held_out = [test_set[i] for i in range(min(len(lengths), len(test_set)))]
    natural = wrap_to_circle(torch.stack([a for a, _ in held_out]).numpy() + angle_mean)
    natural_mask = torch.stack([m for _, m in held_out]).numpy()

    shown_idx = np.linspace(0, len(generated) - 1, n_structures_shown).astype(int)
    alpha_carbons = slice(1, None, 3)  # only keep CA, every third atom from index 1 of N, CA, C
    traces = [angles_to_backbone(generated[i][generated_mask[i]], bond_lengths,
                                 first_bond_angle)[alpha_carbons] for i in shown_idx]
    plot_results(traces, phi_psi_pairs(natural, natural_mask),
                 phi_psi_pairs(generated, generated_mask),
                 save_path="Imgs/foldingdiff.png")
