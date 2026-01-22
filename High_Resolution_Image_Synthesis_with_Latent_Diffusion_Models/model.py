import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, vocab_size=49408, hidden_size=768, max_position_embeddings=77):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        # position_ids is stored as a buffer to match SD state dict
        self.register_buffer("position_ids",
                             torch.arange(max_position_embeddings).unsqueeze(0))

    def forward(self, input_ids):
        pos_ids = self.position_ids[:, :input_ids.shape[1]]
        return self.token_embedding(input_ids) + self.position_embedding(pos_ids)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, causal_mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if causal_mask is not None:
            attn_scores = attn_scores + causal_mask  # mask should be -inf above diagonal
        attn = attn_scores.softmax(dim=-1)
        y = torch.matmul(attn, v)  # B, h, T, d
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = SelfAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.mlp = CLIPMLP(hidden_size, hidden_size * mlp_ratio)

    def forward(self, x, causal_mask):
        x = x + self.self_attn(self.layer_norm1(x), causal_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, num_layers=12):
        super().__init__()
        self.layers = nn.ModuleList(
            [CLIPEncoderLayer(hidden_size, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        # causal mask for 77 tokens
        T = x.shape[1]
        mask = torch.full((1, 1, T, T), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class CLIPTextModel_(nn.Module):
    def __init__(self, vocab_size=49408, hidden_size=768, max_position_embeddings=77,
                 num_heads=12, num_layers=12):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        self.encoder = CLIPEncoder(hidden_size, num_heads, num_layers)
        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        x = self.encoder(x)
        x = self.final_layer_norm(x)
        return x  # (B, 77, 768)


class CLIPTextTransformerWrapper(nn.Module):
    """ Matches: cond_stage_model.transformer.text_model.* """
    def __init__(self):
        super().__init__()
        self.text_model = CLIPTextModel_()


class FrozenCLIPEmbedder(nn.Module):
    """ Matches: cond_stage_model.transformer.text_model... """
    def __init__(self):
        super().__init__()
        self.transformer = CLIPTextTransformerWrapper()

    def forward(self, input_ids):
        return self.transformer.text_model(input_ids)


# -----------------------------
# ------------ VAE ------------
# -----------------------------

class ResnetBlockVAE(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = nn.Conv2d(
            in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlockVAE(nn.Module):
    def __init__(self, channels, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=1e-5)
        self.q = nn.Conv2d(channels, channels, 1, bias=True)
        self.k = nn.Conv2d(channels, channels, 1, bias=True)
        self.v = nn.Conv2d(channels, channels, 1, bias=True)
        self.proj_out = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = self.norm(x)
        q = self.q(h_).reshape(b, c, h * w)
        k = self.k(h_).reshape(b, c, h * w)
        v = self.v(h_).reshape(b, c, h * w)
        attn = torch.einsum("bct,bcs->bts", q, k) * (1.0 / math.sqrt(c))
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bcs,bst->bct", v, attn).reshape(b, c, h, w)
        out = self.proj_out(out)
        return x + out


class DownsampleVAE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpsampleVAE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class EncoderVAE(nn.Module):
    def __init__(self, in_channels=3, ch=128):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)

        self.down = nn.ModuleList()
        d0 = nn.Module()
        d0.block = nn.ModuleList([ResnetBlockVAE(ch, ch), ResnetBlockVAE(ch, ch)])
        d0.downsample = DownsampleVAE(ch)
        self.down.append(d0)

        d1 = nn.Module()
        d1.block = nn.ModuleList([ResnetBlockVAE(ch, ch*2), ResnetBlockVAE(ch*2, ch*2)])
        d1.downsample = DownsampleVAE(ch*2)
        self.down.append(d1)

        d2 = nn.Module()
        d2.block = nn.ModuleList([ResnetBlockVAE(ch*2, ch*4), ResnetBlockVAE(ch*4, ch*4)])
        d2.downsample = DownsampleVAE(ch*4)
        self.down.append(d2)

        d3 = nn.Module()
        d3.block = nn.ModuleList([ResnetBlockVAE(ch*4, ch*4), ResnetBlockVAE(ch*4, ch*4)])
        self.down.append(d3)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlockVAE(ch*4, ch*4)
        self.mid.attn_1 = AttnBlockVAE(ch*4)
        self.mid.block_2 = ResnetBlockVAE(ch*4, ch*4)

        self.norm_out = nn.GroupNorm(32, ch*4, eps=1e-5)
        self.conv_out = nn.Conv2d(ch*4, 8, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for d in self.down:
            for block in d.block:
                h = block(h)
            if hasattr(d, "downsample"):
                h = d.downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class DecoderVAE(nn.Module):
    def __init__(self, out_channels=3, ch=128):
        super().__init__()
        self.conv_in = nn.Conv2d(4, ch*4, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlockVAE(ch*4, ch*4)
        self.mid.attn_1 = AttnBlockVAE(ch*4)
        self.mid.block_2 = ResnetBlockVAE(ch*4, ch*4)

        u3 = nn.Module()
        u3.block = nn.ModuleList([ResnetBlockVAE(ch*4, ch*4),
                                  ResnetBlockVAE(ch*4, ch*4),
                                  ResnetBlockVAE(ch*4, ch*4)])
        u3.upsample = UpsampleVAE(ch*4)

        u2 = nn.Module()
        u2.block = nn.ModuleList([ResnetBlockVAE(ch*4, ch*4),
                                  ResnetBlockVAE(ch*4, ch*4),
                                  ResnetBlockVAE(ch*4, ch*4)])
        u2.upsample = UpsampleVAE(ch*4)

        u1 = nn.Module()
        u1.block = nn.ModuleList([ResnetBlockVAE(ch*4, ch*2),
                                  ResnetBlockVAE(ch*2, ch*2),
                                  ResnetBlockVAE(ch*2, ch*2)])
        u1.upsample = UpsampleVAE(ch*2)

        u0 = nn.Module()
        u0.block = nn.ModuleList([ResnetBlockVAE(ch*2, ch),
                                  ResnetBlockVAE(ch, ch),
                                  ResnetBlockVAE(ch, ch)])
        self.up = nn.ModuleList([u0, u1, u2, u3])

        self.norm_out = nn.GroupNorm(32, ch, eps=1e-5)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for idx in [3, 2, 1, 0]:
            blk = self.up[idx]
            for b in blk.block:
                h = b(h)
            if hasattr(blk, "upsample"):
                h = blk.upsample(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class FirstStageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderVAE()
        self.decoder = DecoderVAE()
        self.quant_conv = nn.Conv2d(8, 8, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)

    @staticmethod
    def _split_mu_logvar(m):
        return torch.chunk(m, 2, dim=1)

    def encode(self, x):
        moments = self.encoder(x)
        moments = self.quant_conv(moments)
        return self._split_mu_logvar(moments)

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x, sample_posterior=True):
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar) if sample_posterior else mu
        x_rec = self.decode(z)
        return x_rec, mu, logvar, z


# -----------------------------
# ----------- UNET ------------
# -----------------------------

class Downsample(nn.Module):
    """ Matches: *.op.* in UNet input blocks """
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    """ Matches: *.conv.* in UNet output blocks """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class GEGLU(nn.Module):
    """ Used inside FeedForward to match ff.net.0.proj.* keys """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """ Matches keys like: ff.net.0.proj.*, ff.net.2.* """
    def __init__(self, dim, mult=4, glu=True):
        super().__init__()
        inner = int(dim * mult)  # in SD checkpoints, mult=4 with GEGLU -> 8x params on first proj
        if glu:
            self.net = nn.ModuleList([GEGLU(dim, inner), nn.Identity(), nn.Linear(inner, dim)])
        else:
            self.net = nn.ModuleList([nn.Linear(dim, inner), nn.GELU(), nn.Linear(inner, dim)])

    def forward(self, x):
        x = self.net[0](x)
        x = self.net[2](x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=None, dim_head=64):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.num_heads = num_heads if num_heads is not None else dim // dim_head
        inner = self.num_heads * self.dim_head

        self.to_q = nn.Linear(dim, inner, bias=False)
        kdim = context_dim if context_dim is not None else dim
        self.to_k = nn.Linear(kdim, inner, bias=False)
        self.to_v = nn.Linear(kdim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=True))

    def forward(self, x, context=None):
        if context is None:
            context = x
        B, N, _ = x.shape
        M = context.shape[1]
        h = self.num_heads
        d = self.dim_head

        q = self.to_q(x).view(B, N, h, d).transpose(1, 2)       # B, h, N, d
        k = self.to_k(context).view(B, M, h, d).transpose(1, 2)  # B, h, M, d
        v = self.to_v(context).view(B, M, h, d).transpose(1, 2)  # B, h, M, d

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, h * d)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """ Matches: *.transformer_blocks.0.attn1.*, *.attn2.*, *.ff.net.*, *.norm1/2/3.* """
    def __init__(self, dim, n_heads=8, d_head=64, context_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, num_heads=n_heads, dim_head=d_head)  # self-attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, num_heads=n_heads, dim_head=d_head)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=4, glu=True)

    def forward(self, x, context=None):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    """ Matches blocks with keys: *.norm.*, *.proj_in.*, *.transformer_blocks.0.*, *.proj_out.* """
    def __init__(self, channels, n_heads, d_head=64, depth=1, context_dim=768):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_heads, d_head, context_dim) for _ in range(depth)]
        )
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = self.proj_out(x)
        return x + x_in


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch=1280):
        super().__init__()
        self.in_layers = nn.ModuleList([nn.GroupNorm(32, in_ch, eps=1e-5), nn.SiLU(),
                                        nn.Conv2d(in_ch, out_ch, 3, padding=1)])
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(temb_ch, out_ch))
        self.out_layers = nn.ModuleList([nn.GroupNorm(32, out_ch, eps=1e-5), nn.SiLU(),
                                         nn.Identity(), nn.Conv2d(out_ch, out_ch, 3, padding=1)])
        self.skip_connection = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.in_layers[2](self.in_layers[1](self.in_layers[0](x)))
        h = h + self.emb_layers(temb)[:, :, None, None]
        h = self.out_layers[3](self.out_layers[1](self.out_layers[0](h)))
        return h + (x if isinstance(self.skip_connection, nn.Identity) else self.skip_connection(x))


class UNetModel(nn.Module):
    def __init__(self, in_channels=4, model_channels=320, out_channels=4,
                 channel_mult=(1, 2, 4, 4), num_res_blocks=2, context_dim=768):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels

        self.time_embed = nn.Sequential(nn.Linear(model_channels, model_channels * 4),
                                        nn.SiLU(),
                                        nn.Linear(model_channels * 4, model_channels * 4))

        self.input_blocks = nn.ModuleList()
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks.append(nn.ModuleList([nn.Conv2d(in_channels, ch, 3, padding=1)]))
        input_block_chans = [ch]

        downsample_factor = 1
        attn_downsample_set = {1, 2, 4}

        def make_st(ch):
            n_heads = 8
            d_head = ch // n_heads
            return SpatialTransformer(ch, n_heads, d_head=d_head, depth=1, context_dim=context_dim)

        for level, mult in enumerate(channel_mult):
            for i in range(num_res_blocks):
                layers = nn.ModuleList()
                layers.append(ResBlock(ch, int(mult * model_channels)))
                ch = int(mult * model_channels)
                if downsample_factor in attn_downsample_set:
                    layers.append(make_st(ch))
                self.input_blocks.append(layers)
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([Downsample(ch)]))
                input_block_chans.append(ch)
                downsample_factor *= 2

        self.middle_block = nn.ModuleList([ResBlock(ch, ch), make_st(ch), ResBlock(ch, ch)])

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = nn.ModuleList()
                layers.append(ResBlock(ch + ich, int(model_channels * mult)))
                ch = int(model_channels * mult)
                ds_here = downsample_factor
                if ds_here in attn_downsample_set:
                    layers.append(make_st(ch))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    downsample_factor //= 2
                self.output_blocks.append(layers)

        self.out = nn.ModuleList([nn.GroupNorm(32, ch, eps=1e-5), nn.SiLU(),
                                  nn.Conv2d(ch, out_channels, 3, padding=1)])

    def forward(self, x, timesteps, context: Optional[torch.Tensor] = None):
        half = self.model_channels // 2
        freqs = torch.exp(-math.log(10000.0) *
                          torch.arange(0, half, device=x.device, dtype=torch.float32) / (half - 1))
        args = timesteps.float()[:, None] * freqs[None, :]
        t_embed_320 = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 320)
        temb = self.time_embed(t_embed_320)

        hs = []
        h = x
        for block in self.input_blocks:
            # initial conv
            if len(block) == 1 and isinstance(block[0], nn.Conv2d):
                h = block[0](h)
                hs.append(h)
                continue
            # downsample
            if len(block) == 1 and isinstance(block[0], Downsample):
                h = block[0](h)
                hs.append(h)
                continue
            # resblock (+ optional ST)
            h = block[0](h, temb)
            if len(block) > 1:
                h = block[1](h, context)
            hs.append(h)

        h = self.middle_block[0](h, temb)
        h = self.middle_block[1](h, context)
        h = self.middle_block[2](h, temb)

        for block in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block[0](h, temb)
            if len(block) > 1 and isinstance(block[1], SpatialTransformer):
                h = block[1](h, context)
                if len(block) == 3:
                    h = block[2](h)
            elif len(block) > 1:
                h = block[1](h)

        h = self.out[2](self.out[1](self.out[0](h)))
        return h


class DiffusionModel(nn.Module):
    """Holds UNet under .diffusion_model to match 'model.diffusion_model.*"""
    def __init__(self):
        super().__init__()
        self.diffusion_model = UNetModel()


class ModelEMA(nn.Module):
    """Just a holder so checkpoint keys model_ema.decay / num_updates load."""
    def __init__(self):
        super().__init__()
        self.register_buffer("decay", torch.tensor(0.0))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))
