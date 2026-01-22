import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer
from safetensors.torch import load_file
from model import FirstStageModel, FrozenCLIPEmbedder, DiffusionModel, ModelEMA


class SD15_Minimal(nn.Module):

    def __init__(self):
        super().__init__()

        self.first_stage_model = FirstStageModel()
        self.cond_stage_model = FrozenCLIPEmbedder()
        self.model = DiffusionModel()
        self.model_ema = ModelEMA()

        # diffusion schedule buffers (placeholders; load from ckpt)
        for name in ["betas", "alphas_cumprod", "alphas_cumprod_prev",
                     "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                     "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod",
                     "sqrt_recipm1_alphas_cumprod", "posterior_variance",
                     "posterior_log_variance_clipped", "posterior_mean_coef1",
                     "posterior_mean_coef2"]:
            self.register_buffer(name, torch.zeros(1000))

    @torch.no_grad()
    def encode_text(self, input_ids):  # (B, 77) -> (B, 77, 768)
        return self.cond_stage_model(input_ids)

    @torch.no_grad()
    def decode_vae(self, z):  # (B, 4, H/8, W/8) -> (B, 3, H, W)
        return self.first_stage_model.decode(z)

    def unet(self, x, t, context):
        return self.model.diffusion_model(x, t, context)


def tokenize(tokenizer, prompts, device):
    toks = tokenizer(prompts, padding="max_length", truncation=True,
                     max_length=77, return_tensors="pt")
    return toks.input_ids.to(device)  # (B,77)


@torch.no_grad()
def predict_eps(m, z, t_int, ctx_c, ctx_u, cfg=7.5):
    B = z.shape[0]
    t = torch.full((B,), int(t_int), device=z.device, dtype=torch.long)
    eps_u = m.unet(z, t, ctx_u)
    eps_c = m.unet(z, t, ctx_c)
    return eps_u + cfg * (eps_c - eps_u)


@torch.no_grad()
def text2img_ddpm(m, prompts, tokenizer, negative_prompts=None, steps=1000,
                  cfg=7.5, H=512, W=512, device='cuda'):
    """ Pure DDPM ancestral sampling """
    B = len(prompts)
    if negative_prompts is None:
        negative_prompts = [""] * B

    # Text -> context (B, 77, 768)
    ids_c = tokenize(tokenizer, prompts, device)
    ids_u = tokenize(tokenizer, negative_prompts, device)
    ctx_c = m.encode_text(ids_c)
    ctx_u = m.encode_text(ids_u)

    h, w = H // 8, W // 8
    z = torch.randn(B, 4, h, w, device=device)

    time_steps = torch.arange(999, -1, -1, device=device)

    for t in tqdm(time_steps):
        t_int = int(t.item())

        eps = predict_eps(m, z, t_int, ctx_c, ctx_u, cfg=cfg)

        # x0 prediction from eps
        a_t = m.alphas_cumprod[t_int].view(B, 1, 1, 1)
        x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()

        # posterior mean: coef1 * x0 + coef2 * x_t
        mean = m.posterior_mean_coef1[t_int].view(
            B, 1, 1, 1) * x0 + m.posterior_mean_coef2[t_int].view(B, 1, 1, 1) * z

        # sample z_{t-1}
        if t_int > 0:
            var = m.posterior_variance[t_int].clamp(min=1e-20).view(B, 1, 1, 1)
            noise = torch.randn_like(z)
            z = mean + var.sqrt() * noise
        else:
            z = mean

    x = m.decode_vae(z / SCALE).clamp(-1, 1)  # (B, 3, H, W)
    return x


if __name__ == "__main__":
    device = "cuda"
    SCALE = 0.18215  # SD 1.x latent scale

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tensor_dict = load_file("../v1-5-pruned-emaonly.safetensors")
    m = SD15_Minimal().to(device)
    m.load_state_dict(tensor_dict, strict=True)
    m.eval()
    prompt = "a photo of a corgi puppy"

    imgs = text2img_ddpm(m, [prompt], tokenizer, steps=1000, cfg=7.5, device=device)
    img = ((imgs[0] + 1) / 2).permute(1, 2, 0).cpu()
    Image.fromarray((255. * img.numpy()).astype(np.uint8)).save('Imgs/sample.png')
