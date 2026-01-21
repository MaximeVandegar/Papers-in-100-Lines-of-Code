import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel


@torch.no_grad()
def sample(tokenizer, text_encoder, vae, unet, prompts, size=512, device="cuda:0"):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              clip_sample=False, set_alpha_to_one=False, steps_offset=1)

    pipe = StableDiffusionPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
                                   unet=unet, scheduler=scheduler, safety_checker=None,
                                   feature_extractor=None, requires_safety_checker=False).to(device)
    pipe.set_progress_bar_config(disable=True)

    out = pipe(prompts, num_images_per_prompt=1, guidance_scale=7.5, num_inference_steps=25,
               height=size, width=size)
    return out.images


class DreamBoothPairedDataset(torch.utils.data.Dataset):
    def __init__(self, instance_path, class_path, tokenizer, instance_prompt, class_prompt, size=512):
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        self.instance_paths = [p for p in Path(instance_path).iterdir() if p.suffix.lower() == ".jpg"]
        self.class_paths = [p for p in Path(class_path).iterdir() if p.suffix.lower() == ".jpg"]

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            ])

    def __len__(self):
        return len(self.class_paths)

    def _tok(self, prompt: str) -> torch.Tensor:
        return self.tokenizer(prompt, padding="max_length", truncation=True,
                              max_length=self.tokenizer.model_max_length,
                              return_tensors="pt").input_ids[0]

    def __getitem__(self, idx):
        inst_path = self.instance_paths[idx % len(self.instance_paths)]
        cls_path = self.class_paths[idx]

        inst_img = self.transform(Image.open(inst_path).convert("RGB"))
        cls_img = self.transform(Image.open(cls_path).convert("RGB"))

        inst_ids = self._tok(self.instance_prompt)
        cls_ids = self._tok(self.class_prompt)

        return {"instance_pixel_values": inst_img, "class_pixel_values": cls_img,
                "instance_input_ids": inst_ids, "class_input_ids": cls_ids}


if __name__ == "__main__":
    model_path = "runwayml/stable-diffusion-v1-5"
    device = "cuda"
    instance_folder = "data"
    class_images_folder = "corgi_puppy"
    instance_prompt = "a photo of sks corgi puppy"
    class_prompt = "a photo of a corgi puppy"
    num_class_images = 1000
    prior_loss_weight = 1.0
    max_train_steps = 1000
    vae_scale = 0.18215  # SD1.x latent scale

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(device)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device)
    alphas_cumprod = DDPMScheduler.from_config(
        model_path, subfolder="scheduler").alphas_cumprod.to(device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-6, betas=(0.9, 0.999))

    os.makedirs(class_images_folder, exist_ok=True)
    for i in tqdm(range(num_class_images), desc="Generating class images"):
        img = sample(tokenizer, text_encoder, vae, unet, [class_prompt], size=512, device=device)
        img[0].save(f"{class_images_folder}/class_image_{i}.jpg")

    train_dataset = DreamBoothPairedDataset(
        instance_path=instance_folder, class_path=class_images_folder, tokenizer=tokenizer,
        instance_prompt=instance_prompt, class_prompt=class_prompt, size=512)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                   shuffle=True, drop_last=True)

    unet.train()
    global_step = 0
    pbar = tqdm(total=max_train_steps, desc="Training")
    while global_step < max_train_steps:
        for batch in train_dataloader:
            if global_step >= max_train_steps:
                break

            with torch.no_grad():  # Encode to latents
                inst_latents = vae.encode(batch["instance_pixel_values"].to(device)).latent_dist.sample()
                cls_latents = vae.encode(batch["class_pixel_values"].to(device)).latent_dist.sample()
                inst_latents = inst_latents * vae_scale
                cls_latents = cls_latents * vae_scale

            # Sample timesteps
            timesteps = torch.randint(0, alphas_cumprod.shape[0], (inst_latents.shape[0],),
                                      device=device).long()

            inst_noise = torch.randn_like(inst_latents)
            cls_noise = torch.randn_like(cls_latents)
            alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            # Forward diffusion
            inst_noisy = sqrt_alpha_t * inst_latents + sqrt_one_minus_alpha_t * inst_noise
            cls_noisy = sqrt_alpha_t * cls_latents + sqrt_one_minus_alpha_t * cls_noise

            with torch.no_grad():  # Text conditioning
                inst_h = text_encoder(batch["instance_input_ids"].to(device))[0]
                cls_h = text_encoder(batch["class_input_ids"].to(device))[0]

            # Predict noise
            inst_pred = unet(inst_noisy, timesteps, inst_h).sample
            cls_pred = unet(cls_noisy,  timesteps, cls_h).sample

            loss_inst = F.mse_loss(inst_pred.float(), inst_noise.float(), reduction="mean")
            loss_prior = F.mse_loss(cls_pred.float(),  cls_noise.float(),  reduction="mean")
            loss = loss_inst + prior_loss_weight * loss_prior

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.update(1)
    pbar.close()

    test_prompts = ["a photo of sks corgi puppy in the Acropolis",
                    "a photo of sks corgi puppy swimming underwater",
                    "a photo of sks corgi puppy sleeping",
                    "a photo of sks corgi puppy in a doghouse",
                    "a photo of sks corgi puppy in a bucket",
                    "a photo of sks corgi puppy getting a haircut"]

    captions = ["in the Acropolis", "swimming", "sleeping", "in a doghouse",
                "in a bucket", "getting a haircut"]

    images = sample(tokenizer, text_encoder, vae, unet, test_prompts, size=512, device=device)

    plt.figure(figsize=(18, 7), facecolor="white")
    # Big left
    ax = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)
    ax.imshow(images[0])
    ax.axis("off")
    ax.set_title(captions[0], fontsize=14, pad=10)
    # Small middle 2x2
    small = [(1, (0, 2)), (2, (0, 3)), (3, (1, 2)), (4, (1, 3))]
    for idx, (r, c) in small:
        ax = plt.subplot2grid((2, 6), (r, c))
        ax.imshow(images[idx])
        ax.axis("off")
        ax.set_title(captions[idx], fontsize=12, pad=6)
    # Big right
    ax = plt.subplot2grid((2, 6), (0, 4), rowspan=2, colspan=2)
    ax.imshow(images[5])
    ax.axis("off")
    ax.set_title(captions[5], fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig("Imgs/sample_dreambooth.png")
