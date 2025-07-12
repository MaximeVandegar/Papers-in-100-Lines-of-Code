import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


class DiffusionModel:
    def __init__(self, T, model, device):
        self.T = T
        self.function_approximator = model.to(device)
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def training(self, batch_size, optimizer):
        pass  # See https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/blob/main/Denoising_Diffusion_Probabilistic_Models/diffusion_models.py#L31

    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32), use_tqdm=True):
        pass  # See https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/blob/main/Denoising_Diffusion_Probabilistic_Models/diffusion_models.py#L54

    @torch.no_grad()
    def pndm_sampling(self, n_samples=1, image_channels=1, img_size=(32, 32), n_steps=50, use_tqdm=True):
        """
        Algorithm 2 in https://arxiv.org/pdf/2202.09778
        """

        step_size = self.T // n_steps
        timesteps = [self.T - i * step_size for i in range(n_steps)]
        if timesteps[-1] != 0:
            timesteps.append(0)

        x = torch.randn((n_samples, image_channels, *img_size), device=self.device)
        eps_buffer = []
        self.counter = 0

        iterator = tqdm(zip(timesteps[:-1], timesteps[1:])) if use_tqdm else zip(timesteps[:-1], timesteps[1:])
        for t, t_next in iterator:
            if self.counter < 3:
                x, e_t = self._step_prk(x, t, t_next)
            else:
                x, e_t = self._step_plms(x, t, t_next, eps_buffer)

            # maintain a buffer of the last 3 raw ε’s
            eps_buffer.append(e_t)
            if len(eps_buffer) > 3:
                eps_buffer.pop(0)
            self.counter += 1
        return x

    def _step_prk(self, x, t, t_next):
        """
        Do one PRK update Eq(13)
        """
        # half‐step & full‐step times
        delta = t - t_next
        tm = int(t - delta/2)

        t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=self.device)
        tm_vec = torch.full((x.shape[0],), tm, dtype=torch.long, device=self.device)
        tnext_vec = torch.full((x.shape[0],), t_next, dtype=torch.long, device=self.device)

        e1 = self.function_approximator(x, t_vec)
        x1 = self._phi(x, e1, t, tm)
        e2 = self.function_approximator(x1, tm_vec)
        x2 = self._phi(x, e2, t, tm)
        e3 = self.function_approximator(x2, tm_vec)
        x3 = self._phi(x, e3, t, t_next)
        e4 = self.function_approximator(x3, tnext_vec)

        e_prime = (e1 + 2*e2 + 2*e3 + e4) / 6.0
        x_next = self._phi(x, e_prime, t, t_next)
        return x_next, e_prime

    def _step_plms(self, x, t, t_next, eps_buffer):
        """
        Do one PLMS update Eq(12):
        """
        t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=self.device)
        e_t = self.function_approximator(x, t_vec)

        past = torch.stack([e_t,
                            eps_buffer[-1],
                            eps_buffer[-2],
                            eps_buffer[-3]], dim=0)

        e_prime = (55 * past[0] - 59 * past[1] + 37 * past[2] - 9 * past[3]) / 24.0
        x_next = self._phi(x, e_prime, t, t_next)
        return x_next, e_t

    def _phi(self, x, eps, t, t_next):
        #  Eq(11) from Sec.3.3 of the paper
        if t > 0:
            ab_t = self.alpha_bar[t-1]
        else:
            ab_t = torch.tensor(1.0, device=self.device)
        if t_next > 0:
            ab_next = self.alpha_bar[t_next-1]
        else:
            ab_next = torch.tensor(1.0, device=self.device)

        denom = ab_t.sqrt() * (((1 - ab_next).sqrt()) * ab_t.sqrt() + ((1 - ab_t).sqrt()) * ab_next.sqrt())
        return (ab_next.sqrt() / ab_t.sqrt()) * x - ((ab_next - ab_t) / denom) * eps


if __name__ == "__main__":
    model = torch.load('model_ddpm_mnist')
    diffusion = DiffusionModel(1000, model, 'cuda')

    nb_images = 81
    samples = diffusion.pndm_sampling(n_samples=nb_images, n_steps=50, use_tqdm=True)

    plt.figure(figsize=(17, 17))
    for i in range(nb_images):
        plt.subplot(9, 9, 1+i)
        plt.axis('off')
        img = samples[i].squeeze().clamp(0, 1).cpu().numpy()
        plt.imshow(img, cmap='gray')
    plt.savefig('Imgs/pndms_samples.png')
    plt.show()
