import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import lr_scheduler
import random
import torchvision.transforms as transforms
from PIL import Image
from os import listdir
from torch.utils.data import DataLoader
import numpy as np

class ResnetBlock(nn.Module):

    def __init__(self, ch):
        super(ResnetBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(ch, affine=False, track_running_stats=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(ch, affine=False, track_running_stats=False))

    def forward(self, x):
        return x + self.model(x)

class NLayerDiscriminator(nn.Module):

    def __init__(self):
        super(NLayerDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128, affine=False, track_running_stats=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256, affine=False, track_running_stats=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(512, affine=False, track_running_stats=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):

    def __init__(self, n_blocks=9):
        super(Generator, self).__init__()

        self.conv_in = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=True),
                                     nn.InstanceNorm2d(64, affine=False, track_running_stats=False),
                                     nn.ReLU(True))
        self.down = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
                                  nn.InstanceNorm2d(128, affine=False, track_running_stats=False),
                                  nn.ReLU(True),
                                  nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
                                  nn.InstanceNorm2d(256, affine=False, track_running_stats=False),
                                  nn.ReLU(True))
        self.middle = nn.Sequential(*torch.nn.ModuleList([ResnetBlock(256) for _ in range(n_blocks)]))
        self.upsampling = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1,
                                                           output_padding=1, bias=True),
                                        nn.InstanceNorm2d(128, affine=False, track_running_stats=False),
                                        nn.ReLU(True),
                                        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,
                                                           output_padding=1, bias=True),
                                        nn.InstanceNorm2d(64, affine=False, track_running_stats=False),
                                        nn.ReLU(True),
                                        nn.ReflectionPad2d(3), nn.Conv2d(64, 3, kernel_size=7, padding=0),
                                        nn.Tanh())

    def forward(self, x):
        return self.upsampling(self.middle(self.down(self.conv_in(x))))

class Buffer():

    def __init__(self, size=50):
        self.size = size
        self.buffer = []

    def query(self, image):
        if len(self.buffer) < self.size:
            self.buffer.append(image)
        else:
            if torch.rand(1).item() > 0.5:
                idx = random.randint(0, self.size - 1)
                poped_img = self.buffer[idx].clone()
                self.buffer[idx] = image
                return poped_img
        return image


def init_weights(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

class Dataset():

    def __init__(self, path_domain_x, path_domain_y):
        self.path_domain_x = path_domain_x
        self.path_domain_y = path_domain_y
        self.imgs_domain_x = [f for f in listdir(path_domain_x) if f.endswith(".jpg")]
        self.imgs_domain_y = [f for f in listdir(path_domain_y) if f.endswith(".jpg")]
        self.len = max(len(self.imgs_domain_x), len(self.imgs_domain_y))
        self.transform = transforms.Compose([
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Resize([286, 286], Image.BICUBIC),
                                             transforms.RandomCrop(256),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return self.len

    def __getitem__(self, index=None):
        index_x = random.randint(0, len(self.imgs_domain_x) - 1)
        index_y = random.randint(0, len(self.imgs_domain_y) - 1)
        A = Image.open(self.path_domain_x + "/" + self.imgs_domain_x[index_x]).convert("RGB")
        B = Image.open(self.path_domain_y + "/" + self.imgs_domain_y[index_y]).convert("RGB")
        return self.transform(A), self.transform(B)

def training_loop(G, F, Dx, Dy, optimizer_generators, optimizer_discriminators, scheduler_g,
                  scheduler_d, dataloader, nb_epochs=100, device='cpu', lambda_=10, lambda_identity=0.5):

    for _ in tqdm(range(nb_epochs)):
        for x, y in dataloader:

            x = x.to(device)
            y = y.to(device)
            fake_y = G(x)
            fake_x = F(y)
            reconstructed_x = F(fake_y)
            reconstructed_y = G(fake_x)

            # Training the generators
            # Compute generator loss (Y → X)
            cycle_loss_yxy = nn.functional.l1_loss(reconstructed_y, y)
            idt_loss_yx = nn.functional.l1_loss(G(y), y)
            d_score = Dx(fake_x)
            loss_g_yx = (nn.functional.mse_loss(d_score, torch.ones_like(d_score))
                         + lambda_ * cycle_loss_yxy + lambda_ * lambda_identity * idt_loss_yx)
            # Compute generator loss (X → Y)
            cycle_loss_xyx = nn.functional.l1_loss(reconstructed_x,  x)
            idt_loss_xy = nn.functional.l1_loss(F(x), x)
            d_score = Dy(fake_y)
            loss_g_xy = (nn.functional.mse_loss(d_score, torch.ones_like(d_score)) + lambda_ * cycle_loss_xyx
                         + lambda_ * lambda_identity * idt_loss_xy)
            loss_g = loss_g_yx + loss_g_xy
            optimizer_generators.zero_grad()
            loss_g.backward()
            optimizer_generators.step()

            # Training the discriminators
            fake_x_ = buffer_x.query(fake_x)
            fake_y_ = buffer_y.query(fake_y)
            pred_real_x = Dx(x)
            pred_real_y = Dy(y)
            pred_fake_x = Dx(fake_x_.detach())
            pred_fake_y = Dy(fake_y_.detach())
            loss_D_A = 0.5 * (nn.functional.mse_loss(pred_real_x, torch.ones_like(pred_real_x))
                              + nn.functional.mse_loss(pred_fake_x, torch.zeros_like(pred_fake_x)))
            loss_D_B = 0.5 * (nn.functional.mse_loss(pred_real_y, torch.ones_like(pred_real_y))
                              + nn.functional.mse_loss(pred_fake_y, torch.zeros_like(pred_fake_y)))
            optimizer_discriminators.zero_grad()
            loss_D_A.backward()
            loss_D_B.backward()
            optimizer_discriminators.step()
        scheduler_g.step()
        scheduler_d.step()

if __name__ == "__main__":
    device = 'cuda'
    Dx = NLayerDiscriminator().to(device)
    Dy = NLayerDiscriminator().to(device)
    Dx.apply(init_weights)
    Dy.apply(init_weights)
    G = Generator().to(device)
    F = Generator().to(device)
    G.apply(init_weights)
    F.apply(init_weights)

    buffer_x = Buffer()
    buffer_y = Buffer()
    optimizer_generators = optim.Adam(list(G.parameters()) + list(F.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_discriminators = optim.Adam(list(Dx.parameters()) + list(Dy.parameters()), lr=0.0002, betas=(0.5, 0.999))

    # Retrieved from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L53
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - 100) / float(100 + 1)
        return lr_l
    scheduler_g = lr_scheduler.LambdaLR(optimizer_generators, lr_lambda=lambda_rule)
    scheduler_d = lr_scheduler.LambdaLR(optimizer_discriminators, lr_lambda=lambda_rule)

    dataloader = DataLoader(Dataset("horse2zebra/horse2zebra/trainA", "horse2zebra/horse2zebra/trainB"), batch_size=1)
    training_loop(G, F, Dx, Dy, optimizer_generators, optimizer_discriminators, scheduler_g,
                  scheduler_d, dataloader, nb_epochs=200, device=device)

    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    original = torch.from_numpy(np.array(Image.open('horse2zebra/horse2zebra/testA/n02381460_1300.jpg'))) / 255.
    g_x = G(original[None].transpose(-1, -2).transpose(-2, -3).to(device))
    f_g_x = F(g_x)
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(g_x.transpose(2, 3).transpose(1, 3)[0].data.cpu().numpy() * .5 + .5)
    axes[0, 1].set_title("G(x)")
    axes[0, 2].imshow(f_g_x.transpose(2, 3).transpose(1, 3)[0].data.cpu().numpy() * .5 + .5)
    axes[0, 2].set_title("F(G(x))")

    original = torch.from_numpy(np.array(Image.open('horse2zebra/horse2zebra/testB/n02391049_5100.jpg'))) / 255.
    f_x = F(original[None].transpose(-1, -2).transpose(-2, -3).to(device))
    g_f_x = F(f_x)
    axes[1, 0].imshow(original)
    axes[1, 0].set_title("Original")
    axes[1, 1].imshow(f_x.transpose(2, 3).transpose(1, 3)[0].data.cpu().numpy() * .5 + .5)
    axes[1, 1].set_title("F(x)")
    axes[1, 2].imshow(g_f_x.transpose(2, 3).transpose(1, 3)[0].data.cpu().numpy() * .5 + .5)
    axes[1, 2].set_title("G(F(x))")
    for ax in axes.flat:
        ax.axis('off')

    plt.savefig('Imgs/cycle_gan.png', bbox_inches='tight')
    plt.close()
