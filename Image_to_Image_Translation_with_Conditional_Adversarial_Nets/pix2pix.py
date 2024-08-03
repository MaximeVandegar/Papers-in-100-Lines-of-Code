import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from os import listdir
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


class DownConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super(UpConvBlock, self).__init__()
        layers = [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ]
        if dropout:
            layers.append(nn.Dropout(p=0.5, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv_in = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                 bias=False)

        self.down1 = DownConvBlock(64, 128)
        self.down2 = DownConvBlock(128, 256)
        self.down3 = DownConvBlock(256, 512)
        self.down4 = DownConvBlock(512, 512)
        self.down5 = DownConvBlock(512, 512)
        self.down6 = DownConvBlock(512, 512)

        self.middle = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.up1 = UpConvBlock(1024, 512, dropout=True)
        self.up2 = UpConvBlock(1024, 512, dropout=True)
        self.up3 = UpConvBlock(1024, 512, dropout=True)
        self.up4 = UpConvBlock(1024, 256)
        self.up5 = UpConvBlock(512, 128)
        self.up6 = UpConvBlock(256, 64)

        self.outermost = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x0 = self.conv_in(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        x = self.middle(x6)

        x = self.up1(torch.cat((x, x6), dim=1))
        x = self.up2(torch.cat((x, x5), dim=1))
        x = self.up3(torch.cat((x, x4), dim=1))
        x = self.up4(torch.cat((x, x3), dim=1))
        x = self.up5(torch.cat((x, x2), dim=1))
        x = self.up6(torch.cat((x, x1), dim=1))
        return self.outermost(torch.cat((x, x0), dim=1))


class PatchGAN(nn.Module):

    def __init__(self):
        super(PatchGAN, self).__init__()

        self.network = nn.Sequential(
                            nn.Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)))

    def forward(self, x):
        return self.network(x)


class Dataset():

    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.files = [f for f in listdir(data_path) if f.endswith(".jpg")]
        self.len = len(self.files)

        self.transform_list = []
        self.transform_list += [transforms.ToTensor()]
        self.transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        transform_list = self.transform_list.copy()
        if np.random.rand() > 0.5:  # Make sure to apply the same transform to the input and output
            transform_list = [transforms.RandomHorizontalFlip(p=1.0)] + transform_list
        transform = transforms.Compose(transform_list)

        AB = Image.open(self.data_path + "/" + self.files[index])
        w, h = AB.size
        B = AB.crop((0, 0, w // 2, h))  # Output
        A = AB.crop((w // 2, 0, w, h))  # Input
        return transform(A), transform(B)


def train(netD, netG, optimizer_G, optimizer_D, scheduler_G, scheduler_D, dataloader, NB_EPOCHS, device, lambda_L1=100,
          criterionL1=torch.nn.L1Loss()):

    for epoch in tqdm(range(NB_EPOCHS)):
        for batch in dataloader:
            real_A, real_B = batch
            real_A, real_B = real_A.to(device), real_B.to(device)

            fake_B = netG(real_A)

            pred_fake = netD(torch.cat((real_A, fake_B), 1).detach())
            loss_D_fake = torch.nn.functional.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = netD(real_AB)
            loss_D_real = torch.nn.functional.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            pred_fake = netD(torch.cat((real_A, fake_B), 1))
            loss_G_GAN = torch.nn.functional.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = lambda_L1 * criterionL1(fake_B, real_B)
            loss_G = loss_G_GAN + loss_G_L1
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
        scheduler_D.step()
        scheduler_G.step()


def init_weights(m):  # define the initialization function
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == "__main__":
    device = 'cuda'
    netG = UNet().to(device)
    netD = PatchGAN().to(device)
    netG.apply(init_weights)
    netD.apply(init_weights)
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Retrieved from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L53
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - 100) / float(100 + 1)
        return lr_l
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    dataloader = DataLoader(Dataset("datasets/facades/train"), batch_size=1, shuffle=True, num_workers=0)
    train(netD, netG, optimizer_G, optimizer_D, scheduler_G, scheduler_D, dataloader, 200, device)

    netG.eval()
    test_dataloader = DataLoader(Dataset("datasets/facades/test"), batch_size=1, shuffle=True, num_workers=0)
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(6, 8), dpi=150)
    for ax, col_title in zip(axes[0], ["Input", "Ground truth", "Output"]):
        ax.set_title(col_title)
    for idx, batch in enumerate(test_dataloader):
        input, ground_truth = batch
        input, ground_truth = input.to(device), ground_truth.to(device)
        output = netG(input)
        axes[idx, 0].imshow(input[0].cpu().transpose(0, 1).transpose(1, 2).numpy() / 2. + .5)
        axes[idx, 0].axis('off')
        axes[idx, 1].imshow(ground_truth[0].cpu().transpose(0, 1).transpose(1, 2).numpy() / 2. + .5)
        axes[idx, 1].axis('off')
        axes[idx, 2].imshow(output[0].data.cpu().transpose(0, 1).transpose(1, 2).numpy() / 2. + .5)
        axes[idx, 2].axis('off')
        if idx == 3:
            break
    plt.savefig("Imgs/pix2pix.png", bbox_inches="tight")
