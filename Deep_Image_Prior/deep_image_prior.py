import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt


class D(nn.Module):

    def __init__(self, in_channels, nd, kd):
        super(D, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, nd, kd, stride=2, padding=1, bias=True),  # Downsample
            nn.BatchNorm2d(nd), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nd, nd, kd, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(nd), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.model(x)


class S(nn.Module):

    def __init__(self, in_channels, ns, ks):
        super(S, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ns, ks, 1, padding=0, bias=True),
            nn.BatchNorm2d(ns), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.model(x)


class U(nn.Module):

    def __init__(self, in_channels, nu, ku):
        super(U, self).__init__()

        self.model = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.Conv2d(in_channels, nu, ku, 1, padding=1,
                                             bias=True),
                                   nn.BatchNorm2d(nu), nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(nu, nu, 1, 1, padding=0, bias=True),
                                   nn.BatchNorm2d(nu), nn.LeakyReLU(0.2, inplace=True),
                                   nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.d1 = D(3, 8, 3)
        self.d2 = D(8, 16, 3)
        self.d3 = D(16, 32, 3)
        self.d4 = D(32, 64, 3)
        self.d5 = D(64, 128, 3)

        self.u1 = U(16, 8, 3)
        self.u2 = U(32, 16, 3)
        self.u3 = U(64, 32, 3)
        self.u4 = U(128 + 4, 64, 3)
        self.u5 = U(128 + 4, 128, 3)

        self.s4 = S(32, 4, 1)
        self.s5 = S(64, 4, 1)

        self.conv_out = nn.Conv2d(8, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        h = self.d1(x)
        h = self.d2(h)
        h = self.d3(h)
        skip3 = self.s4(h)
        h = self.d4(h)
        skip4 = self.s5(h)
        h = self.d5(h)

        h = self.u5(torch.cat((skip4[:, :, 4:-4, 6:-6], h), dim=1))
        h = self.u4(torch.cat((skip3[:, :, 8:-8, 12:-12], h), dim=1))
        h = self.u3(h)
        h = self.u2(h)
        h = self.u1(h)

        return torch.sigmoid(self.conv_out(h))


if __name__ == "__main__":
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    image = Image.open('Imgs/snail.jpg')
    w, h = image.size
    image = image.resize((w - w % 32, h - h % 32), resample=Image.LANCZOS)
    image = torch.from_numpy(np.array(image) / 255.0).unsqueeze(0).float()
    corrupted_img = (image + torch.randn_like(image) * .1).clip(0, 1)
    corrupted_img = corrupted_img.transpose(2, 3).transpose(1, 2)
    z = torch.randn(corrupted_img.shape) * .1

    for epoch in tqdm(range(2400)):
        img_pred = model.forward(z)
        loss = torch.nn.functional.mse_loss(img_pred, corrupted_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(figsize=(18, 3.5))
    plt.subplot(1, 3, 1)
    plt.imshow(corrupted_img[0].transpose(0, 1).transpose(1, 2).data.numpy())
    plt.title('Input', fontsize=15)
    plt.subplot(1, 3, 2)
    plt.imshow(img_pred[0].transpose(0, 1).transpose(1, 2).data.numpy())
    plt.title('Prediction', fontsize=15)
    plt.subplot(1, 3, 3)
    plt.imshow(image[0].data.numpy())
    plt.title('Ground truth', fontsize=15)
    plt.savefig('Imgs/deep_image_prior.png')
