import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from .conv import Conv

class SAGamma(nn.Module):
    def __init__(self, G_min=0.8, G_max=1.2, epsilon=1e-6):
        """ Self-Adaptive Gamma Correction Module.
        Args:
            G_min: Minimum gamma value.
            G_max: Maximum gamma value.
            epsilon: Small constant to prevent division by zero.
        """
        super(SAGamma, self).__init__()
        self.G_min = G_min
        self.G_max = G_max
        self.epsilon = epsilon

    def forward(self, x):
        # Convert to grayscale: gray = 0.299 * R + 0.587 * G + 0.114 * B
        gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        gray_mean = torch.mean(gray, dim=[1, 2], keepdim=True)  # Batch-wise mean
        gray_std = torch.std(gray, dim=[1, 2], keepdim=True)  # Batch-wise std

        # Compute adaptive gamma
        gamma = self.G_min + (self.G_max - self.G_min) * gray_mean / (gray_mean + gray_std.clamp_min(self.epsilon))
        gamma = gamma.clamp(self.G_min, self.G_max)
        # print(gamma)
        # Apply gamma correction
        x = torch.pow(x + self.epsilon, gamma.unsqueeze(1))  # Apply per-channel
        return x


class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """CARAFE Upsampling Module."""
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, padding=(k_up // 2) * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * c_mid * h * w
        W = self.enc(W)  # b * (scale^2 * k_up^2) * h * w
        W = self.pix_shf(W)  # b * k_up^2 * h_ * w_
        W = torch.softmax(W, dim=1)  # Normalize

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * k_up^2 * c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * k_up^2 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # Dynamic reassembly
        return X


class GCModule(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """Gamma-Carafe Module (GCModule) = Dynamic Gamma Correction + CARAFE."""
        super(GCModule, self).__init__()
        self.gamma_corr = SAGamma()
        self.carafe = CARAFE(c, k_enc, k_up, c_mid, scale)

    def forward(self, x):
        x = self.gamma_corr(x)  # Dynamic Gamma Correction
        x = self.carafe(x)  # CARAFE Upsampling
        return x
