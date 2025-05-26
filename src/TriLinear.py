import torch
import torch.nn as nn
import torch.nn.functional as F


def tricube_kernel(size):
    offset = (size - 1) / 2
    distances = torch.linspace(-offset, offset, steps=size) / offset  # normalize to [-1, 1]
    weights = (1 - torch.abs(distances) ** 3) ** 3
    weights = weights / weights.sum()
    return weights.to(dtype=torch.float32)

class TricubeSmoothing1D(nn.Module):
    def __init__(self, weights, stride=1):
        super(TricubeSmoothing1D, self).__init__()

        assert weights.ndim == 1, "Expected 1D tensor for weights"
        self.stride = stride

        weights = weights.view(1, 1, -1)
        self.register_buffer('weights', weights)

    def forward(self, x):
        B, T, C = x.shape
        kernel_size = self.weights.shape[-1]

        # Manual replicate padding
        pad_left = kernel_size // 2
        pad_right = kernel_size - 1 - pad_left
        front = x[:, 0:1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        x = torch.cat([front, x, end], dim=1)


        # Prepare for conv1d
        x = x.permute(0, 2, 1)
        # weights = self.weights.to(x.device)
        out = F.conv1d(x, self.weights.expand(C, 1, -1), stride=self.stride, groups=C)
        out = out.permute(0, 2, 1)
        return out


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, weights):
        super(series_decomp, self).__init__()
        self.tricube_smooth = TricubeSmoothing1D(weights, stride=1)

    def forward(self, x):
        out = self.tricube_smooth(x)
        residual = x - out
        return residual, out


class Model(nn.Module):
    def __init__(self, seq_length, output_size, kernel_frac=0.6666666666666666):
        super(Model, self).__init__()

        # decomposition Kernel Size

        kernel_size = max(3, int(round(kernel_frac * seq_length)))
        self.weight = tricube_kernel(kernel_size)
        self.decomposition = series_decomp(self.weight)
        self.linear_seasonal = nn.Linear(seq_length, output_size)
        self.linear_trend = nn.Linear(seq_length, output_size)

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)

        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)
