import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt, log

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        step = torch.arange(self.dim//2)/(self.dim//2)
        #self.embedding_vector = torch.exp(-log(1e4) * step.unsqueeze(0))
        self.embedding_vector = 10.0 ** (step * 4.0/63)

        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        x = self._build_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _build_embedding(self, diffusion_step):
        self.embedding_vector = self.embedding_vector.to(diffusion_step.device)
        encoding = diffusion_step * self.embedding_vector
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1) # [B, self.dim]
        return encoding


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, fix_in=False, split=True):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        self.split = split
        self.fix_in = fix_in
        if self.split:
            # print("2 individual Conv1d")
            self.output_projection = Conv1d(residual_channels, residual_channels, 1)
            self.output_residual = Conv1d(residual_channels, residual_channels, 1)
        else:
            if self.fix_in:
                print("1 big and 1 small Conv1d")
                self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
                self.output_residual = Conv1d(residual_channels, residual_channels, 1)
            else:
                print("1 big Conv1d")
                self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        if self.split:
            residual = self.output_residual(y)
            skip = self.output_projection(y)
        elif self.fix_in:
            # calculate residual from non-fixed parameter
            residual = self.output_residual(y)
            # calculate skip from fixed parameter
            y = self.output_projection(y)
            _, skip = torch.chunk(y, 2, dim=1)
        else:
            y = self.output_projection(y)
            residual, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self,
                 num_samples,
                 num_timesteps,
                 n_mels,
                 residual_channels=64,
                 residual_layers=30,
                 dilation_cycle_length=10,
                 ):
        super().__init__()
        self.input_projection = Conv1d(1, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding()
        self.spectrogram_upsampler = SpectrogramUpsampler(n_mels)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(n_mels, residual_channels, 2 ** (i % dilation_cycle_length))

            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spectrogram, audio, diffusion_step):
        """
            spectrogram: [B, 1, n_freq, n_time]
            audio: [B, 1, T]
            diffusion_step [B, 1, 1]
        """
        diffusion_step = diffusion_step.squeeze(-1)
        x = self.input_projection(audio)
        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, spectrogram, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)

        return x
