import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import floor, log

# Positional Encoding and FiLM are borrowed from wavegrad
# subject to change

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, noise_level):
        """
        Arguments:
          x:
              (shape: [N,C,T], dtype: float32)
          noise_level:
              (shape: [N], dtype: float32)

        Returns:
          noise_level:
              (shape: [N,C,T], dtype: float32)
        """
        if noise_level.ndim > 1:
            noise_level = torch.squeeze(noise_level)
        N = x.shape[0]
        T = x.shape[2]

        return (x + self._build_encoding(noise_level)[:, :, None])

    def _build_encoding(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FiLM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.encoding = PositionalEncoding(input_size)
        self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
        self.output_conv = nn.Conv1d(input_size, output_size * 2, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_conv.weight)
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.input_conv.bias)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, noise_scale):
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.encoding(x, noise_scale)
        shift, scale = torch.chunk(self.output_conv(x), 2, dim=1)
        return shift, scale


class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, conv_type, padding='same', transpose=False):
        """
        conv_type: "gn", "bn", "normal"
        padding parameter only valid for normal conv
        """
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type

        # How many channels should be normalised as one group if GroupNorm is activated
        # WARNING: Number of channels has to be divisible by this number!
        NORM_CHANNELS = 8

        if self.transpose:
            self.padding = (self.kernel_size - self.stride)//2
            self.filter = nn.ConvTranspose1d(n_inputs, n_outputs, self.kernel_size, self.stride, padding=self.padding)
        else:
            self.padding = padding
            self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, self.stride, padding=self.padding)

        if conv_type == "gn":
            assert(n_outputs % NORM_CHANNELS == 0)
            self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)
        elif conv_type == "bn":
            self.norm = nn.BatchNorm1d(n_outputs, momentum=0.01)
        # Add you own types of variations here!

    def forward(self, x):
        # Apply the convolution
        if self.conv_type == "gn" or self.conv_type == "bn":
            temp1 = self.filter(x)
            temp2 = self.norm(temp1)
            out = F.relu(temp2)
        else: # Add your own variations here with elifs conditioned on "conv_type" parameter!
            assert(self.conv_type == "normal")
            out = F.leaky_relu(self.filter(x))
        return out

    def get_output_size(self, input_size):
        # Transposed
        if self.transpose:
            if isinstance(self.padding, float) or isinstance(self.padding, int):
                assert(input_size > 1)
                output_size = floor((input_size - 1)*self.stride -2*self.padding + self.kernel_size)
            else:
                raise NotImplementedError
        else:
            if self.padding == 'same':
                output_size = input_size
            elif isinstance(self.padding, float) or isinstance(self.padding, int)  :
                output_size = floor((input_size + 2*self.padding - self.kernel_size + self.stride)/self.stride)
            else:
                raise NotImplementedError(f'padding: {self.padding}')

        # Conv
        assert (output_size > 0)
        return output_size

class UpsampleLayer(nn.Module):
    def __init__(self, n_channels, upsample_kernel_size, upsample_stride, conv_type):
        super().__init__()
        self.up = ConvLayer(n_channels, n_channels, upsample_kernel_size, upsample_stride, conv_type, transpose=True)

    def forward(self, x):
        return self.up(x)

    def get_output_size(self, input_size):
        return self.up.get_output_size(input_size)


class DownsampleLayer(nn.Module):
    def __init__(self, n_channels, downsample_kernel_size, downsample_stride, conv_type):
        super().__init__()
        padding = (downsample_kernel_size-downsample_stride)//2
        self.down = ConvLayer(n_channels, n_channels, downsample_kernel_size, downsample_stride, conv_type, padding=padding)

    def forward(self, x):
        return self.down(x)

    def get_output_size(self, input_size):
        return self.down.get_output_size(input_size)

class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, depth, conv_type,
                 upsample_kernel_size=4, resample_stride=2):
        super(UpsamplingBlock, self).__init__()
        assert(resample_stride > 1)

        # CONV 1 for UPSAMPLING
        self.upconv = UpsampleLayer(n_inputs, upsample_kernel_size, resample_stride, conv_type)

        # following conv doesn't change size at last dim

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList(
                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1) ] +
                [ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)])

    def forward(self, x, film_shift, film_scale):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # combine film features
        combined = upsampled
        # Combine  features
        for conv in self.post_shortcut_convs:
            combined = conv(film_scale*combined + film_shift)
        return combined

    def get_output_size(self, input_size):

        # Upsampling convs
        curr_size = self.upconv.get_output_size(input_size)
        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, depth, conv_type, downsample_kernel_size=4, downsample_stride=2):
        super(DownsamplingBlock, self).__init__()

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in
                                                  range(depth - 1)])

        # above conv doesn't change size at last dim
        self.downconv = DownsampleLayer(n_outputs, downsample_kernel_size, downsample_stride, conv_type)

        # CONV 2 with decimation

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_output_size(self, input_size):
        curr_size = self.downconv.get_output_size(input_size)

        return curr_size

class Waveunet2(nn.Module):
    def __init__(self, num_inputs, num_channels, downconv_kernel_size, upconv_kernel_size, bottleneck_kernel_size, conv_stride, conv_type, depth=1,
                 downsample_kernel_size=4, upsample_kernel_size=4, resample_stride=2):
        super(Waveunet2, self).__init__()
        self.num_levels = len(num_channels)
        self.downconv_kernel_size = downconv_kernel_size
        self.upconv_kernel_size = upconv_kernel_size
        self.num_inputs = num_inputs
        self.depth = depth
        self.resample_stride = resample_stride
        self.upsample_kernel_size = upsample_kernel_size
        self.downsample_kernel_size = downsample_kernel_size

        # Only odd filter kernels allowed
        assert(downconv_kernel_size % 2 == 1)
        assert(upconv_kernel_size % 2 == 1)
        assert((downsample_kernel_size- resample_stride) % 2 == 0)
        assert((upsample_kernel_size- resample_stride) % 2 == 0)

        module = nn.Module()

        module.downsampling_blocks = nn.ModuleList()
        module.upsampling_blocks = nn.ModuleList()
        module.film_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            in_ch = num_inputs if i == 0 else num_channels[i]

            module.downsampling_blocks.append(
                DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], downconv_kernel_size, depth, conv_type, downsample_kernel_size, resample_stride))
            module.film_blocks.append(FiLM(num_channels[i], num_channels[i]))

        for i in range(self.num_levels - 1, 0, -1):
            module.upsampling_blocks.append(
                UpsamplingBlock(num_channels[i], num_channels[i-1], num_channels[i-1],upconv_kernel_size, depth, conv_type, upsample_kernel_size, resample_stride))

        module.bottlenecks = nn.ModuleList(
            [ConvLayer(num_channels[-1], num_channels[-1], bottleneck_kernel_size, 1, conv_type) for _ in range(depth)])

        # Output conv
        module.output_conv = nn.Conv1d(num_channels[0], 1, 1)

        self.waveunet = module


    def check_output_size(self, input_size):
        if input_size < 0:
            return

        module = self.waveunet
        curr_size = input_size
        print(f'Input: {curr_size}')
        for idx, block in enumerate(module.downsampling_blocks):
            curr_size = block.get_output_size(curr_size)
        print(f'after down sample: {curr_size}')

        # Bottleneck-Conv
        for block in module.bottlenecks:
            curr_size = block.get_output_size(curr_size)
        print(f'after bottleneck: {curr_size}')

        for idx, block in enumerate(reversed(module.upsampling_blocks)):
            curr_size = block.get_output_size(curr_size)
        print(f'after up sample: {curr_size}')

        assert(curr_size == input_size)

    def forward(self, x, y_t, noise_level):
        """
        A forward pass through Wave-U-Net
        :param x: Conditional input [B, 1, T]
        :param y_t: signal from last iteration [B, 1, T]
        :param noise_level: noise level
        """

        module = self.waveunet
        films = []
        out = torch.concat([x, y_t], dim=1)

        # DOWNSAMPLING BLOCKS

        for block, film in zip(module.downsampling_blocks, module.film_blocks):
            out, short = block(out)
            films.append(film(short, noise_level))

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for block, (film_shift, film_scale) in zip(module.upsampling_blocks, reversed(films)):
            out = block(out, film_shift, film_scale)

        # OUTPUT CONV
        out = module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)

        return out
