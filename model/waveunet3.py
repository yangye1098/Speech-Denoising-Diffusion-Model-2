import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, log, sqrt

# Positional Encoding and FiLM are borrowed from wavegrad
# subject to change

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
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

        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]

        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1,  1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel_size, stride, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv1d(dim, dim_out, kernel_size, padding='same')
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, kernel_size, stride, noise_level_emb_dim=1, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
                noise_level_emb_dim, dim_out, use_affine_level)


        self.block1 = Block(dim, dim_out, kernel_size, stride, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, kernel_size, stride, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv1d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv1d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv1d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, ntime = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, ntime)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bncl, bncx -> bnlx", query, key
        ).contiguous() / sqrt(channel)
        attn = attn.view(batch, n_head, ntime, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, ntime, ntime)

        out = torch.einsum("bnlx, bncx -> bncl", attn, value).contiguous()
        out = self.out(out.view(batch, channel, ntime))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, kernel_size, stride, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, kernel_size, stride, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


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

    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride,
                 conv_type, upsample_kernel_size=4, upsample_stride=2,
                 noise_level_channel=1, norm_groups=32, dropout=0, use_attn=True):
        super(UpsamplingBlock, self).__init__()
        assert(upsample_stride > 1)

        # UPSAMPLING
        self.upconv = UpsampleLayer(n_inputs, upsample_kernel_size, upsample_stride, conv_type)

        self.pre_shortcut = nn.ModuleList([ResnetBlocWithAttn(n_inputs, n_shortcut, kernel_size, stride,
                                                              noise_level_emb_dim=noise_level_channel,
                                                              norm_groups=norm_groups,
                                                              dropout=dropout, with_attn=use_attn
                                                              )])

        self.post_shortcut = nn.ModuleList([ResnetBlocWithAttn(n_shortcut, n_outputs, kernel_size, stride,
                                                               noise_level_emb_dim=noise_level_channel,
                                                               norm_groups=norm_groups,
                                                               dropout=dropout, with_attn=use_attn
                                                               )])

    def forward(self, x, shortcut, noise_level):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for block in self.pre_shortcut:
            upsampled = block(upsampled, noise_level)

        # combine shortcuts
        combined = upsampled + shortcut
        # Combine  features
        for block in self.post_shortcut:
            combined = block(combined, noise_level)
        return combined

    def get_output_size(self, input_size):

        # Upsampling convs
        curr_size = self.upconv.get_output_size(input_size)
        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride,
                 conv_type, downsample_kernel_size=4, downsample_stride=2,
                 noise_level_channel=1, norm_groups=32, dropout=0, use_attn=True):
        super(DownsamplingBlock, self).__init__()
        self.pre_shortcut = nn.ModuleList([ResnetBlocWithAttn(n_inputs, n_shortcut, kernel_size, stride,
                                                              noise_level_emb_dim=noise_level_channel,
                                                              norm_groups=norm_groups,
                                                              dropout=dropout, with_attn=use_attn
                                                              )])

        self.post_shortcut = nn.ModuleList([ResnetBlocWithAttn(n_shortcut, n_outputs, kernel_size, stride,
                                                               noise_level_emb_dim=noise_level_channel,
                                                               norm_groups=norm_groups,
                                                               dropout=dropout, with_attn=use_attn
                                                               )])

        # above conv doesn't change size at last dim
        self.downconv = DownsampleLayer(n_outputs, downsample_kernel_size, downsample_stride, conv_type)

        # CONV 2 with decimation

    def forward(self, x, noise_level):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for block in self.pre_shortcut:
            shortcut = block(shortcut, noise_level)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for block in self.post_shortcut:
            out = block(out, noise_level)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_output_size(self, input_size):
        curr_size = self.downconv.get_output_size(input_size)

        return curr_size


class Waveunet3(nn.Module):
    def __init__(self, num_inputs, num_channels, downconv_kernel_size, upconv_kernel_size, bottleneck_kernel_size, conv_stride,
                 conv_type, downsample_kernel_size=4, upsample_kernel_size=4, resample_stride=2,
                 with_noise_level_emb=False, norm_groups=32, with_attn=True, dropout=0):
        super(Waveunet3, self).__init__()
        self.num_levels = len(num_channels)

        assert((downsample_kernel_size- resample_stride) % 2 == 0)
        assert((upsample_kernel_size- resample_stride) % 2 == 0)
        assert(num_channels[0]==norm_groups)

        if with_noise_level_emb:
            raise NotImplementedError
        else:
            noise_level_channel = 1
            self.noise_level_mlp = None

        module = nn.Module()

        module.downsampling_blocks = nn.ModuleList()
        module.upsampling_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            if i == 0:
                in_ch = num_inputs
                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], downconv_kernel_size, conv_stride,
                                      conv_type, downsample_kernel_size, resample_stride,
                                      noise_level_channel, norm_groups=in_ch, dropout=dropout, use_attn=with_attn))
            else:
                in_ch = num_channels[i]
                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], downconv_kernel_size, conv_stride,
                                      conv_type, downsample_kernel_size, resample_stride,
                                      noise_level_channel, norm_groups=norm_groups, dropout=dropout, use_attn=with_attn))

        for i in range(self.num_levels - 1, 0, -1):
            module.upsampling_blocks.append(
                UpsamplingBlock(num_channels[i], num_channels[i-1], num_channels[i-1], upconv_kernel_size, conv_stride,
                                conv_type, upsample_kernel_size, resample_stride,
                                noise_level_channel, norm_groups, dropout, with_attn))

        module.bottlenecks = nn.ModuleList(
            [ResnetBlocWithAttn(num_channels[-1], num_channels[-1], bottleneck_kernel_size, conv_stride, noise_level_channel, norm_groups, dropout, with_attn),
             ResnetBlocWithAttn(num_channels[-1], num_channels[-1], bottleneck_kernel_size, conv_stride, noise_level_channel, norm_groups, dropout, with_attn=False )])

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

        if self.noise_level_mlp is not None:
            noise_level = self.noise_level_mlp(noise_level)

        module = self.waveunet
        shorts = []
        out = torch.concat([x, y_t], dim=1)

        # DOWNSAMPLING BLOCKS

        for block in module.downsampling_blocks:
            out, short = block(out, noise_level)
            shorts.append(short)

        # BOTTLENECK CONVOLUTION
        for block in module.bottlenecks:
            out = block(out, noise_level)

        # UPSAMPLING BLOCKS
        for block, short in zip(module.upsampling_blocks, reversed(shorts)):
            out = block(out, short, noise_level)

        # OUTPUT CONV
        out = module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out
