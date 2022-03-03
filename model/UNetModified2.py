import math
import torch
from torch import nn

class SignalToFrames(nn.Module):
    """
    it is for torch tensor
    """

    def __init__(self, n_samples, F=512, stride=256):
        super().__init__()

        assert((n_samples-F) % stride == 0)
        self.n_samples = n_samples
        self.n_frames = (n_samples - F) // stride + 1
        self.idx_mat = torch.empty((self.n_frames, F), dtype=torch.long)
        start = 0
        for i in range(self.n_frames):
            self.idx_mat[i, :] = torch.arange(start, start+F)
            start += stride


    def forward(self, sig):
        """
            sig: [B, 1, n_samples]
            return: [B, 1, nframes, F]
        """
        return sig[:, :, self.idx_mat]

    def overlapAdd(self, input):
        """
            reverse the segementation process
            input [B, 1, n_frames, F]
            return [B, 1, n_samples]
        """

        output = torch.zeros((input.shape[0], input.shape[1], self.n_samples), device=input.device)
        for i in range(self.n_frames):
            output[:, :, self.idx_mat[i, :]] += input[:, :, i, :]

        return output



# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
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
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
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
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x






class UNetModified2(nn.Module):
    def __init__(
        self,
        num_samples,
        in_channel=2,
        out_channel=1,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 3, 4, 5),
        attn_layer=(4),
        res_blocks=3,
        dropout=0,
        segment_len=128,
        segment_stride=64,
        with_noise_level_emb=True,
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        self.segment = SignalToFrames(num_samples, segment_len, segment_stride)
        # first conv raise # channels to inner_channel

        self.downs = nn.ModuleList([nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)])

        # record the number of output channels
        feat_channels = [inner_channel]

        num_mults = len(channel_mults)

        for ind in range(num_mults-1):
            use_attn = (ind in attn_layer)

            n_channel_in = inner_channel * channel_mults[ind]
            n_channel_out = inner_channel * channel_mults[ind+1]

            for _ in range(0, res_blocks):
                self.downs.append(ResnetBlocWithAttn(
                    n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(n_channel_out)
                n_channel_in = n_channel_out

            # doesn't change # channels
            self.downs.append(Downsample(n_channel_out))
            feat_channels.append(n_channel_out)

        n_channel_out = n_channel_in
        self.mid = nn.ModuleList([
                ResnetBlocWithAttn(n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=False)
            ])
        self.ups = nn.ModuleList([])


        for ind in reversed(range(num_mults)):
            use_attn = ind in attn_layer

            n_channel_in = inner_channel * channel_mults[ind]
            n_channel_out = n_channel_in
            if ind >= 1:
                # combine down sample layer skip connection
                self.ups.append(ResnetBlocWithAttn(
                    n_channel_in + feat_channels.pop(), n_channel_out, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))

                # up sample
                self.ups.append(Upsample(n_channel_out))

            if ind == 0:
                n_channel_out = inner_channel
            else:
                n_channel_out = inner_channel * channel_mults[ind-1]

            # combine resnet block skip connection
            for _ in range(0, res_blocks):
                self.ups.append(ResnetBlocWithAttn(
                    n_channel_in+feat_channels.pop(), n_channel_out, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                n_channel_in = n_channel_out





        n_channel_in = n_channel_out
        self.final_conv = Block(n_channel_in, out_channel, groups=norm_groups)

    def forward(self, x, y_t, noise_level):
        """
            x: [B, 1, T]
            y_t: [B, 1, T]
            time: [B, 1, 1]
        """
        noise_level = torch.unsqueeze(noise_level, -1)
        b = x.shape[0]
        # naive segmentation

        x = self.segment(x)
        y_t = self.segment(y_t)

        input = torch.cat([x, y_t], dim=1)
        if self.noise_level_mlp is not None:
            t = self.noise_level_mlp(noise_level)
        else:
            t = None



        feats = []
        for layer in self.downs:
            print(input.shape)
            if isinstance(layer, ResnetBlocWithAttn):
                input = layer(input, t)
            else:
                input = layer(input)
            feats.append(input)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                input = layer(input, t)
            else:
                input = layer(input)
        print('after mid', input.shape)
        print('up')
        for layer in self.ups:
            print(input.shape)
            if isinstance(layer, ResnetBlocWithAttn):
                input = layer(torch.cat((input, feats.pop()), dim=1), t)
            else:
                input = layer(input)

        output = self.final_conv(input)
        output = self.segment.overlapAdd(output)
        return output
