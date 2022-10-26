
import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)




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
    def __init__(self, dim, dim_out, dropout=0, norm_groups=32):
        super().__init__()

        self.block1 = Block(dim, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        return h + self.res_conv(x)

class SNRBlock(nn.Module):
    def __init__(self, dim, n_segment_in, len_segment_in, n_segment_out, norm_groups=32):
        super().__init__()
        # downsample to [dim, n_segment_in, 1]
        dim_out = dim*len_segment_in
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=len_segment_in, stride=len_segment_in),
            nn.GroupNorm(norm_groups, dim_out),
            Swish(),
        )
        self.dense = nn.Linear(dim*n_segment_in, n_segment_out)


    def forward(self, x):
        [B, C, N, L]  = x.shape
        h = self.block(x) # [B, C, N/L, 1]
        h = h.view(B, -1)
        h = self.dense(h)
        return h


class SNREstimator(nn.Module):
    def __init__(
            self,
            n_segments,
            segment_len,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 3, 4, 5),
            res_blocks=3,
            dropout=0,
    ):
        super().__init__()
        n_segment_now = n_segments
        segment_len_now = segment_len

        self.downs = nn.ModuleList([nn.Conv2d(1, inner_channel,
                                              kernel_size=3, padding=1)])

        # record the number of output channels

        num_mults = len(channel_mults)

        n_channel_in = inner_channel
        for ind in range(num_mults):

            n_channel_out = inner_channel * channel_mults[ind]

            for _ in range(0, res_blocks):
                self.downs.append(ResnetBlock(
                    n_channel_in, n_channel_out, norm_groups=norm_groups, dropout=dropout))
                n_channel_in = n_channel_out

            # doesn't change # channels
            self.downs.append(Downsample(n_channel_out))
            n_segment_now = n_segment_now//2
            segment_len_now = segment_len_now//2

        n_channel_out = n_channel_in
        self.mid = nn.ModuleList([
            ResnetBlock(n_channel_in, n_channel_out, norm_groups=norm_groups,
                        dropout=dropout),
        ])

        n_channel_in = n_channel_out

        self.final_block = SNRBlock(n_channel_in, n_segment_now, segment_len_now, n_segments, norm_groups)

    def forward(self, x):
        """
            x: [B, 1, N, L]
        """
        input = x
        for layer in self.downs:
            input = layer(input)

        for layer in self.mid:
            input = layer(input)

        output = self.final_block(input)
        return output

