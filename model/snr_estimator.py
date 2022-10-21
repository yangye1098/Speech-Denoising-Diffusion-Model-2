
import torch
from torch import nn


def segment_sisnr(s_hat, s):
    """
    Calculate segment SISNR
    Args:
        s_hat: [B, n_segment, segment_length]
        s: [B, n_segment, segment_length], the true sources
    Returns:
        SI-SNR: [B, n_segment, segment_length]

    """

    # normalize to zero mean
    s_hat = s_hat - torch.mean(s_hat, dim=-1, keepdim=True)  # [B, n, L]
    s = s - torch.mean(s, dim=-1, keepdim=True)  # [B, n, L]
    # <s, s_hat>s/||s||^2
    s_shat = torch.sum(s_hat * s, dim=-1, keepdim=True)  # [B, n, L]
    s_2 = torch.sum(s ** 2, dim=-1, keepdim=True)  # [B, n, L]
    s_target = s_shat * s / s_2  # [B, n, L]

    # e_noise = s_hat - s_target
    e_noise = s_hat - s_target  # [B, n, L]
    sisnr = 10 * torch.log10(torch.sum(s_target ** 2, dim=-1, keepdim=True) \
                             / torch.sum(e_noise ** 2, dim=-1, keepdim=True)) # [B, n, L]

    return sisnr.squeeze() #[B, n]

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


class EstimateNet(nn.Module):
    def __init__(
            self,
            n_segment,
            segment_len,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 3, 4, 5),
            res_blocks=3,
            dropout=0,
    ):
        super().__init__()
        n_segment_now = n_segment
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

        self.final_block = SNRBlock(n_channel_in, n_segment_now, segment_len_now, n_segment, norm_groups)

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


class SNR_Estimator(nn.Module):
    def __init__(
            self,
            num_samples,
            segment_len,
            segment_stride,
            inner_channel,
            norm_groups,
            channel_mults,
            res_blocks,
            dropout,
    ):
        super().__init__()
        self.segmentor = SignalToFrames(num_samples, segment_len, segment_stride)
        self.net = EstimateNet(self.segmentor.n_frames, segment_len, inner_channel, norm_groups, channel_mults, res_blocks, dropout)


    def forward(self, clean, noisy):
        """
            clean: [B, 1, T]
            noisy: [B, 1, T]
        """
        clean = self.segmentor(clean)
        noisy = self.segmentor(noisy)
        sisnr = segment_sisnr(noisy, clean)
        output = self.net(noisy)
        return output, sisnr
