
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

class Segmentor(nn.Module):

    """
    it is for torch tensor
    """

    def __init__(self, num_samples, F=512, stride=256):
        super().__init__()

        assert((num_samples-F) % stride == 0)
        self.num_samples = num_samples
        self.n_segments = (num_samples - F) // stride + 1
        self.idx_mat = torch.empty((self.n_segments, F), dtype=torch.long)
        start = 0
        self.weight_mat = torch.ones((self.n_segments, F))
        self.F = F
        half_frame = F//2
        down_ramp = torch.linspace(1., 0., steps=half_frame)
        up_ramp = torch.linspace(0., 1., steps=half_frame)
        for i in range(self.n_segments):
            if i == 0 :
                self.weight_mat[i, half_frame:] = down_ramp
            elif i == self.n_segments:
                self.weight_mat[i, 0:half_frame] = up_ramp
            else:
                self.weight_mat[i, 0:half_frame] = up_ramp
                self.weight_mat[i, half_frame:] = down_ramp

            self.idx_mat[i, :] = torch.arange(start, start+F)
            start += stride


    def forward(self, sig):
        """
            sig: [B, 1, n_samples]
            return: [B, 1, nframes, F]
        """
        return self.weight_mat * sig[:, :, self.idx_mat]

    def overlapAdd(self, input):
        """
            reverse the segementation process
            input [B, 1, n_segments, F]
            return [B, 1, n_samples]
        """

        output = torch.zeros((input.shape[0], input.shape[1], self.num_samples), device=input.device)
        for i in range(self.n_segments):
            output[:, :, self.idx_mat[i, :]] += input[:, :, i, :]

        return output
