import math
import torch
from torch import nn
import numpy as np
from functools import partial


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion class
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        schedule='linear',
        n_timestep=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        device='cuda'
    ):
        super().__init__()
        self.num_timesteps = n_timestep
        self.device=device


        # set noise schedule
        # all variables have length of num_timesteps + 1
        betas = torch.zeros(n_timestep + 1, dtype=torch.float32, device=device)
        if schedule == 'linear':
            betas[1:] = torch.linspace(linear_start, linear_end, n_timestep, device=device, dtype=torch.float32)
        else:
            raise NotImplementedError

        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, axis=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', sqrt_alpha_bar)
        # standard deviation

        self.calculate_p_coeffs()

    def calculate_p_coeffs(self):

        # for infer
        sigma = torch.zeros_like(self.betas)
        sigma[1:] = ((1.0 - self.alpha_bar[:-1]) / (1.0 - self.alpha_bar[1:]) * self.betas[1:]) ** 0.5
        predicted_noise_coeff = torch.zeros_like(self.betas)
        predicted_noise_coeff[1:] = self.betas[1:]/ torch.sqrt(1-self.alpha_bar[1:])

        self.register_buffer('predicted_noise_coeff', predicted_noise_coeff)
        self.register_buffer('sigma', sigma)


    @torch.no_grad()
    def p_transition_sr3(self, y_t, t, predicted):
        """
        sr3 p_transition
        noise variance is different from Ho et al 2020
        """
        y_t_1 = (y_t - self.predicted_noise_coeff[t] * predicted)/self.alpha_bar[t]
        if t > 1:
            noise = torch.randn_like(y_t)
            y_t_1 += torch.sqrt(self.betas[t]) * noise

        y_t_1.clamp_(-1., 1.)
        return y_t_1

    @torch.no_grad()
    def p_transition(self, y_t, t, predicted):
        """
        p_transition from Ho et al 2020, and wavegrad, conditioned on t, t is scalar
        """

        # mean
        y_t_1 = (y_t - self.predicted_noise_coeff[t] * predicted)/self.alpha_bar[t]
        # add gaussian noise with std of sigma
        if t > 1:
            noise = torch.randn_like(y_t)
            y_t_1 += self.sigma[t] * noise

        y_t_1.clamp_(-1., 1.)
        return y_t_1

    def q_stochastic(self, y_0, noise):
        """
        y_0 has shape of [B, 1, T]
        choose a random diffusion step to calculate loss
        """
        # 0 dim is the batch size
        b = y_0.shape[0]
        alpha_bar_sample_shape = torch.ones(y_0.ndim, dtype=torch.int)
        alpha_bar_sample_shape[0] = b

        # choose random step for each one in this batch
        # change to torch
        t = torch.randint(1, self.num_timesteps + 1, [b], device=y_0.device)
        # sample noise level using uniform distribution
        l_a, l_b = self.sqrt_alpha_bar[t - 1], self.sqrt_alpha_bar[t]
        alpha_bar_sample = l_a + torch.rand(b, device=y_0.device) * (l_b - l_a)
        alpha_bar_sample = alpha_bar_sample.view(tuple(alpha_bar_sample_shape))

        y_t = alpha_bar_sample * y_0 + torch.sqrt((1. - torch.square(alpha_bar_sample))) * noise

        return y_t, alpha_bar_sample

    def get_noise_level(self, t):
        """
        noise level is sqrt alphas comprod
        """
        return self.sqrt_alphas_cumprod_prev[t]


if __name__ == '__main__':
    diffussion = GaussianDiffusion(device='cpu')
    y_0 = torch.ones([2,1, 3])
    noise = torch.randn_like(y_0)
    diffussion.q_stochastic(y_0, noise)

    predicted = noise
    y_t = y_0
    diffussion.p_transition(y_t, 0, predicted)
    diffussion.p_transition(y_t, 1, predicted)
