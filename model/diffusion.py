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
        self.schedule = schedule
        self.num_timesteps = n_timestep
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.device=device

        # set noise schedule
        betas = make_beta_schedule(
            schedule=schedule,
            n_timestep=n_timestep,
            linear_start=linear_start,
            linear_end=linear_end)

        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        # sqrt_alphas_comprode_prev is equivalent to noise_level
        sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))


        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)


        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('sqrt_alphas_cumprod_prev',
                             to_torch(sqrt_alphas_cumprod_prev))
        # for infer
        # 1/sqrt(alpha[t]
        self.register_buffer('one_over_sqrt_one_minus_alpha', to_torch(1/np.sqrt(alphas)))
        # (1 - alpha[t]) / sqrt(1 - gamma[t])
        self.register_buffer('one_minus_alpha_over_sqrt_one_minus_gamma',
                             to_torch((1-alphas)/np.sqrt((1-alphas_cumprod))))




    def q_transition_original(self):
        """
        q_transition from Ho et al 2020
        """
        pass


    @torch.no_grad()
    def q_transition_sr3(self, y_t, t, predicted):
        """
        sr3 q_transition
        noise scalar is different
        """
        c1 = self.one_over_sqrt_one_minus_alpha[t]
        c2 = self.one_minus_alpha_over_sqrt_one_minus_gamma[t]
        if t > 1:
            noise = torch.randn_like(y_t)
            y_t_1 = c1 * (y_t - c2 * predicted)  + self.betas[t] * noise
        else:
            # noise = torch.zeros_like(y_t)
            # y_t_1 = c1 * (y_t - c2 * predicted)  + sigma * noise
            y_t_1 = c1 * (y_t - c2 * predicted)

        y_t_1.clamp_(-1., 1.)
        return y_t_1

    @torch.no_grad()
    def q_transition(self, y_t, t, predicted):
        """
        wavegrad q_transition, conditioned on t, t is scalar
        """

        c1 = self.one_over_sqrt_one_minus_alpha[t]
        c2 = self.one_minus_alpha_over_sqrt_one_minus_gamma[t]
        if t > 0:
            noise = torch.randn_like(y_t)
            sigma = ((1.0 - self.alphas_cumprod[t-1]) / (1.0 - self.alphas_cumprod[t]) * self.betas[t])**0.5
            y_t_1 = c1 * (y_t - c2 * predicted)  + sigma * noise
        else:
            # noise = torch.zeros_like(y_t)
            # y_t_1 = c1 * (y_t - c2 * predicted)  + sigma * noise
            y_t_1 = c1 * (y_t - c2 * predicted)

        y_t_1.clamp_(-1., 1.)
        return y_t_1

    def p_stochastic(self, y_0, noise):
        """
        y_0 has shape of [B, 1, T]
        choose a random diffusion step to calculate loss
        """
        # 0 dim is the batch size
        b = y_0.size(0)
        noise_level_sample_shape = np.ones(y_0.ndim, dtype='int')
        noise_level_sample_shape[0] = b

        # choose random step for each one in this batch
        t = np.random.randint(1, self.num_timesteps + 1, size=b)
        # sample noise level using uniform distribution
        l_a, l_b = self.sqrt_alphas_cumprod_prev[t - 1], self.sqrt_alphas_cumprod_prev[t]
        noise_level_sample = l_a + torch.rand(b, device=y_0.device) * (l_b - l_a)
        noise_level_sample = noise_level_sample.view(tuple(noise_level_sample_shape))

        y_t = noise_level_sample * y_0 + torch.sqrt((1. - torch.square(noise_level_sample))) * noise

        return y_t, noise_level_sample

    def get_noise_level(self, t):
        """
        noise level is sqrt alphas comprod
        """
        return self.sqrt_alphas_cumprod_prev[t]


if __name__ == '__main__':
    diffussion = GaussianDiffusion(device='cpu')
    y_0 = torch.ones([2,1, 3])
    noise = torch.randn_like(y_0)
    diffussion.p_stochastic(y_0, noise)

    predicted = noise
    y_t = y_0
    diffussion.q_transition(y_t, 0, predicted)
    diffussion.q_transition(y_t, 1, predicted)
