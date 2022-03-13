import torch
from torch import nn
from base import BaseModel
from .diffusion import GaussianDiffusion
from tqdm import tqdm

class SDDM(BaseModel):
    def __init__(self, diffusion:GaussianDiffusion, noise_estimate_model:nn.Module, noise_condition='alpha_bar'):
        super().__init__()
        self.diffusion = diffusion
        self.noise_estimate_model = noise_estimate_model
        self.num_timesteps = self.diffusion.num_timesteps
        self.noise_condition = noise_condition
        if noise_condition != 'alpha_bar' and noise_condition != 'time_step':
            raise NotImplementedError

    # train step
    def forward(self, target, condition):
        """
        target is the target sourse
        condition is the noisy conditional input
        """

        # generate noise
        noise = torch.randn_like(target, device=target.device)
        y_t, noise_level, t = self.diffusion.q_stochastic(target, noise)
        if self.noise_condition == 'alpha_bar':
            predicted = self.noise_estimate_model(condition, y_t, noise_level)
        elif self.noise_condition == 'time_step':
            predicted = self.noise_estimate_model(condition, y_t, t)

        return predicted, noise

    @torch.no_grad()
    def infer(self, condition, continuous=False):
        # condition is audio
        # initial input
        y_t = torch.randn_like(condition, device=condition.device)
        # TODO: predict noise level to reduce computation cost


        num_timesteps = self.diffusion.num_timesteps
        sample_inter = (1 | (num_timesteps // 100))

        batch_size = condition.shape[0]
        b = condition.shape[0]
        noise_level_sample_shape = torch.ones(condition.ndim, dtype=torch.int)
        noise_level_sample_shape[0] = b
        # iterative refinement
        if continuous:
            assert batch_size==1, 'Batch size must be 1 to do continuous sampling'
            samples = [condition]
            for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                if self.noise_condition == 'alpha_bar':
                    noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                                                                                 device=condition.device)
                    predicted = self.noise_estimate_model(condition, y_t, noise_level)
                elif self.noise_condition == 'time_step':
                    time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=condition.device)
                    predicted = self.noise_estimate_model(condition, y_t, time_steps)

                y_t = self.diffusion.p_transition(y_t, t, predicted)
                if t % sample_inter == 0:
                    samples.append(y_t)

            return samples

        else:
            for t in reversed(range(0, self.num_timesteps)):
                if self.noise_condition == 'alpha_bar':
                    noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                                                                                 device=condition.device)
                    predicted = self.noise_estimate_model(condition, y_t, noise_level)
                elif self.noise_condition == 'time_step':
                    time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=condition.device)
                    predicted = self.noise_estimate_model(condition, y_t, time_steps)

                y_t = self.diffusion.p_transition(y_t, t, predicted)

            return y_t


class SDDM_spectrogram(SDDM):

    def __init__(self, diffusion:GaussianDiffusion, noise_estimate_model:nn.Module, hop_samples:int, noise_condition='alpha_bar'):
        super().__init__(diffusion, noise_estimate_model, noise_condition)
        self.hop_samples = hop_samples

    @torch.no_grad()
    def infer(self, condition, continuous=False):
        # condition is spectrogram
        # initial input
        y_t = torch.randn(condition.shape[0], 1, self.hop_samples * condition.shape[-1], device=condition.device)
        # TODO: predict noise level to reduce computation cost

        num_timesteps = self.diffusion.num_timesteps
        sample_inter = (1 | (num_timesteps // 100))

        batch_size = condition.shape[0]
        b = condition.shape[0]
        noise_level_sample_shape = torch.ones(condition.ndim, dtype=torch.int)
        noise_level_sample_shape[0] = b

        # iterative refinement
        if continuous:
            assert batch_size==1, 'Batch size must be 1 to do continuous sampling'
            samples = [condition]
            for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                if self.noise_condition == 'alpha_bar':
                    noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                                                                                 device=condition.device)
                    predicted = self.noise_estimate_model(condition, y_t, noise_level)
                elif self.noise_condition == 'time_step':
                    time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=condition.device)
                    predicted = self.noise_estimate_model(condition, y_t, time_steps)
                y_t = self.diffusion.q_transition(y_t, t, predicted)
                if t % sample_inter == 0:
                    samples.append(y_t)

            return samples

        else:
            for t in reversed(range(0, self.num_timesteps)):
                if self.noise_condition == 'alpha_bar':
                    noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                                                                                 device=condition.device)
                    predicted = self.noise_estimate_model(condition, y_t, noise_level)
                elif self.noise_condition == 'time_step':
                    time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=condition.device)
                    predicted = self.noise_estimate_model(condition, y_t, time_steps)

                y_t = self.diffusion.q_transition(y_t, t, predicted)

            return y_t
