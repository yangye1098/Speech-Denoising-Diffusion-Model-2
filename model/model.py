import torch
from torch import nn
from base import BaseModel
from .diffusion import GaussianDiffusion
from tqdm import tqdm

class SDDM(BaseModel):
    def __init__(self, diffusion:GaussianDiffusion, denoise_network:nn.Module):
        super().__init__()
        self.diffusion = diffusion
        self.denoise_network = denoise_network
        self.num_timesteps = self.diffusion.num_timesteps

    # train step
    def forward(self, clean, x):
        """
        clean is the clean sourse
        x is the noisy conditional input
        """

        # generate noise

        noise = torch.randn_like(clean, device=clean.device)
        y_t, noise_level = self.diffusion.p_stochastic(clean, noise)
        predicted = self.denoise_network(x, y_t, noise_level)
        return predicted, noise

    @torch.no_grad()
    def infer(self, x, continuous=False):

        # initial input
        y_t = torch.randn_like(x, device=x.device)
        # TODO: predict noise level to reduce computation cost

        num_timesteps = self.diffusion.num_timesteps
        sample_inter = (1 | (num_timesteps // 100))

        batch_size = x.shape[0]
        # iterative refinement
        if continuous:
            assert batch_size==1, 'Batch size must be 1 to do continuous sampling'
            samples = [x]
            for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                noise_level = self.diffusion.get_noise_level(t)* torch.ones(batch_size)
                predicted = self.denoise_network(x, y_t, noise_level)
                y_t = self.diffusion.q_transition(y_t, t, predicted)
                if t % sample_inter == 0:
                    samples.append(y_t)

            return samples

        else:
            for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                noise_level = self.diffusion.get_noise_level(t)* torch.ones(batch_size)
                predicted = self.denoise_network(x, y_t, noise_level)
                y_t = self.diffusion.q_transition(y_t, t, predicted)

            return y_t


