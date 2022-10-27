import math
import torch
from torch import nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        half_dim = self.dim //2
        step = torch.arange(half_dim)
        # TODO: check embedding vector
        self.embedding_vector = 10.0 ** (step * 4.0/half_dim)
        self.embedding_vector = self.embedding_vector.view(-1, 1, 1)


    def forward(self, diffusion_step):
        # diffusion_step [B, 1, N, 1]
        x = self._build_embedding(diffusion_step)
        return x

    def _build_embedding(self, diffusion_step):
        self.embedding_vector = self.embedding_vector.to(diffusion_step.device)
        encoding = diffusion_step * self.embedding_vector
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=1)  # [B, self.dim]
        return encoding


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Upsample_NoiseLevel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode="nearest"),
            nn.Conv2d(dim, dim, (3, 1), (1,1), padding=(1, 0)),
            Swish()
        )


    def forward(self, x):
        return self.up(x)


class Downsample(nn.Module):
    def __init__(self, dim ):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample_NoiseLevel(nn.Module):
    def __init__(self, noise_emb_dim):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(noise_emb_dim, noise_emb_dim, (3, 1), (2, 1), (1, 0)),
            Swish()
        )


    def forward(self, x):
        return self.down(x)


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
    def __init__(self, dim, dim_out, noise_level_emb_dim, dropout=0, norm_groups=32):
        super().__init__()
        self.noise_func = nn.Conv2d(noise_level_emb_dim, dim_out, kernel_size=1, stride=1)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        #x [B, dim, N, L]
        # time_emb [B, noise_level_emb_dim, N, 1]

        h = self.block1(x) # [B, dim_out, N, L]
        h = h + self.noise_func(time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)



class UNetModified2_VariableNoiseLevel(nn.Module):
    def __init__(
        self,
        in_channel=2,
        out_channel=1,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 3, 4, 5),
        res_blocks=3,
        dropout=0,
    ):
        super().__init__()


        # first conv raise # channels to inner_channel


        noise_level_channel = 128

        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(noise_level_channel),
            nn.Conv2d(noise_level_channel, noise_level_channel * 4, kernel_size=1, stride=1),
            Swish(),
            nn.Conv2d(noise_level_channel * 4, noise_level_channel, kernel_size=1, stride=1),
            Swish()
        )

        self.first_conv = nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])

        self.noise_level_down = nn.ModuleList([])
        self.noise_level_up = nn.ModuleList([])

        # record the number of output channels
        feat_channels = [inner_channel]

        num_mults = len(channel_mults)

        n_channel_in = inner_channel
        for ind in range(num_mults):

            n_channel_out = inner_channel * channel_mults[ind]

            for _ in range(0, res_blocks):
                self.downs.append(ResnetBlock(
                    n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(n_channel_out)
                n_channel_in = n_channel_out

            # doesn't change # channels
            self.downs.append(Downsample(n_channel_out))
            self.noise_level_down.append(Downsample_NoiseLevel(noise_level_channel))
            feat_channels.append(n_channel_out)

        n_channel_out = n_channel_in
        self.mid = nn.ModuleList([
                ResnetBlock(n_channel_in, n_channel_out, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                   dropout=dropout),
        ])
        self.ups = nn.ModuleList([])


        for ind in reversed(range(num_mults)):

            n_channel_in = inner_channel * channel_mults[ind]
            n_channel_out = n_channel_in

                # combine down sample layer skip connection
            self.ups.append(ResnetBlock(
                    n_channel_in + feat_channels.pop(), n_channel_out, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout))

            # up sample
            self.ups.append(Upsample(n_channel_out))
            self.noise_level_up.append(Upsample_NoiseLevel(noise_level_channel))

            if ind == 0:
                n_channel_out = inner_channel
            else:
                n_channel_out = inner_channel * channel_mults[ind-1]

            # combine resnet block skip connection
            for _ in range(0, res_blocks):
                self.ups.append(ResnetBlock(
                    n_channel_in+feat_channels.pop(), n_channel_out, noise_level_emb_dim=noise_level_channel , norm_groups=norm_groups,
                    dropout=dropout))
                n_channel_in = n_channel_out

        n_channel_in = n_channel_out
        self.final_conv = Block(n_channel_in, out_channel, groups=norm_groups)

    def forward(self, x, y_t, noise_level):
        """
            x: [B, 1, N, L]
            y_t: [B, 1, N, L]
            noise_level: [B, 1, N, 1]
        """
        # expand to 4d

        input = torch.cat([x, y_t], dim=1)
        t = self.noise_level_mlp(noise_level)

        input = self.first_conv(input)
        feats = [input]
        n_downsample = 0
        for layer in self.downs:
            if isinstance(layer, ResnetBlock):
                input = layer(input, t)
            else:
                input = layer(input)
                t = self.noise_level_down[n_downsample](t)
                print(input.shape)
                print(t.shape)
                n_downsample = n_downsample+1
            feats.append(input)

        for layer in self.mid:
            input = layer(input, t)
        n_upsample = 0
        for layer in self.ups:
            if isinstance(layer, ResnetBlock):
                input = layer(torch.cat((input, feats.pop()), dim=1), t)
            else:
                input = layer(input)
                t = self.noise_level_up[n_upsample](t)
                n_upsample = n_upsample+1

        output = self.final_conv(input)
        return output
