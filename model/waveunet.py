import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import floor, log

# Positional Encoding and FiLM are borrowed from wavegrad
# subject to change

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, noise_level):
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
        N = x.shape[0]
        T = x.shape[2]

        return (x + self._build_encoding(noise_level)[:, :, None])

    def _build_encoding(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FiLM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.encoding = PositionalEncoding(input_size)
        self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
        self.output_conv = nn.Conv1d(input_size, output_size * 2, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_conv.weight)
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.input_conv.bias)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, noise_scale):
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.encoding(x, noise_scale)
        shift, scale = torch.chunk(self.output_conv(x), 2, dim=1)
        return shift, scale


def centre_crop(x, target):
    '''
    Center-crop 3-dim. input tensor along last axis so it fits the target tensor shape
    :param x: Input tensor
    :param target: Shape of this tensor will be used as target shape
    :return: Cropped input tensor
    '''
    if x is None:
        return None
    if target is None:
        return x

    target_shape = target.shape
    diff = x.shape[-1] - target_shape[-1]
    assert (diff % 2 == 0)
    crop = diff // 2

    if crop == 0:
        return x
    if crop < 0:
        raise ArithmeticError

    return x[:, :, crop:-crop].contiguous()

def build_sinc_filter(kernel_size, cutoff):
    # FOLLOWING https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
    # Sinc lowpass filter
    # Build sinc kernel
    assert(kernel_size % 2 == 1)
    M = kernel_size - 1
    filter = np.zeros(kernel_size, dtype=np.float32)
    for i in range(kernel_size):
        if i == M//2:
            filter[i] = 2 * np.pi * cutoff
        else:
            filter[i] = (np.sin(2 * np.pi * cutoff * (i - M//2)) / (i - M//2)) * \
                    (0.42 - 0.5 * np.cos((2 * np.pi * i) / M) + 0.08 * np.cos(4 * np.pi * M))

    filter = filter / np.sum(filter)
    return filter



class ResampleSinc(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding="reflect", transpose=False, trainable=False):
        '''
        Creates a resampling layer for time series data (using 1D convolution) - (N, C, W) input format
        :param channels: Number of features C at each time-step
        :param kernel_size: Width of sinc-based lowpass-filter (>= 15 recommended for good filtering performance)
        :param stride: Resampling factor (integer)
        :param transpose: False for down-, true for upsampling
        :param padding: Either "reflect" to pad or "valid" to not pad
        :param trainable: Optionally activate this to train the lowpass-filter, starting from the sinc initialisation
        '''
        super(ResampleSinc, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels

        cutoff = 0.5 / stride

        assert(kernel_size > 2)
        assert ((kernel_size - 1) % 2 == 0)
        assert(padding == "reflect" or padding == "valid")

        filter = build_sinc_filter(kernel_size, cutoff)

        self.filter = torch.nn.Parameter(torch.from_numpy(np.repeat(np.reshape(filter, [1, 1, kernel_size]), channels, axis=0)), requires_grad=trainable)

    def forward(self, x):
        # Pad here if not using transposed conv
        input_size = x.shape[2]
        if self.padding != "valid":
            num_pad = (self.kernel_size-1)//2
            out = F.pad(x, (num_pad, num_pad), mode=self.padding)
        else:
            out = x

        # Lowpass filter (+ 0 insertion if transposed)
        if self.transpose:
            expected_steps = ((input_size - 1) * self.stride + 1)
            if self.padding == "valid":
                expected_steps = expected_steps - self.kernel_size + 1

            out = F.conv_transpose1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)
            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert(diff_steps % 2 == 0)
                out = out[:,:,diff_steps//2:-diff_steps//2]
        else:
            assert(input_size % self.stride == 1)
            out = F.conv1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)

        return out

    def get_output_size(self, input_size):
        '''
        Returns the output dimensionality (number of timesteps) for a given input size
        :param input_size: Number of input time steps (Scalar, each feature is one-dimensional)
        :return: Output size (scalar)
        '''
        assert(input_size > 1)
        if self.transpose:
            if self.padding == "valid":
                return ((input_size - 1) * self.stride + 1) - self.kernel_size + 1
            else:
                return ((input_size - 1) * self.stride + 1)
        else:
            assert(input_size % self.stride == 1) # Want to take first and last sample
            if self.padding == "valid":
                return input_size - self.kernel_size + 1
            else:
                return input_size

    def get_input_size(self, output_size):
        '''
        Returns the input dimensionality (number of timesteps) for a given output size
        :param input_size: Number of input time steps (Scalar, each feature is one-dimensional)
        :return: Output size (scalar)
        '''

        # Strided conv/decimation
        if not self.transpose:
            curr_size = (output_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        if self.padding == "valid":
            curr_size = curr_size + self.kernel_size - 1 # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)# We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert(curr_size > 0)
        return curr_size


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



class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, depth, conv_type, res, resample_kernel_size=4, resample_stride=2):
        super(UpsamplingBlock, self).__init__()
        assert(resample_stride > 1)

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = ResampleSinc(n_inputs, 15, resample_stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, resample_kernel_size, resample_stride, conv_type, transpose=True)

        # following conv doesn't change size at last dim

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList(
                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1) ] +
                [ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)])

    def forward(self, x, film_shift, film_scale):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        #film_shift = centre_crop(film_shift, upsampled)
        #film_scale = centre_crop(film_scale, upsampled)

        # combine film features
        combined = upsampled
        # Combine  features
        for conv in self.post_shortcut_convs:
            combined = conv(film_scale*combined + film_shift)
        return combined

    def get_output_size(self, input_size):

        # Upsampling convs
        curr_size = self.upconv.get_output_size(input_size)

        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, depth, conv_type, res, resample_kernel_size=4, resample_stride=2):
        super(DownsamplingBlock, self).__init__()
        assert(resample_stride > 1)

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in
                                                  range(depth - 1)])

        # above conv doesn't change size at last dim

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = ResampleSinc(n_outputs, 15, resample_stride) # Resampling with fixed-size sinc lowpass filter
        else:
            padding=(resample_kernel_size-resample_stride)//2
            self.downconv = ConvLayer(n_outputs, n_outputs, resample_kernel_size, resample_stride, conv_type, padding=padding)

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_output_size(self, input_size):
        curr_size = self.downconv.get_output_size(input_size)

        return curr_size

class Waveunet(nn.Module):
    def __init__(self, num_inputs, num_channels,  kernel_size, input_size, conv_type, res, depth=1, resample_kernel_size= 4, resample_stride=2):
        super(Waveunet, self).__init__()
        self.num_levels = len(num_channels)
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.depth = depth
        self.resample_stride = resample_stride
        self.resample_kernel_size = resample_kernel_size

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)
        assert((resample_kernel_size- resample_stride)% 2 == 0)

        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        module = nn.Module()

        module.downsampling_blocks = nn.ModuleList()
        module.upsampling_blocks = nn.ModuleList()
        module.film_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            in_ch = num_inputs if i == 0 else num_channels[i]

            module.downsampling_blocks.append(
                DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], kernel_size, depth, conv_type, res, resample_kernel_size, resample_stride))
            module.film_blocks.append(FiLM(num_channels[i], num_channels[i]))

        for i in range(self.num_levels - 1, 0, -1):
            module.upsampling_blocks.append(
                UpsamplingBlock(num_channels[i], num_channels[i-1], num_channels[i-1], kernel_size, depth, conv_type, res, resample_kernel_size, resample_stride))

        module.bottlenecks = nn.ModuleList(
            [ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1, conv_type) for _ in range(depth)])

        # Output conv
        module.output_conv = nn.Conv1d(num_channels[0], 1, 1)

        self.waveunet = module

        #self.set_output_size(target_output_size)
        self.check_output_size(input_size)

    def check_output_size(self, input_size):
        module = self.waveunet
        curr_size = input_size
        for idx, block in enumerate(module.downsampling_blocks):
            print(curr_size)
            curr_size = block.get_output_size(curr_size)
        print('after down sample:')
        print(curr_size)

        # Bottleneck-Conv
        for block in module.bottlenecks:
            print(curr_size)
            curr_size = block.get_output_size(curr_size)
        print('after bottleneck:')
        print(curr_size)

        for idx, block in enumerate(reversed(module.upsampling_blocks)):
            print(curr_size)
            curr_size = block.get_output_size(curr_size)
        print('after up sample: ')
        print(curr_size)

        assert(curr_size == input_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size
        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunet
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                print(curr_size)
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size
            print('output:')
            print(output_size)

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                print(curr_size)
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                print(curr_size)
                curr_size = block.get_input_size(curr_size)
            print('input: ')
            print(curr_size)
            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward(self, x, y_t, noise_level):
        """
        A forward pass through Wave-U-Net
        :param x: Conditional input [B, 1, T]
        :param y_t: signal from last iteration [B, 1, T]
        :param noise_level: noise level
        """

        module = self.waveunet
        films = []
        out = torch.concat([x, y_t], dim=1)

        # DOWNSAMPLING BLOCKS

        for block, film in zip(module.downsampling_blocks, module.film_blocks):
            out, short = block(out)
            films.append(film(short, noise_level))

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for block, (film_shift, film_scale) in zip(module.upsampling_blocks, reversed(films)):
            out = block(out, film_shift, film_scale)

        # OUTPUT CONV
        out = module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)

        return out
