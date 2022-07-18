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
    def __init__(self, noise_level_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        n_expand_channels = noise_level_channels * 4

        self.noise_func = nn.Sequential(
            PositionalEncoding(noise_level_channels),
            nn.Linear(noise_level_channels, n_expand_channels),
            nn.PReLU(n_expand_channels),
            nn.Linear(n_expand_channels, out_channels*(1+self.use_affine_level))
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

class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward)
        self.gru = nn.GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model)
        if bidirectional:
            self.linear2 = nn.Linear(d_model * 2 * 2, d_model)
        else:
            self.linear2 = nn.Linear(d_model * 2, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.functional.relu
        elif activation == "gelu":
            self.activation = nn.functional.gelu
        else:
            raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.functional.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class Dual_Transformer(nn.Module):

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super(Dual_Transformer, self).__init__()


        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU(input_size//2)
        )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=True))
            self.col_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=True))
            self.row_norm.append(nn.GroupNorm(1, input_size//2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size//2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(
                                    nn.Conv2d(input_size//2, output_size, 1),
                                    nn.PReLU(output_size)
                                    )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape
        output = self.input(input)

        for i in range(len(self.row_trans)):
            row_input = output.permute(3, 0, 2, 1).contiguous().view(dim1, b*dim2, -1)  # [dim1, b*dim2, c]
            row_output = self.row_trans[i](row_input)  # [dim1, b*dim2, c]
            row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [b, c, dim2, dim1]
            row_output = self.row_norm[i](row_output)  # [b, c, dim2, dim1]
            output = output + row_output  # [b, c, dim2, dim1]

            col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b*dim1, -1)  # [dim2, b*dim1, c]
            col_output = self.col_trans[i](col_input)  # [dim2, b*dim1, c]
            col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()  # [b, c, dim2, dim1]
            col_output = self.col_norm[i](col_output)  # [b, c, dim2, dim1]
            output = output + col_output  # [b, c, dim2, dim1]

        del row_input, row_output, col_input, col_output
        output = self.output(output)  # [b, c, dim2, dim1]

        return output


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels*r, kernel_size=kernel_size, stride=(1, 1), padding=(0, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)

        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class EncodeLayer(nn.Module):
    def __init__(self,
                 n_in_channels,
                 frame_length,
                 n_out_channels,
                 noise_level_channels,
                 depth = 5,
                 use_affine_level = False
                 ):
        super().__init__()
        self.dense = DenseBlock(frame_length, depth, n_in_channels)
        self.noise_func = FeatureWiseAffine(noise_level_channels, n_in_channels, use_affine_level)
        # down sample
        self.downsample = nn.Sequential(
            nn.Conv2d(n_in_channels, n_out_channels, (1, 3), (1, 2), (0, 1)),
            nn.LayerNorm(frame_length//2),
            nn.PReLU(n_out_channels)
        )


    def forward(self, x, noise_level):
        # embed noise_level
        x = self.noise_func(x, noise_level)
        x = self.dense(x)

        return self.downsample(x)


class DecodeLayer(nn.Module):
    def __init__(self,
                 n_in_channels,
                 frame_length,
                 n_out_channels,
                 noise_level_channels,
                 depth = 5,
                 use_affine_level = False
                 ):
        super().__init__()
        self.dense = DenseBlock(frame_length, depth, n_in_channels)
        self.noise_func = FeatureWiseAffine(noise_level_channels, n_in_channels, use_affine_level)
        # up sample

        self.upsample = nn.Sequential(
            SPConvTranspose2d(n_in_channels*2, n_out_channels, (1,3), r=2 ),
            nn.LayerNorm(frame_length*2),
            nn.PReLU(n_out_channels)
        )

    def forward(self, x, skip, noise_level):
        x = self.noise_func(x, noise_level)
        x = self.dense(x)
        x = torch.cat([x, skip], dim=1)
        return self.upsample(x)


class CAUNet(nn.Module):
    def __init__(
        self,
        num_samples,
        inner_channel=64,
        n_encode_layers=4,
        dense_depth=3,
        n_TSTB=6,
        segment_len=128,
        segment_stride=64,
        use_affine_level=False
    ):
        super().__init__()


        self.segment = SignalToFrames(num_samples, segment_len, segment_stride)
        # first conv raise # channels to inner_channel

        self.first_conv = nn.Conv2d(2, inner_channel, kernel_size=(1,1), stride=(1,1))
        noise_level_channels = inner_channel
        self.downs = nn.ModuleList([])
        current_segment_len = segment_len
        for ind in range(n_encode_layers):
            self.downs.append(EncodeLayer(inner_channel, current_segment_len, inner_channel,
                                                     noise_level_channels, dense_depth, use_affine_level))
            current_segment_len = current_segment_len//2

        self.mid = Dual_Transformer(inner_channel, inner_channel, 0, n_TSTB)
        self.ups = nn.ModuleList([])

        for ind in range(n_encode_layers):
            self.ups.append(DecodeLayer(inner_channel, current_segment_len, inner_channel,
                                                     noise_level_channels, dense_depth, use_affine_level))
            current_segment_len = current_segment_len * 2

        self.final_conv = nn.Conv2d(inner_channel, 1, kernel_size=(1,1))

    def forward(self, x, y_t, noise_level):
        """
            x: [B, 1, T]
            y_t: [B, 1, T]
            time: [B, 1, 1]
        """
        # expand to 4d
        noise_level = noise_level.squeeze() #[B]
        x = self.segment(x)
        y_t = self.segment(y_t)

        feats = []
        input = torch.cat([x, y_t], dim=1)
        input = self.first_conv(input)
        for layer in self.downs:
            input = layer(input, noise_level)
            feats.append(input)

        input = self.mid(input)

        for layer in self.ups:
            input = layer(input, feats.pop(), noise_level)

        output = self.final_conv(input)
        output = self.segment.overlapAdd(output)

        return output

