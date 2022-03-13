import numpy as np
import torch
import torchaudio
from torchaudio import transforms as TT
from parse_config import ConfigParser

from glob import glob
from tqdm import tqdm
import argparse
import librosa

def main(path, config):
    window_length = config['spectrogram']['window_length']
    hop_samples = config['spectrogram']['hop_samples']
    n_mels = config['mel_spectrogram']['n_mels']
    filenames = glob(f'{path}/**/*.wav', recursive=True)

    sample_rate = config['sample_rate']
    spectrogram = TT.Spectrogram(n_fft=window_length,
                                 hop_length=hop_samples,
                                 window_fn=torch.hamming_window,
                                 power=1,
                                 normalized=True,
                                 )

    mel_spec = TT.MelSpectrogram(n_fft=window_length,
                                 hop_length=hop_samples,
                                 f_min = 20.0,
                                 f_max = sample_rate/2.0,
                                 n_mels=n_mels,
                                 sample_rate=sample_rate,
                                 power=1.0,
                                 normalized=True
                                 )

    # multiprocess processing
    for i, filename in tqdm(enumerate(filenames), desc='Preprocessing', total=len(filenames)):
        audio, sr = torchaudio.load(filename)
        assert (sr == sample_rate)
        mel = mel_spec(audio)
        # keep value in range [1e-4, 10]
        mel = torch.log10(mel) - 1
        if torch.max(mel) > 0:
            print(f'mel min: {torch.min(mel)}, max: {torch.max(mel)}')
        mel = torch.clamp((mel + 5) / 5, 0.0, 1.0)
        np.save(f'{filename}.mel.npy', torch.squeeze(mel).cpu().numpy())

        spec = spectrogram(audio)
        # keep value in range [1e-4, 10]
        spec = torch.log10(spec) - 1
        if torch.max(spec) > 0:
            print(f'spec min: {torch.min(spec)}, max: {torch.max(spec)}')
        spec = torch.clamp((spec + 5) / 5, 0.0, 1.0)
        np.save(f'{filename}.spec.npy', torch.squeeze(spec).cpu().numpy())


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Speech denoising diffusion model')
    args.add_argument('path', type=str,
                      help='data path')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    args = args.parse_args()
    args.resume = None
    args.device = None

    config = ConfigParser.from_args(args)
    main(args.path, config)

