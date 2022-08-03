import numpy as np
import torch
import torchaudio
from torchaudio import transforms as TT
from parse_config import ConfigParser
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import argparse

#import librosa

def log_modulus_normalize(audio, expand_order):
    # -1 < audio < 1
    # log_modulus = sign(x) * log10(|x * 10**expand_order| + 1)
    # -expand_order < log_modulus < expand_order
    # normalize =  log_modulus  / (2*order)
    # return -1 < audio_log_modulus < 1
    audio_log_modulus = torch.sign(audio)*torch.log10(torch.abs(10.**expand_order * audio) + 1.)
    audio_log_modulus = audio_log_modulus / (2*expand_order)
    return audio_log_modulus

def log_modulus_normalize_reverse(audio_log_modulus, expand_order):
    # reverse normalization
    audio_log_modulus = audio_log_modulus * 2 * expand_order
    sign = torch.sign(audio_log_modulus)
    return sign * (torch.pow(10, torch.abs(audio_log_modulus)) - 1.) / 10.**expand_order


def main(path, config):


    filenames = glob(f'{path}/**/*.wav', recursive=True)

    sample_rate = config['sample_rate']
    expand_order = 3
    # multiprocess processing
    for i, filename in tqdm(enumerate(filenames), desc='Preprocessing', total=len(filenames)):
        audio, sr = torchaudio.load(filename)
        assert (sr == sample_rate)
        # audio is in range (-1, 1)

        logwav = log_modulus_normalize(audio, expand_order)

        if torch.max(logwav) > 1:
            print(f'min: {torch.min(logwav)}, max: {torch.max(logwav)}')
        if torch.min(logwav) < -1:
            print(f'min: {torch.min(logwav)}, max: {torch.max(logwav)}')

        np.save(f'{filename}.logwav.npy', logwav.cpu().numpy())



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

