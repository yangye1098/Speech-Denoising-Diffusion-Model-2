import numpy as np
import torch
import torchaudio
from torchaudio import transforms as TT
from parse_config import ConfigParser

from glob import glob
from tqdm import tqdm
import argparse

#import librosa

def main(path, config):


    filenames = glob(f'{path}/**/*.wav', recursive=True)

    sample_rate = config['sample_rate']

    # multiprocess processing
    for i, filename in tqdm(enumerate(filenames), desc='Preprocessing', total=len(filenames)):
        audio, sr = torchaudio.load(filename)
        assert (sr == sample_rate)
        # audio should in range [0, 1]
        logwav = torch.log10(audio)

        if torch.max(logwav) > 0:
            print(f'mel min: {torch.min(logwav)}, max: {torch.max(logwav)}')

        logwav = torch.clamp((logwav + 5) / 5, 0.0, 1.0)
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

