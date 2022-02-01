from base import BaseDataLoader

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path


def generate_inventory(path, file_type='.wav'):
    path = Path(path)
    assert path.is_dir(), '{:s} is not a valid directory'.format(path)

    file_paths = path.glob('*'+file_type)
    file_names = [ file_path.name for file_path in file_paths ]
    assert file_names, '{:s} has no valid {} file'.format(path, file_type)
    return file_names


class AudioDataset(Dataset):
    def __init__(self, data_root, datatype, sample_rate=8000, T=-1):
        if datatype not in ['.wav', '.spec.npy', '.mel.npy']:
            raise NotImplementedError
        self.datatype = datatype
        self.sample_rate = sample_rate
        self.T = T

        self.clean_path = Path('{}/clean'.format(data_root))
        self.noisy_path = Path('{}/noisy'.format(data_root))

        self.inventory = generate_inventory(self.clean_path, datatype)
        self.data_len = len(self.inventory)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        if self.datatype == '.wav':
            clean, sr = torchaudio.load(self.clean_path/self.inventory[index], num_frames=self.T)
            assert(sr==self.sample_rate)
            noisy, sr = torchaudio.load(self.noisy_path/self.inventory[index], num_frames=self.T)
            assert (sr == self.sample_rate)
        elif self.datatype == '.spec.npy' or self.datatype == '.mel.npy':
            # load the two grams
            clean = torch.from_numpy(np.load(self.clean_path/self.inventory[index]))
            noisy = torch.from_numpy(np.load(self.noisy_path/self.inventory[index]))

        return clean, noisy, index

    def getName(self, idx):
        return self.inventory[idx]



class AudioDataLoader(BaseDataLoader):
    """
    Load Audio data
    """
    def __init__(self, dataset,  batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset =dataset
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':

    try:
        import simpleaudio as sa
        hasAudio = True
    except ModuleNotFoundError:
        hasAudio = False



