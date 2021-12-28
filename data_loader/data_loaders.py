from base import BaseDataLoader

import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from random import shuffle

try:
    import simpleaudio as sa
    hasAudio = True
except ModuleNotFoundError:
    hasAudio = False


def generate_inventory(path, sound_type='.wav'):
    path = Path(path)
    assert path.is_dir(), '{:s} is not a valid directory'.format(path)

    snd_paths = path.glob('*'+sound_type)
    snd_names = [ snd_path.name for snd_path in snd_paths ]
    assert snd_names, '{:s} has no valid sound file'.format(path)
    shuffle(snd_names)
    return snd_names


class AudioDataset(Dataset):
    def __init__(self, data_root, snr, T=-1, sample_rate=8000, sound_type='.wav'):
        self.snr = snr
        self.T = T
        self.sample_rate = sample_rate

        self.clean_path = Path('{}/clean'.format(data_root))
        self.noisy_path = Path('{}/noisy_{}'.format(data_root, snr))
        self.inventory = generate_inventory( self.clean_path, sound_type=sound_type)
        self.data_len = len(self.inventory)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        clean_snd, sr = torchaudio.load(self.clean_path/self.inventory[index], num_frames=self.T)
        assert(sr==self.sample_rate)
        noisy_snd, sr = torchaudio.load(self.noisy_path/self.inventory[index], num_frames=self.T)
        assert (sr == self.sample_rate)

        return clean_snd, noisy_snd

    def get_name(self, index):
        return self.inventory[index]

    def playIdx(self, idx):
        if hasAudio:
            clean, noisy = self.__getitem__(idx)
            play_obj = sa.play_buffer(clean.numpy(), 1, 32//8, self.sample_rate)
            play_obj.wait_done()
            play_obj = sa.play_buffer(noisy.numpy(), 1, 32//8, self.sample_rate)
            play_obj.wait_done()



class AudioDataLoader(BaseDataLoader):
    """
    Load Audio data
    """
    def __init__(self, dataset,  batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset =dataset
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':
    snr = 0
    dataroot = f'../data/wsj0_si_tr_{snr}'
    dataset_tr = AudioDataset(dataroot, snr)
    dataset_tr.playIdx(0)


