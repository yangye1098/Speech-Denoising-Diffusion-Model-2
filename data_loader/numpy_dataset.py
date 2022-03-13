# Copied from wavegrad with modification to fit the framework
# ==============================================================================

import numpy as np
import os
import random
import torch
import torchaudio
from pathlib import Path

from glob import glob
from torch.utils.data.distributed import DistributedSampler
from .data_loaders import generate_inventory

class NumpyDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, datatype, sample_rate=8000, T=-1):
        if datatype not in ['.wav', '.spec.npy', '.mel.npy']:
            raise NotImplementedError
        self.datatype = datatype
        self.sample_rate = sample_rate
        # number of frame to load
        self.T = T

        self.clean_path = Path('{}/clean'.format(data_root))
        self.noisy_path = Path('{}/noisy'.format(data_root))

        self.inventory = generate_inventory(self.clean_path, '.wav')
        self.data_len = len(self.inventory)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        audio_filename = self.inventory[idx]
        audio, _ = torchaudio.load(self.clean_path/audio_filename)

        if self.datatype == '.spec.npy' or self.datatype == '.mel.npy':
            spec_filename = f'{audio_filename}{self.datatype}'
            noisy = np.load(self.noisy_path/spec_filename)

        return {
            'audio': audio,
            'spectrogram': noisy,
            'index': idx
        }

    def getName(self, idx):
        filename = self.inventory[idx]

        return filename.split('.', 1)[0]


class Collator:
    def __init__(self, hop_samples, crop_mel_frames):
        self.hop_samples = hop_samples
        self.crop_mel_frames = crop_mel_frames

    def collate(self, minibatch):
        samples_per_frame = self.hop_samples
        for record in minibatch:
            # Filter out records that aren't long enough.
            if record['spectrogram'].shape[-1] < self.crop_mel_frames:
                del record['spectrogram']
                del record['audio']
                del record['index']
                continue

            start = random.randint(0, record['spectrogram'].shape[-1] - self.crop_mel_frames)
            end = start + self.crop_mel_frames
            record['spectrogram'] = record['spectrogram'][:, start:end]

            start *= samples_per_frame
            end *= samples_per_frame
            record['audio'] = record['audio'][:, start:end]
            record['audio'] = np.pad(record['audio'], ((0,0), (0, (end - start) - record['audio'].shape[-1])), mode='constant')

        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
        index = np.stack([record['index'] for record in minibatch if 'index' in record])

        return torch.from_numpy(audio), torch.from_numpy(spectrogram), torch.from_numpy(index)


class WaveGradDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, hop_samples, crop_mel_frames, num_workers, is_distributed=False):
        super().__init__(dataset,
                         batch_size=batch_size,
                         collate_fn=Collator(hop_samples, crop_mel_frames).collate,
                         shuffle=not is_distributed,
                         sampler=DistributedSampler(dataset) if is_distributed else None,
                         pin_memory=True,
                         drop_last=True,
                         num_workers=num_workers)
