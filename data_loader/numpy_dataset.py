# Copied from wavegrad with modification to fit the framework
# ==============================================================================

import numpy as np
import os
import random
import torch
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, _ = torchaudio.load(audio_filename, normalize=False)
    spectrogram = np.load(spec_filename)
    return {
        'audio': signal[0] / 32767.5,
        'spectrogram': spectrogram.T,
    }

  def getName(self, idx):
      full_filename = self.filenames[idx]
      _, filename = os.path.split(full_filename)

      return filename.split('.', 1)[0]


class InferDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, _ = torchaudio.load(audio_filename, normalize=False)
    spectrogram = np.load(spec_filename)
    return signal[0] / 32767.5, torch.from_numpy(spectrogram)

  def getName(self, idx):
      full_filename = self.filenames[idx]
      _, filename = os.path.split(full_filename)

      return filename.split('.', 1)[0]



class Collator:
  def __init__(self, hop_samples, crop_mel_frames ):
      self.hop_samples = hop_samples
      self.crop_mel_frames = crop_mel_frames

  def collate(self, minibatch):
    samples_per_frame = self.hop_samples
    for record in minibatch:
      # Filter out records that aren't long enough.
      if len(record['spectrogram']) < self.crop_mel_frames:
        del record['spectrogram']
        del record['audio']
        del record['index']
        continue

      start = random.randint(0, record['spectrogram'].shape[0] - self.crop_mel_frames)
      end = start + self.crop_mel_frames
      record['spectrogram'] = record['spectrogram'][start:end].T

      start *= samples_per_frame
      end *= samples_per_frame
      record['audio'] = record['audio'][start:end]
      record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')

    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    return torch.from_numpy(audio), torch.from_numpy(spectrogram), None


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

