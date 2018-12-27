import numpy as np
from torch.utils.data import Dataset
import torch

sample_rate = 22050
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = int(sample_rate * 0.0125) # 12.5ms
win_length = int(sample_rate * 0.05)   # 50ms
fmin = 40
min_level_db = -100
ref_level_db = 20
seq_len = hop_length * 5

bits = 9


class AudiobookDataset(Dataset):
    def __init__(self, ids, path):
        self.path = path
        self.metadata = ids

    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(f'{self.path}/mels{file}.npy')
        x = np.load(f'{self.path}/quant{file}.npy')
        return m, x

    def __len__(self):
        return len(self.metadata)


def collate(batch):
    pad = 2
    mel_win = seq_len // hop_length + 2 * pad
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    coarse = [x[1][sig_offsets[i]:sig_offsets[i] + seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.int64)

    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)

    x_input = 2 * coarse[:, :seq_len].float() / (2 ** bits - 1.) - 1.

    y_coarse = coarse[:, 1:]

    return x_input, mels, y_coarse