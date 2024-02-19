import os
import pandas as pd
import torchaudio
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


""" =========== Constants =============== """""
SAMPLE_RATE = 44100

"""
The class imports helps import the Dataset into the training.
"""


class CustomImageDataset(Dataset):
    def __init__(self, dataset_paths_file, img_dir):
        self.sample_rate = SAMPLE_RATE
        self.audio_labels = pd.read_csv(dataset_paths_file)
        self.audio_dir = img_dir
        self.duration_in_sec = 5.0
        self.num_samples = int(self.duration_in_sec * self.sample_rate)
        self.generator = torch.manual_seed(0)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        original_audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 0])
        no_drums_audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 1])
        orig_waveform, sr1 = torchaudio.load(original_audio_path)
        no_drums_waveform, sr2 = torchaudio.load(no_drums_audio_path)
        label = self.audio_labels.iloc[idx, 2]

        # if the data is not in the correct sample rate, we resample it.
        if sr1 != self.sample_rate:
            orig_waveform = torchaudio.functional.resample(orig_waveform, sr1, self.sample_rate)
        if sr2 != self.sample_rate:
            no_drums_waveform = torchaudio.functional.resample(no_drums_waveform, sr2, self.sample_rate)

        orig_waveform = self.crop_audio_randomly(orig_waveform, self.num_samples, self.generator)
        no_drums_waveform = self.crop_audio_randomly(no_drums_waveform)
        # create log mel spec
        log_mel_orig, stft_orig = self._create_mel_spectogram(orig_waveform)
        log_mel_no_drums, stft_nodrums = self._create_mel_spectogram(no_drums_waveform)
        return log_mel_orig, log_mel_no_drums, orig_waveform, no_drums_waveform

    import torch

    def crop_audio_randomly(self, audio_tensor, length, generator=None):
        """
        Crops the audio tensor to a specified length at a random start index,
        using a PyTorch generator for randomness.

        Parameters:
        - audio_tensor (torch.Tensor): The input audio tensor.
        - length (int): The desired length of the output tensor.
        - generator (torch.Generator, optional): A PyTorch generator for deterministic randomness.

        Returns:
        - torch.Tensor: The cropped audio tensor of the specified length.
        """

        # Ensure the desired length is not greater than the audio tensor length
        if length > audio_tensor.size(0):
            raise ValueError("Desired length is greater than the audio tensor length.")

        # Calculate the maximum start index for cropping
        max_start_index = audio_tensor.size(0) - length

        # Generate a random start index from 0 to max_start_index using the specified generator
        start_index = torch.randint(0, max_start_index + 1, (1,), generator=generator).item()

        # Crop the audio tensor from the random start index to the desired length
        cropped_audio = audio_tensor[start_index:start_index + length]

        return cropped_audio

    """
    Creates a mel spectogram.
    """

    def _create_mel_spectogram(self, waveform):
        if torch.min(waveform) < -1.0:
            print("train min value is ", torch.min(waveform))
        if torch.max(waveform) > 1.0:
            print("train max value is ", torch.max(waveform))

        if self.mel_fmax not in self.mel_basis:
            mel = librosa_mel_fn(
                self.sampling_rate,
                self.filter_length,
                self.n_mel,
                self.mel_fmin,
                self.mel_fmax,
            )
            self.mel_basis[str(self.mel_fmax) + "_" + str(waveform.device)] = (
                torch.from_numpy(mel).float().to(waveform.device)
            )
            self.hann_window[str(waveform.device)] = torch.hann_window(self.win_length).to(
                waveform.device
            )

        waveform = torch.nn.functional.pad(
            waveform.unsqueeze(1),
            (
                int((self.filter_length - self.hop_length) / 2),
                int((self.filter_length - self.hop_length) / 2),
            ),
            mode="reflect",
        )

        waveform = waveform.squeeze(1)

        stft_spec = torch.stft(
            waveform,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(waveform.device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        stft_spec = torch.abs(stft_spec)

        mel = spectral_normalize_torch(
            torch.matmul(
                self.mel_basis[str(self.mel_fmax) + "_" + str(waveform.device)], stft_spec
            )
        )

        return mel[0], stft_spec[0]
