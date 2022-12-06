from math import ceil, floor

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def next_power_of_2(x: Tensor) -> int:
    return 2 ** ceil(x.item()).bit_length()


def get_center_frequencies(
    num_octaves: int, num_bins_per_octave: int, sample_rate: int  # C  # B  # Xi_s
) -> Tensor:  # Xi_k for k in [1, 2*K+1]
    """Compute log scaled center frequencies tensor"""
    frequency_nyquist = sample_rate / 2
    frequency_min = frequency_nyquist / (2**num_octaves)
    num_bins = num_octaves * num_bins_per_octave  # K
    # Exponential increase from min to Nyquist
    frequencies = frequency_min * (2 ** (torch.arange(num_bins) / num_bins_per_octave))
    frequencies_all = torch.cat(
        [
            frequencies,
            torch.tensor([frequency_nyquist]),
            # sample_rate - torch.flip(frequencies, dims=[0]) # not necessary
        ],
        dim=0,
    )
    return frequencies_all


def get_bandwidths(
    num_octaves: int,  # C
    num_bins_per_octave: int,  # B
    sample_rate: int,  # Xi_s
    frequencies: Tensor,  # Xi_k for k in [1, 2*K+1]
) -> Tensor:  # Omega_k for k in [1, 2*K+1]
    """Compute bandwidths tensor from center frequencies"""
    num_bins = num_octaves * num_bins_per_octave  # K
    q_factor = 1.0 / (
        2 ** (1.0 / num_bins_per_octave) - 2 ** (-1.0 / num_bins_per_octave)
    )
    bandwidths = frequencies[1 : num_bins + 1] / q_factor
    bandwidths_symmetric = (
        torch.flip(frequencies[1 : num_bins + 1], dims=[0]) / q_factor
    )
    bandwidths_all = torch.cat(
        [
            bandwidths,
            torch.tensor([sample_rate - 2 * frequencies[num_bins]]),
            bandwidths_symmetric,
        ],
        dim=0,
    )
    return bandwidths_all


def get_windows_range_indices(
    lengths: Tensor, positions: Tensor, power_of_2_length: bool
) -> Tensor:
    """Compute windowing tensor of indices"""
    num_bins = lengths.shape[0] // 2
    max_length = next_power_of_2(lengths.max()) if power_of_2_length else lengths.max()
    ranges = []
    for i in range(num_bins):
        start = positions[i] - max_length
        ranges += [torch.arange(start=start, end=start + max_length)]  # type: ignore
    return torch.stack(ranges, dim=0).long()


def get_windows(lengths: Tensor, power_of_2_length: bool) -> Tensor:
    """Compute tensor of stacked (centered) windows"""
    num_bins = lengths.shape[0] // 2
    max_length = next_power_of_2(lengths.max()) if power_of_2_length else lengths.max()
    windows = []
    for length in lengths[:num_bins]:
        # Pad windows left and right to center them
        pad_left = floor(max_length / 2 - length / 2)  # type: ignore
        pad_right = int(max_length - length - pad_left)
        windows += [F.pad(torch.hann_window(int(length)), pad=(pad_left, pad_right))]
    return torch.stack(windows, dim=0)


def get_windows_inverse(windows: Tensor, lengths: Tensor) -> Tensor:
    num_bins = windows.shape[0]
    return torch.einsum("k m, k -> k m", windows**2, lengths[:num_bins])


class CQT(nn.Module):
    def __init__(
        self,
        num_octaves: int,
        num_bins_per_octave: int,
        sample_rate: int,
        block_length: int,
        power_of_2_length: bool = False,
    ):
        super().__init__()
        self.block_length = block_length

        frequencies = get_center_frequencies(
            num_octaves=num_octaves,
            num_bins_per_octave=num_bins_per_octave,
            sample_rate=sample_rate,
        )

        bandwidths = get_bandwidths(
            num_octaves=num_octaves,
            num_bins_per_octave=num_bins_per_octave,
            sample_rate=sample_rate,
            frequencies=frequencies,
        )

        window_lengths = torch.round(bandwidths * block_length / sample_rate)

        self.register_buffer(
            "windows_range_indices",
            get_windows_range_indices(
                lengths=window_lengths,
                positions=torch.round(frequencies * block_length / sample_rate),
                power_of_2_length=power_of_2_length,
            ),
        )

        self.register_buffer(
            "windows",
            get_windows(lengths=window_lengths, power_of_2_length=power_of_2_length),
        )

        self.register_buffer(
            "windows_inverse",
            get_windows_inverse(windows=self.windows, lengths=window_lengths),  # type: ignore # noqa
        )

    def encode(self, waveform: Tensor) -> Tensor:
        frequencies = torch.fft.fft(waveform)
        crops = frequencies[:, :, self.windows_range_indices]
        crops_windowed = torch.einsum("... t k, t k -> ... t k", crops, self.windows)
        transform = torch.fft.ifft(crops_windowed)
        return transform

    def decode(self, transform: Tensor) -> Tensor:
        b, c, length = *transform.shape[0:2], self.block_length
        crops_windowed = torch.fft.fft(transform)
        crops_unwindowed = crops_windowed  # TODO crops_unwindowed = torch.einsum('... t k, t k -> ... t k', transformed, self.windows_inverse) # noqa
        frequencies = torch.zeros(b, c, length).to(transform)
        frequencies.scatter_add_(
            dim=-1,
            index=self.windows_range_indices.view(-1).expand(b, c, -1) % length,  # type: ignore # noqa
            src=crops_unwindowed.view(b, c, -1),
        )
        waveform = torch.fft.ifft(frequencies)
        return waveform
