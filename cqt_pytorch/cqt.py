from math import ceil, floor
from typing import Optional, TypeVar

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, einsum, nn


def next_power_of_2(x: float) -> int:
    return 2 ** ceil(x).bit_length()


T = TypeVar("T")


def default(val: Optional[T], default: T) -> T:
    return val if val is not None else default


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
            sample_rate - torch.flip(frequencies, dims=[0]),
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
    q = 1.0 / (2 ** (1.0 / num_bins_per_octave) - 2 ** (-1.0 / num_bins_per_octave))
    bandwidths = frequencies[1 : num_bins + 1] / q
    bandwidths_symmetric = torch.flip(frequencies[1 : num_bins + 1], dims=[0]) / q
    bandwidths_all = torch.cat(
        [
            bandwidths,
            torch.tensor([sample_rate - 2 * frequencies[num_bins]]),
            bandwidths_symmetric,
        ],
        dim=0,
    )
    return bandwidths_all


def get_windows_range_idx(positions: Tensor, max_length: int) -> Tensor:
    """Compute windowing tensor of indices"""
    num_bins = positions.shape[0] // 2
    ranges = []
    for i in range(num_bins):
        start = positions[i] - max_length // 2
        ranges += [torch.arange(start=start, end=start + max_length)]  # type: ignore # noqa
    return torch.stack(ranges, dim=0).long()


def get_windows(lengths: Tensor, max_length: int) -> Tensor:
    """Compute tensor of stacked (centered) windows"""
    num_bins = lengths.shape[0] // 2
    windows = []
    for length in lengths[:num_bins]:
        # Pad windows left and right to center them
        pad_left = floor(max_length / 2 - length / 2)  # type: ignore
        pad_right = int(max_length - length - pad_left)
        windows += [F.pad(torch.hann_window(int(length)), pad=(pad_left, pad_right))]
    return torch.stack(windows, dim=0)


def get_windows_inverse(
    windows: Tensor, windows_range_idx: Tensor, max_length: int, block_length: int
) -> Tensor:
    """Compute tensor of stacked (centered) inverse windows"""
    windows_overlap = torch.zeros(block_length).scatter_add_(
        dim=0,
        index=windows_range_idx.view(-1),
        src=(windows**2).view(-1),
    )
    return windows / (windows_overlap[windows_range_idx] + 1e-8)


class CQT(nn.Module):
    def __init__(
        self,
        num_octaves: int,
        num_bins_per_octave: int,
        sample_rate: int,
        block_length: Optional[int] = None,
        power_of_2_length: bool = False,
    ):
        super().__init__()
        # Default to 1s block length
        block_length = default(block_length, sample_rate)

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
        max_window_length = int(window_lengths.max())

        if power_of_2_length:
            max_window_length = next_power_of_2(max_window_length)

        windows_range_idx = (
            get_windows_range_idx(
                max_length=max_window_length,
                positions=torch.round(frequencies * block_length / sample_rate),
            )
            % block_length
        )

        windows = get_windows(lengths=window_lengths, max_length=max_window_length)

        windows_inverse = get_windows_inverse(
            windows=windows,
            windows_range_idx=windows_range_idx,
            max_length=max_window_length,
            block_length=block_length,
        )

        self.block_length = block_length  # t
        self.max_window_length = max_window_length  # l
        self.register_buffer("windows_range_idx", windows_range_idx)
        self.register_buffer("windows", windows)
        self.register_buffer("windows_inverse", windows_inverse)

    def encode(self, waveform: Tensor) -> Tensor:
        nt, t = *waveform.shape[2:], self.block_length
        assert nt % t == 0, f"wave length {nt} must be divisible by block_length {t}"
        # Stack blocks in batch dimension
        waveform = rearrange(waveform, "b c (n t) -> (b n) c t", t=t)
        # Compute frequencies from waveform
        frequencies = torch.fft.fft(waveform)
        # Extract window crops from frequencies
        crops = frequencies[:, :, self.windows_range_idx]
        # Apply windows on all crops
        crops_windowed = einsum("... k l, k l -> ... k l", crops, self.windows)
        # Transform back to original domain
        transform = torch.fft.ifft(crops_windowed)
        # Rearrange transformed blocks lengthwise
        return rearrange(transform, "(b n) c k l -> b c k (n l)", n=nt // t)

    def decode(self, transform: Tensor) -> Tensor:
        b, c, _, nl, t, l = *transform.shape, self.block_length, self.max_window_length  # type: ignore # noqa
        assert nl % l == 0, f"transform length {nl} must be divisible by {l}"  # noqa
        # Compute number of blocks n
        n = nl // l
        # Stack transforms in batch dimension
        transform = rearrange(transform, "b c k (n l) -> (b n) c k l", n=n)
        # Compute frequencies of for each bin
        crops_windowed = torch.fft.fft(transform)
        # Apply inverse windows
        crops = einsum("... k l, k l -> ... k l", crops_windowed, self.windows_inverse)
        # Create tensor for output signal
        frequencies = torch.zeros(b * n, c, t).to(transform)
        # Overlap inverse windowed bins at right indices
        frequencies.scatter_add_(
            dim=-1,
            index=repeat(self.windows_range_idx, "k l -> bn c (k l)", bn=b * n, c=c),  # type: ignore # noqa
            src=rearrange(crops, "bn c k l -> bn c (k l)", l=l),
        )
        # Go back to real waveform domain
        waveform = torch.fft.irfft(frequencies, t)
        # Rearrange waveform blocks timewise
        return rearrange(waveform, "(b n) c t -> b c (n t)", n=n)
