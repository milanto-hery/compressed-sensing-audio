"""Utility functions: framing, file IO, small helpers."""

import numpy as np
import soundfile as sf
import os
import librosa


def load_mono_wav(path):
    x, sr = sf.read(path)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return np.asarray(x), int(sr)


def save_wav(path, x, sr, dtype='float32'):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    sf.write(path, np.asarray(x, dtype=dtype), sr)


def frame_indices(n_samples, frame_size, hop):
    """Return list of (start,end) for frames that cover the signal and include last tail."""
    starts = list(range(0, n_samples - frame_size + 1, hop))
    if len(starts) == 0:
        starts = [0]
    elif starts[-1] + frame_size < n_samples:
        starts.append(n_samples - frame_size)
    return [(st, st + frame_size) for st in starts]


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

       
def compute_mse(x1, x2):
    """Compute mean squared error between two audio arrays."""
    min_len = min(len(x1), len(x2))
    return np.mean((x1[:min_len] - x2[:min_len])**2)

def spec_db(x, n_fft=2048, hop_length=512):
    """Compute log-amplitude spectrogram (in dB) from 1D audio."""
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db