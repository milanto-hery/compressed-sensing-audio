"""Encoding: global seeded sampling encoder (compact .npz output).

This module chooses M = floor(R * N) unique samples from the full audio using a seeded RNG
and saves only the sampled values plus small metadata. Indices are not saved; they are
regenerated at decode time using the same seed.
"""

import numpy as np
from src.utils import load_mono_wav, ensure_dir


def encode_audio_global(wav_path, out_cs_path, R=0.10, seed=0, frame_size=2048, overlap=0.5, dtype=np.float32):
    """Encode audio and save compressed .npz file containing only y (samples) + metadata.

    Args:
        wav_path: input wav file path
        out_cs_path: output compressed path (npz)
        R: compression ratio (0<R<=1)
        seed: RNG seed for deterministic sampling
        frame_size: frame length for reconstruction stage
        overlap: frame overlap fraction (0..1)
        dtype: dtype to store the y array (default float32)

    Returns:
        out_cs_path, meta dict
    """
    x, sr = load_mono_wav(wav_path)
    N = len(x)
    M = int(np.floor(R * N))
    if M < 1:
        raise ValueError("R * N must be >= 1")

    rng = np.random.default_rng(int(seed))
    global_idx = rng.choice(N, size=M, replace=False)
    global_idx = np.sort(global_idx)
    y = x[global_idx].astype(dtype)

    meta = dict(seed=int(seed), N=int(N), M=int(M), frame_size=int(frame_size), overlap=float(overlap), hop=int(frame_size * (1.0 - overlap)), R=float(R), sr=int(sr))

    ensure_dir(out_cs_path)
    np.savez_compressed(out_cs_path, y=y, **meta)
    return out_cs_path, meta