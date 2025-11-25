"""Reconstruction utilities and decoders.

Contains these reconstruction solvers for one frame:
 - FISTA (DCT-domain implicit operator) -- recommended
 - LASSO (sklearn; builds explicit sensing matrix) -- small frames only
 - OMP (sklearn; small frames only)

Also contains decode_and_reconstruct() that reads the .npz compressed file and reconstructs the full wav with overlap-add.
"""

import numpy as np
from scipy.fft import dct, idct
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from joblib import Parallel, delayed
from .utils import save_wav, frame_indices


# --- helpers

def csmtx_dct(N, idx):
    K = len(idx)
    A = np.zeros((K, N), dtype=float)
    for i, j in enumerate(idx):
        e = np.zeros(N, dtype=float)
        e[j] = 1.0
        A[i, :] = dct(e, norm='ortho')
    return A


def soft_thresh(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


# --- frame solvers
def reconstruct_frame_fista(y_loc, idx_loc, N_frame, lam=1e-3, max_iter=400, tol=1e-6, verbose=False):
    mask = np.zeros(N_frame, dtype=bool)
    mask[idx_loc] = True
    y = np.zeros(N_frame, dtype=float)
    y[idx_loc] = y_loc

    def A(alpha):
        return idct(alpha, norm='ortho') * mask

    def At(r):
        return dct(r * mask, norm='ortho')

    L = 1.0
    t = 1.0 / L
    alpha = dct(y, norm='ortho')
    z = alpha.copy()
    tk = 1.0

    for k in range(max_iter):
        Az = A(z)
        grad = At(Az - y)
        alpha_next = soft_thresh(z - t * grad, lam * t)
        tk_next = (1 + np.sqrt(1 + 4 * tk**2)) / 2.0
        z = alpha_next + ((tk - 1) / tk_next) * (alpha_next - alpha)
        rel = np.linalg.norm(alpha_next - alpha) / (np.linalg.norm(alpha) + 1e-12)
        alpha = alpha_next
        tk = tk_next
        if rel < tol:
            break
    return idct(alpha, norm='ortho')


def reconstruct_frame_lasso(y_loc, idx_loc, N_frame, alpha=1e-5):
    if len(idx_loc) == 0:
        return np.zeros(N_frame, dtype=float)
    A = csmtx_dct(N_frame, idx_loc)
    model = Lasso(alpha=alpha, max_iter=10000, fit_intercept=False)
    model.fit(A, y_loc)
    coeffs = model.coef_
    return idct(coeffs, norm='ortho')


def reconstruct_frame_omp(y_loc, idx_loc, N_frame, n_nonzero=200):
    if len(idx_loc) == 0:
        return np.zeros(N_frame, dtype=float)
    A = csmtx_dct(N_frame, idx_loc)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(n_nonzero, A.shape[1]))
    omp.fit(A, y_loc)
    coeffs = omp.coef_
    return idct(coeffs, norm='ortho')


# --- top-level decoder
def decode_and_reconstruct(cs_npz_path, out_wav_path, solver='fista', fista_lam=1e-3, lasso_alpha=1e-3, omp_nonzero=200, n_jobs=-1, verbose=False):
    npz = np.load(cs_npz_path, allow_pickle=True)
    y = npz['y']
    seed = int(npz['seed'].tolist())
    N = int(npz['N'].tolist())
    M = int(npz['M'].tolist())
    frame_size = int(npz['frame_size'].tolist())
    overlap = float(npz['overlap'].tolist())
    hop = int(npz['hop'].tolist())
    sr = int(npz['sr'].tolist())

    rng = np.random.default_rng(seed)
    global_idx = rng.choice(N, size=M, replace=False)
    global_idx = np.sort(global_idx)

    # frames
    idx_ranges = frame_indices(N, frame_size, hop)

    frame_infos = []
    for st, ed in idx_ranges:
        mask = (global_idx >= st) & (global_idx < ed)
        if not np.any(mask):
            frame_infos.append((np.array([], dtype=float), np.array([], dtype=int), st, ed))
            continue
        loc_idx = (global_idx[mask] - st).astype(int)
        loc_y = y[mask]
        frame_infos.append((loc_y, loc_idx, st, ed))

    def worker(fi):
        loc_y, loc_idx, st, ed = fi
        Nf = ed - st
        if solver == 'fista':
            return reconstruct_frame_fista(loc_y, loc_idx, Nf, lam=fista_lam)
        elif solver == 'lasso':
            return reconstruct_frame_lasso(loc_y, loc_idx, Nf, alpha=lasso_alpha)
        elif solver == 'omp':
            return reconstruct_frame_omp(loc_y, loc_idx, Nf, n_nonzero=omp_nonzero)
        else:
            raise ValueError('Unsupported solver')

    reconstructed_frames = Parallel(n_jobs=n_jobs)(delayed(worker)(fi) for fi in frame_infos)

    recon = np.zeros(N, dtype=float)
    weight = np.zeros(N, dtype=float)
    for frame, (loc_y, loc_idx, st, ed) in zip(reconstructed_frames, frame_infos):
        recon[st:ed] += frame
        weight[st:ed] += 1.0
    nz = weight > 0
    recon[nz] /= weight[nz]

    save_wav(out_wav_path, recon.astype(np.float32), sr)
    if verbose:
        print(f"Saved reconstructed audio to {out_wav_path}")
    return recon
