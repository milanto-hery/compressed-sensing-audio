import os
import glob
from src.compress import encode_audio_global
from src.reconstruct import decode_and_reconstruct

def run_cs_batch(input_folder, output_folder, R=0.15, solver='fista', frame_size=2048, overlap=0.5, seed=0):
    """
    Compress and reconstruct all WAV files in a folder using CS.

    Args:
        input_folder (str): path to folder with .wav files
        output_folder (str): path to save .npz and reconstructed .wav
        R (float): compression ratio per file (0 < R <= 1)
        solver (str): 'fista', 'lasso', or 'omp'
        frame_size (int): frame size for reconstruction
        overlap (float): frame overlap fraction
        seed (int): RNG seed for deterministic sampling
    """
    os.makedirs(output_folder, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_folder, '*.wav'))

    print(f"Found {len(wav_files)} WAV files in {input_folder}")

    for wav_path in wav_files:
        name = os.path.splitext(os.path.basename(wav_path))[0]
        cs_file = os.path.join(output_folder, f"{name}.npz")
        out_wav = os.path.join(output_folder, f"{name}_rec.wav")

        print(f"\nProcessing {name}...")

        # Compress
        encode_audio_global(
            wav_path, cs_file,
            R=R, seed=seed,
            frame_size=frame_size,
            overlap=overlap
        )
        print(f"Compressed saved: {cs_file}")

        # Reconstruct
        decode_and_reconstruct(
            cs_file, out_wav,
            solver=solver
        )
        print(f"Reconstructed WAV saved: {out_wav}")

    print("\nBatch processing complete.")
