# Compressed Sensing for Audio Signal

Pipeline to compress audio by randomly sampling with small of samples and reconstruct using DCT + FISTA / LASSO / OMP.


1. Using the CLI

You can run the compressed sensing pipeline directly from the command line using the cs_audio_cli script.

Steps:

Clone the repository:

    git clone https://github.com/yourusername/compressed-sensing-audio.git
    cd compressed-sensing-audio
    pip install -r requirements.txt

Encode (compress) a single WAV file:

    python -m src.cs_audio_cli encode \
        --wav path/to/input.wav \
        --out path/to/output_compressed.npz \
        --R 0.1 \
        --frame_size 2048 \
        --overlap 0.5 \
        --seed 42
Decode (reconstruct) the compressed file:

    python -m src.cs_audio_cli decode \
        --cs path/to/output_compressed.npz \
        --out path/to/output_reconstructed.wav \
        --solver fista
R is the compression ratio (e.g., 0.1 = 10% of original samples)

solver can be fista, lasso, or omp

After decoding, the reconstructed .wav should sound very close to the original audio.

2. Using the Notebook

For interactive exploration and visualization:

- Open demo.ipynb in Jupyter Notebook or VSCode

- You can load a WAV file, compress and reconstruct it

- Compare waveforms, spectrograms, and MSE values

- Switch between FISTA, LASSO, and OMP

Click here to open the demo notebook