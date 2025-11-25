import streamlit as st
# import os
import glob
# from src.cs_batch import run_cs_batch

import sys
import os

# Add the parent folder to PYTHONPATH so "src" can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cs_batch import run_cs_batch
from src.compress import encode_audio_global
from src.reconstruct import decode_and_reconstruct


st.set_page_config(page_title="Compressed Sensing Audio - Batch", layout="wide")
st.title("Compressed Sensing Algorithm for Audio Signals")

#st.title("Compressed Sensing Audio - Batch Processor")
st.markdown("""
Compress and reconstruct multiple WAV files using compressed sensing.
Supports FISTA, LASSO, and OMP reconstruction.
""")

# Initialize session state ---
if 'batch_running' not in st.session_state:
    st.session_state['batch_running'] = False
if 'progress' not in st.session_state:
    st.session_state['progress'] = 0
if 'progress_text' not in st.session_state:
    st.session_state['progress_text'] = ""
if 'input_folder' not in st.session_state:
    st.session_state['input_folder'] = ""
if 'output_folder' not in st.session_state:
    st.session_state['output_folder'] = "saved_results"

# Layout
left_col, right_col = st.columns([2,3])

# Left column: input folder + show files
with left_col:
    st.subheader("Input Folder")
    input_folder = st.text_input("Paste input folder path here:", value=st.session_state['input_folder'])
    st.session_state['input_folder'] = input_folder

    if input_folder and os.path.exists(input_folder):
        if st.button("Show first 3 WAV files"):
            files = glob.glob(os.path.join(input_folder, "*.wav"))
            st.text_area(
                "First 3 WAV files",
                "\n".join([os.path.basename(f) for f in files[:3]]),
                height=100
            )
    else:
        st.warning("Folder does not exist or is empty.")

    output_folder = st.text_input("Output folder name", value=st.session_state['output_folder'])
    st.session_state['output_folder'] = output_folder

# Right column: parameters
with right_col:
    st.subheader("Batch Parameters")
    solver = st.selectbox("Algorithm", ["fista", "lasso", "omp"])
    R = st.slider("Compression ratio R", 0.01, 1.0, 0.1, 0.01)
    overlap = st.slider("Frame overlap", 0.0, 0.9, 0.5, 0.05)
    seed = st.number_input("Random seed", 0, 9999, 42)

    # Frame size as selectbox
    frame_size = st.selectbox(
        "Frame size",
        options=[32, 64, 128, 256, 512, 1024, 2048],
        index=6  # default 2048
    )

# Progress UI
st.subheader("Progress")
progress_bar = st.progress(st.session_state['progress'])
progress_text = st.empty()
progress_text.text(st.session_state['progress_text'])

# Run batch function
def run_batch():
    if not st.session_state['input_folder'] or not os.path.exists(st.session_state['input_folder']):
        st.error("Please provide a valid input folder!")
        return

    st.session_state['batch_running'] = True
    output_folder_full = os.path.join(os.path.dirname(st.session_state['input_folder']),
                                      st.session_state['output_folder'])
    os.makedirs(output_folder_full, exist_ok=True)

    wav_files = glob.glob(os.path.join(st.session_state['input_folder'], '*.wav'))
    total_files = len(wav_files)

    for i, wav_path in enumerate(wav_files, 1):
        name = os.path.splitext(os.path.basename(wav_path))[0]
        cs_file = os.path.join(output_folder_full, f"{name}.npz")
        out_wav = os.path.join(output_folder_full, f"{name}_rec.wav")

        run_cs_batch(
            input_folder=st.session_state['input_folder'],
            output_folder=output_folder_full,
            R=R,
            solver=solver,
            frame_size=frame_size,
            overlap=overlap,
            seed=seed
        )

        # Update progress
        st.session_state['progress'] = i / total_files
        st.session_state['progress_text'] = f"Processing {i}/{total_files} files..."
        progress_bar.progress(st.session_state['progress'])
        progress_text.text(st.session_state['progress_text'])

    st.session_state['batch_running'] = False
    st.success("Batch processing completed!")

# Run button
if st.button("Run Batch Processing") and not st.session_state['batch_running']:
    run_batch()
