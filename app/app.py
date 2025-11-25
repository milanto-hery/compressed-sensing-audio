import streamlit as st
import glob
import sys
import os

# Add parent folder to Python path (for src imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cs_batch import run_cs_batch
from src.compress import encode_audio_global
from src.reconstruct import decode_and_reconstruct

# Streamlit UI
st.set_page_config(page_title="Compressed Sensing Audio - Batch", layout="wide")
st.title("Compressed Sensing Algorithm for Audio Signals")

st.markdown("""
Process batches of WAV files using compressed sensing.
This version works both **offline** and on **Streamlit Cloud**.
""")

# Session state
if 'batch_running' not in st.session_state:
    st.session_state['batch_running'] = False
if 'progress' not in st.session_state:
    st.session_state['progress'] = 0
if 'progress_text' not in st.session_state:
    st.session_state['progress_text'] = ""
if 'output_folder' not in st.session_state:
    st.session_state['output_folder'] = "saved_results"

# List of files
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# Filter only folders containing WAV files
wav_folders = []
for f in folders:
    full = os.path.join(base_path, f)
    if glob.glob(os.path.join(full, "*.wav")):
        wav_folders.append(f)

# Layout
left_col, right_col = st.columns([2,3])

# Folder selection
with left_col:
    st.subheader("Input Folder")

    if not wav_folders:
        st.error("⚠️ No folder containing WAV files found in the repository.")
    else:
        input_folder_name = st.selectbox("Choose folder with WAV files:", wav_folders)
        input_folder = os.path.join(base_path, input_folder_name)

        # Display preview
        if st.button("Show first 3 WAV files"):
            files = glob.glob(os.path.join(input_folder, "*.wav"))
            st.text_area(
                "First 3 WAV files",
                "\n".join([os.path.basename(f) for f in files[:3]]),
                height=100
            )

    # Output folder
    st.subheader("Output Folder")
    output_name = st.text_input("Folder name", value=st.session_state['output_folder'])
    st.session_state['output_folder'] = output_name
    output_folder = os.path.join(base_path, output_name)

    # Auto-create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

# Parameters
with right_col:
    st.subheader("Batch Parameters")
    solver = st.selectbox("Algorithm", ["fista", "lasso", "omp"])
    R = st.slider("Compression ratio R", 0.01, 1.0, 0.1, 0.01)
    overlap = st.slider("Frame overlap", 0.0, 0.9, 0.5, 0.05)

# -------------------------------------------------------------
# Run buttoms
start_col, stop_col = st.columns([1,1])

with start_col:
    if st.button("🚀 Run Batch Processing"):
        st.session_state['batch_running'] = True

        run_cs_batch(
            input_folder=input_folder,
            output_folder=output_folder,
            solver=solver,
            R=R,
            overlap=overlap,
            update_fn=lambda p, txt: (
                p := st.session_state.update({"progress": p, "progress_text": txt})
            )
        )

with stop_col:
    if st.button("⛔ Cancel", type="secondary"):
        st.warning("Stop functionality not implemented yet.")

# Progress bar
st.progress(st.session_state['progress'])
st.write(st.session_state['progress_text'])
