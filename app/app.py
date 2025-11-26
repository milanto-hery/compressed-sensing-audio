import streamlit as st
import os
import sys
import tempfile

# Add parent folder for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cs_batch import run_cs_batch

# Streamlit Setup
st.set_page_config(page_title="Compressed Sensing Batch App", layout="wide")
st.title("Compressed Sensing Audio – Batch Processing")

st.markdown("""
Upload multiple WAV files from your computer and process them using 
**FISTA**, **LASSO**, or **OMP** compressed sensing reconstruction.
""")

# Session State
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

if "outputs" not in st.session_state:
    st.session_state["outputs"] = []

if "progress" not in st.session_state:
    st.session_state["progress"] = 0

if "progress_text" not in st.session_state:
    st.session_state["progress_text"] = ""

if "output_folder" not in st.session_state:
    st.session_state["output_folder"] = "saved_results"


# File upload and output folder
left, right = st.columns([2, 3])

with left:
    st.subheader("Upload WAV Files")
    uploaded = st.file_uploader(
        "Select .wav files (multiple allowed)",
        type=["wav"],
        accept_multiple_files=True
    )

    if uploaded:
        st.session_state["uploaded_files"] = uploaded
        st.success(f"{len(uploaded)} files uploaded.")


    st.subheader("Output Folder Name")
    output_folder = st.text_input("Folder:", value=st.session_state["output_folder"])
    st.session_state["output_folder"] = output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)



# Parameters and run button
with right:
    st.subheader("Processing Parameters")

    solver = st.selectbox("Algorithm", ["fista", "lasso", "omp"])

    R = st.slider("Compression Ratio R", 0.01, 1.0, 0.1, 0.01)

    overlap = st.slider("Frame Overlap", 0.0, 0.9, 0.5, 0.05)

    frame_size = st.selectbox(
        "Frame Size (samples)",
        [256, 512, 1024, 2048, 4096, 8192],
        index=2
    )

    st.markdown("### ")
    run_button = st.button("🚀 Run Batch Processing", use_container_width=True)

# Helper: Save uploaded files to a temporary folder
def prepare_temp_folder(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    paths = []
    for file in uploaded_files:
        temp_path = os.path.join(temp_dir, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.read())
        paths.append(temp_path)
    return temp_dir



# Run the process
if run_button:
    if not st.session_state["uploaded_files"]:
        st.error("Please upload WAV files first.")
    else:
        st.success("Starting batch processing…")

        # prepare temporary input folder
        temp_input = prepare_temp_folder(st.session_state["uploaded_files"])

        # reset output list
        st.session_state["outputs"] = []

        # Get number of files
        wav_files = [f.name for f in st.session_state["uploaded_files"]]
        total_files = len(wav_files)

        # Progress bar widget
        overall_progress = st.progress(0)
        status = st.empty()

        current_index = 0

        # Process one file at a time manually
        import glob
        import shutil
        from src.compress import encode_audio_global
        from src.reconstruct import decode_and_reconstruct

        for file in glob.glob(os.path.join(temp_input, "*.wav")):

            filename = os.path.basename(file)
            status.write(f"Processing: **{filename}**")

            # Output paths
            name = os.path.splitext(filename)[0]
            cs_file = os.path.join(temp_input, f"{name}.npz")
            out_wav = os.path.join(output_folder, f"{name}_rec.wav")

            # Encode + reconstruct
            encode_audio_global(
                file, cs_file,
                R=R, frame_size=frame_size,
                overlap=overlap, seed=0
            )

            decode_and_reconstruct(
                cs_file, out_wav,
                solver=solver, n_jobs=-1
            )

            # save for download
            st.session_state["outputs"].append(out_wav)

            # update progress
            current_index += 1
            overall_progress.progress(current_index / total_files)


        status.write("✔ All files processed!")


# Download links for the reconstructed files
if st.session_state["outputs"]:
    st.subheader("Download Reconstructed Files")

    for path in st.session_state["outputs"]:
        filename = os.path.basename(path)
        with open(path, "rb") as f:
            st.download_button(
                label=f"⬇ Download {filename}",
                data=f.read(),
                file_name=filename,
                mime="audio/wav"
            )
