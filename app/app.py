import streamlit as st
import os
import sys
import tempfile
import glob

# Add parent folder for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cs_batch import run_cs_batch

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.set_page_config(page_title="Compressed Sensing Batch App", layout="wide")
st.title("Compressed Sensing Audio – Batch Processing")

st.markdown("""
Upload multiple WAV files and process them using compressed sensing.
Supports FISTA, LASSO, and OMP.
""")

# -------------------------------------------------------------
# Session state
# -------------------------------------------------------------
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

if "progress" not in st.session_state:
    st.session_state["progress"] = 0

if "progress_text" not in st.session_state:
    st.session_state["progress_text"] = ""

if "output_folder" not in st.session_state:
    st.session_state["output_folder"] = "saved_results"


# -------------------------------------------------------------
# LEFT: Upload + preview
# -------------------------------------------------------------
left, right = st.columns([2, 3])

with left:
    st.subheader("Upload WAV files")

    uploaded = st.file_uploader(
        "Select multiple WAV/WA files",
        type=["wav", "WA"],
        accept_multiple_files=True
    )

    if uploaded:
        st.session_state["uploaded_files"] = uploaded
        st.success(f"{len(uploaded)} files uploaded.")

        # Preview first 3
        st.text_area(
            "Preview (first 3 files):",
            "\n".join([f.name for f in uploaded[:3]]),
            height=100
        )

    # Output folder
    st.subheader("Output Folder Name")
    output_folder = st.text_input("Name:", value=st.session_state["output_folder"])
    st.session_state["output_folder"] = output_folder

    # Create folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)


# -------------------------------------------------------------
# RIGHT: Parameters
# -------------------------------------------------------------
with right:
    st.subheader("Processing Parameters")

    solver = st.selectbox("Algorithm", ["fista", "lasso", "omp"])

    R = st.slider("Compression ratio R", 0.01, 1.0, 0.1, 0.01)

    overlap = st.slider("Frame overlap", 0.0, 0.9, 0.5, 0.05)

    frame_size = st.selectbox(
        "Frame Size (samples)",
        [256, 512, 1024, 2048, 4096, 8192],
        index=2
    )


# -------------------------------------------------------------
# Convert uploaded files to temporary folder for batch process
# -------------------------------------------------------------
def prepare_temp_folder(uploaded_files):
    temp_dir = tempfile.mkdtemp()

    for f in uploaded_files:
        file_path = os.path.join(temp_dir, f.name)
        with open(file_path, "wb") as out:
            out.write(f.read())

    return temp_dir


# -------------------------------------------------------------
# Run button
# -------------------------------------------------------------
if st.button("🚀 Run Batch Processing"):
    if not st.session_state["uploaded_files"]:
        st.error("Please upload WAV files first.")
    else:
        st.success("Starting batch processing…")

        temp_input = prepare_temp_folder(st.session_state["uploaded_files"])

        # Call your batch processor
        run_cs_batch(
            input_folder=temp_input,
            output_folder=output_folder,
            solver=solver,
            R=R,
            overlap=overlap,
            frame_size=frame_size,
            update_fn=lambda p, txt: (
                st.session_state.update({"progress": p, "progress_text": txt})
            )
        )

# -------------------------------------------------------------
# Progress display
# -------------------------------------------------------------
st.progress(st.session_state["progress"])
st.write(st.session_state["progress_text"])
