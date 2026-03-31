import os
import cv2
import numpy as np
import streamlit as st
from SimilarityEngine import SimilarityEngine

# --- 1. Setup & Resource Loading ---
@st.cache_resource
def get_engine():
    base = os.path.dirname(os.path.abspath(__file__))
    return SimilarityEngine(
        detector_path=os.path.join(base, "face_detection_yunet_2023mar.onnx"), 
        embedder_path=os.path.join(base, "face_recognition_sface_2021dec.onnx")
    )

engine = get_engine()

def prepare_display_img(img, size=(150, 150)):
    """Standardizes image sizes for the UI grid."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def load_example_data():
    """Triggers the example paths in session state."""
    base = os.path.dirname(os.path.abspath(__file__))
    st.session_state['child_file'] = os.path.join(base, "examples", "test_child.jpeg")
    st.session_state['mother_file'] = os.path.join(base, "examples", "test_mother.jpeg")
    st.session_state['father_file'] = os.path.join(base, "examples", "test_father.jpeg")

@st.cache_data
def process_upload(input_source, label):
    """Handles both file buffers and local string paths."""
    if input_source is None:
        return None
    
    # Check if input is a file buffer or a string path and load accordingly
    if isinstance(input_source, str):
        img = cv2.imread(input_source)
    else:
        file_bytes = np.asarray(bytearray(input_source.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        input_source.seek(0) 

    if img is None:
        st.error(f"Could not load image for {label}.")
        return None
    
    # Check lighting conditions
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    if avg_brightness < 40 or avg_brightness > 220:
        st.error(f"⚠️ **{label} image lighting is poor.** Please use a clearer photo.")
        return None

    # Face detection and extracting patches
    full_face, aligned, patches = engine.find_faces(img)
    if full_face is None:
        st.error(f"🚫 **No face detected in {label} image.**")
        return None
    
    return {"full": full_face, "aligned": aligned, "patches": patches}

# --- 2. UI Configuration ---
st.set_page_config(page_title="Family Similarity Project", layout="wide")
st.title("🧬 Baby Matcher")
st.info("📸 Upload your photos to see the resemblance, or use the button on the side to try it out with sample images!")

if 'child_file' not in st.session_state:
    st.session_state['child_file'] = None
    st.session_state['mother_file'] = None
    st.session_state['father_file'] = None

# Sidebar Demo Control
st.sidebar.header("Control Panel")
if st.sidebar.button("🚀 Load Example Family"):
    load_example_data()
if st.sidebar.button("🗑️ Clear All"):
    st.session_state['child_file'] = None
    st.session_state['mother_file'] = None
    st.session_state['father_file'] = None
    st.rerun()
st.sidebar.markdown("**Note:** For best results, use clear, frontal photos with good lighting.")
st.sidebar.markdown("Only use for entertainment purposes. It is not a scientifically accurate tool for determining familial relationships.")
st.sidebar.markdown("Your images are not saved or shared. They are deleted after your session ends.")

# --- 3. Image Inputs ---
cols = st.columns(3)
child_upload = cols[0].file_uploader("Upload Child", type=["jpg", "jpeg", "png"])
mother_upload = cols[1].file_uploader("Upload Mother", type=["jpg", "jpeg", "png"])
father_upload = cols[2].file_uploader("Upload Father", type=["jpg", "jpeg", "png"])

# Priority: Uploaded File > Session State Example
c_input = child_upload if child_upload else st.session_state['child_file']
m_input = mother_upload if mother_upload else st.session_state['mother_file']
f_input = father_upload if father_upload else st.session_state['father_file']

uploads = [c_input, m_input, f_input]
labels = ["Child", "Mother", "Father"]

# Process and Display Previews
results = []
for i, upload in enumerate(uploads):
    res = process_upload(upload, labels[i])
    results.append(res)
    if res:
        preview = prepare_display_img(cv2.cvtColor(res["full"], cv2.COLOR_BGR2RGB))
        cols[i].image(preview, caption=f"{labels[i]} Detected", use_container_width=True)

# --- 4. Comparison Results ---
if all(results):
    c, m, f = results

    st.divider()
    st.subheader("🏆 Overall Resemblance")
    
    # Calculate overall scores using aligned 112x112 faces
    emb_c = engine.get_face_embedding(c["aligned"])
    emb_m = engine.get_face_embedding(m["aligned"])
    emb_f = engine.get_face_embedding(f["aligned"])
    
    sim_mom = max(0, float(engine.compare_faces(emb_c, emb_m)))
    sim_dad = max(0, float(engine.compare_faces(emb_c, emb_f)))
    
    total_overall = sim_mom + sim_dad
    mom_share = (sim_mom / total_overall) if total_overall > 0 else 0.5
    dad_share = 1.0 - mom_share

    v_col1, v_col2 = st.columns([2, 1])
    with v_col1:
        st.progress(float(mom_share), text=f"Balance: {mom_share:.1%} Mother | {dad_share:.1%} Father")
        if abs(mom_share - 0.5) < 0.03:
            st.info("✨ **Perfect Harmony:** The child is an equal blend of both parents.")
        else:
            parent = "Mother" if mom_share > dad_share else "Father"
            st.success(f"🎨 The child significantly favors **{parent}'s** features.")

    with v_col2:
        winner = "MOTHER" if mom_share > dad_share else "FATHER"
        color = "violet" if mom_share > dad_share else "orange"
        st.markdown(f"### Result: :{color}[{winner}]")

    st.divider()
    st.subheader("🔍 Feature-by-Feature Breakdown")
    
    features = ["eyes", "nose", "mouth"]
    for feat in features:
        p_cols = st.columns([1, 1, 1, 2])

        # Extract and standardize patches
        p_c = prepare_display_img(c["patches"][feat])
        p_m = prepare_display_img(m["patches"][feat])
        p_f = prepare_display_img(f["patches"][feat])

        p_cols[0].image(cv2.cvtColor(p_c, cv2.COLOR_BGR2RGB))
        p_cols[1].image(cv2.cvtColor(p_m, cv2.COLOR_BGR2RGB))
        p_cols[2].image(cv2.cvtColor(p_f, cv2.COLOR_BGR2RGB))

        # Relative similarity for patches
        s_m = max(0, float(engine.compare_faces(engine.get_face_embedding(p_c), engine.get_face_embedding(p_m))))
        s_f = max(0, float(engine.compare_faces(engine.get_face_embedding(p_c), engine.get_face_embedding(p_f))))

        t_s = s_m + s_f
        n_m = s_m / t_s if t_s > 0 else 0.5

        with p_cols[3]:
            st.write(f"**{feat.capitalize()} Heritage**")
            st.progress(float(n_m), text=f"Mom: {n_m:.1%} | Dad: {1-n_m:.1%}")