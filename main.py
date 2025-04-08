import streamlit as st
import numpy as np
from skimage import color, graph
from skimage.segmentation import slic, mark_boundaries
from PIL import Image

st.title("Image Segmentation with SLIC and Normalized Cuts")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

@st.cache_data
def run_slic(image_np, n_segments, compactness):
    segments = slic(image_np, n_segments=n_segments, compactness=compactness, start_label=1, convert2lab=True)
    colorized = color.label2rgb(segments, image_np, kind='avg')
    boundaries = mark_boundaries(image_np, segments)
    return colorized, boundaries

@st.cache_data
def run_normalized_cuts(image_np, n_segments, compactness):
    labels = slic(image_np, compactness=compactness, n_segments=n_segments, start_label=1, convert2lab=True)
    rag = graph.rag_mean_color(image_np, labels, mode='similarity')
    labels2 = graph.cut_normalized(labels, rag)
    colorized = color.label2rgb(labels2, image_np, kind='avg')
    boundaries = mark_boundaries(image_np, labels2)
    return colorized, boundaries

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    st.image(image_np, caption="Original Image", use_column_width=True)

    method = st.selectbox("Select Segmentation Method", ["SLIC", "Normalized Cuts"])
    n_segments = st.slider("Number of segments", min_value=10, max_value=1000, value=250)
    compactness = st.slider("Compactness", min_value=1.0, max_value=50.0, value=10.0)

    if method == "SLIC":
        with st.spinner("Running SLIC segmentation..."):
            colorized, boundaries = run_slic(image_np, n_segments, compactness)
    else:
        with st.spinner("Running Normalized Cuts segmentation..."):
            colorized, boundaries = run_normalized_cuts(image_np, n_segments, compactness)

    display_type = st.radio("Display Mode", ["Colorized Segments", "Boundary Overlay"])

    if display_type == "Colorized Segments":
        st.image(colorized, caption=f"{method} - Colorized Segments", use_column_width=True)
    else:
        st.image(boundaries, caption=f"{method} - Boundary Overlay", use_column_width=True)

    st.success("Segmentation complete!")
