import streamlit as st
import numpy as np
from skimage import color, graph
from skimage.segmentation import slic, mark_boundaries
from PIL import Image
from skimage.filters import gaussian
from io import BytesIO
import zipfile

st.title("Image Segmentation with SLIC and Normalized Cuts")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

@st.cache_data
def resize_image_keep_ratio(image: Image.Image, max_width=960) -> Image.Image:
    width, height = image.size
    if width > max_width:
        new_height = int((max_width / width) * height)
        resized_image = image.resize((max_width, new_height), Image.LANCZOS)
        return resized_image
    return image

@st.cache_data
def run_slic(image_np, n_segments, compactness):
    segments = slic(image_np, n_segments=n_segments, compactness=compactness, start_label=1, convert2lab=True)
    colorized = color.label2rgb(segments, image_np, kind='avg')
    boundaries = mark_boundaries(image_np, segments)
    return colorized, boundaries, segments

@st.cache_data
def run_normalized_cuts(image_np, n_segments, compactness):
    labels = slic(image_np, compactness=compactness, n_segments=n_segments, start_label=1, convert2lab=True)
    rag = graph.rag_mean_color(image_np, labels, mode='similarity')
    labels2 = graph.cut_normalized(labels, rag)
    colorized = color.label2rgb(labels2, image_np, kind='avg')
    boundaries = mark_boundaries(image_np, labels2)
    return colorized, boundaries, labels2

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(resize_image_keep_ratio(image))
    image_gaussian = gaussian(image_np, sigma=1, channel_axis=2)
    st.image(image_np, caption="Original Image", use_column_width=True)

    # Process the segmentation form
    with st.form("segmentation_form"):
        method = st.selectbox("Select Segmentation Method", ["SLIC", "Normalized Cuts"])
        n_segments = st.number_input("Number of segments", min_value=10, max_value=500, value=250, step=10)
        compactness = st.number_input("Compactness", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
        display_type = st.radio("Display Mode", ["Colorized Segments", "Boundary Overlay"])
        submitted = st.form_submit_button("Submit")

    if submitted:
        # Run segmentation based on user's choice
        if method == "SLIC":
            with st.spinner("Running SLIC segmentation..."):
                colorized, boundaries, label_map = run_slic(image_np, int(n_segments), float(compactness))
        else:
            with st.spinner("Running Normalized Cuts segmentation..."):
                colorized, boundaries, label_map = run_normalized_cuts(image_np, int(n_segments), float(compactness))

        # Store segmentation results in session state
        st.session_state["segmentation_result"] = {
            "method": method,
            "colorized": colorized,
            "boundaries": boundaries,
            "label_map": label_map,
            "image_np": image_np  # Store the original image
        }

        st.success("Segmentation complete!")

    # Check if segmentation results are in session state
    if "segmentation_result" in st.session_state:
        result = st.session_state["segmentation_result"]
        method = result["method"]
        label_map = result["label_map"]
        image_np = result["image_np"]
        colorized = result["colorized"]
        boundaries = result["boundaries"]

        # Display image based on selected display mode
        if display_type == "Colorized Segments":
            st.image(colorized, caption=f"{method} - Colorized Segments", use_column_width=True)
        else:
            st.image(boundaries, caption=f"{method} - Boundary Overlay", use_column_width=True)

        with st.spinner("Processing download..."):
            # Create zip file with segments
            unique_labels = np.unique(label_map)
            if "segments_zip" not in st.session_state:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for idx, label_id in enumerate(unique_labels, start=1):
                        mask = label_map == label_id
                        segment = np.zeros_like(image_np)
                        segment[mask] = image_np[mask]

                        segment_pil = Image.fromarray(segment)
                        img_bytes = BytesIO()
                        segment_pil.save(img_bytes, format="PNG")
                        img_bytes.seek(0)

                        zip_file.writestr(f"{method.lower().replace(' ', '_')}_segment_{idx}.png", img_bytes.read())
                zip_buffer.seek(0)
                st.session_state["segments_zip"] = zip_buffer

            # Provide option to download all segments as ZIP
            st.download_button(
                label="Download All Segments as ZIP",
                data=st.session_state["segments_zip"],
                file_name=f"{method.lower().replace(' ', '_')}_segments.zip",
                mime="application/zip"
            )

            # Provide individual segment download links
            st.subheader("Download Individual Segments")
            cols_per_row = 3
            cols = st.columns(cols_per_row)

            for idx, label_id in enumerate(unique_labels, start=1):
                col = cols[(idx - 1) % cols_per_row]

                mask = label_map == label_id
                segment = np.zeros_like(image_np)
                segment[mask] = image_np[mask]

                segment_pil = Image.fromarray(segment)
                buf = BytesIO()
                segment_pil.save(buf, format="PNG")
                byte_img = buf.getvalue()

                with col:
                    st.image(segment, caption=f"Segment #{idx}", use_column_width=True)
                    st.download_button(
                        label=f"Download Segment #{idx}",
                        data=byte_img,
                        file_name=f"{method.lower().replace(' ', '_')}_segment_{idx}.png",
                        mime="image/png"
                    )
