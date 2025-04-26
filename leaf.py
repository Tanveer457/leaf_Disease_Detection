import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Leaf/Skin Disease Image Segmentation", layout="wide")

st.title("🌿 Leaf / Skin Disease Detection - Image Segmentation")
st.markdown("### Traditional Image Processing (No Deep Learning)")
st.markdown("Use the controls to apply different filters and segment the image.")

# Sidebar: Image upload
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image_np, caption="🖼️ Original Image", width=500)

    st.sidebar.markdown("## 🔧 Preprocessing Controls")

    # Convert to HSV for processing
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    st.sidebar.markdown("## 🎨 HSV Thresholding")
    h_min = st.sidebar.slider("Hue Min", 0, 179, 25)
    h_max = st.sidebar.slider("Hue Max", 0, 179, 75)
    s_min = st.sidebar.slider("Saturation Min", 0, 255, 50)
    s_max = st.sidebar.slider("Saturation Max", 0, 255, 255)
    v_min = st.sidebar.slider("Value Min", 0, 255, 50)
    v_max = st.sidebar.slider("Value Max", 0, 255, 255)

    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # === Remove white background ===
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    background_threshold = 230
    lower_white = np.array([background_threshold, background_threshold, background_threshold], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    white_background_mask = cv2.inRange(image_bgr, lower_white, upper_white)
    white_background_mask_inv = cv2.bitwise_not(white_background_mask)

    # Combine HSV mask with background exclusion
    combined_mask = cv2.bitwise_and(mask, white_background_mask_inv)

    # === Turn infected areas black ===
    inverted_mask = cv2.bitwise_not(combined_mask)
    segmented_image = cv2.bitwise_and(image_np, image_np, mask=inverted_mask)

    st.markdown("### 🎯 Segmented Image")
    st.image(segmented_image, caption="Segmented Image (Infected Area in Black)", width=500)

    # === 📊 Area Quantification ===
    st.markdown("### 📐 Infection Area Analysis")

    valid_pixels = cv2.countNonZero(white_background_mask_inv)
    infected_pixels = cv2.countNonZero(combined_mask)
    infection_percentage = (infected_pixels / valid_pixels) * 100 if valid_pixels > 0 else 0

    st.markdown(f"🧮 **Infected Area:** {infected_pixels} pixels")
    st.markdown(f"📊 **Percentage of Infection (Excluding Background):** `{infection_percentage:.2f}%`")

    # Severity labeling
    if infection_percentage < 5:
        severity = "🟢 Mild"
    elif infection_percentage < 20:
        severity = "🟡 Moderate"
    else:
        severity = "🔴 Severe"

    st.markdown(f"🧭 **Severity Level:** **{severity}**")

    # === 📈 Histogram of the segmented area ===
    st.markdown("### 📊 Histogram of Segmented Area (Grayscale)")

    gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    masked_gray = cv2.bitwise_and(gray_segmented, gray_segmented, mask=combined_mask)

    fig, ax = plt.subplots()
    ax.hist(masked_gray.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7) # type: ignore
    ax.set_title("Histogram of Segmented Area (Pixel Intensities)")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # === 📤 Download segmented image ===
    st.markdown("### 📁 Export Segmented Image")
    result_img = Image.fromarray(segmented_image)
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    st.download_button(
        label="📥 Download Segmented Image",
        data=buffered.getvalue(),
        file_name="segmented_result.png",
        mime="image/png"
    )

else:
    st.warning("Upload an image to start processing.")
