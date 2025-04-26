import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Define functions for the Plant Health Monitoring System
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def segment_leaf(image, otsu_offset):
    _, binary = cv2.threshold(image, otsu_offset, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned

def detect_diseased_area(original_image, segmented_image, lower_hue, upper_hue):
    mask = cv2.bitwise_and(original_image, original_image, mask=segmented_image)
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    lower_color = np.array([lower_hue, 50, 50])
    upper_color = np.array([upper_hue, 255, 255])
    diseased_area = cv2.inRange(hsv, lower_color, upper_color)
    return diseased_area

def calculate_health_metrics(segmented_image, diseased_area):
    total_leaf_pixels = np.sum(segmented_image == 255)
    total_diseased_pixels = np.sum(diseased_area == 255)
    health_percentage = ((total_leaf_pixels - total_diseased_pixels) / total_leaf_pixels) * 100
    return health_percentage, total_diseased_pixels, total_leaf_pixels

def visualize_results(original_image, segmented_image, diseased_area):
    heatmap = cv2.applyColorMap(diseased_area, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)
    combined = np.hstack((original_image, cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR), overlay))
    return combined

# Streamlit GUI
st.title("🌱 Advanced Plant Health Monitoring System")
st.write("Upload an image of a leaf to analyze its health and explore advanced features.")

# File uploader
uploaded_file = st.file_uploader("📂 Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", width=200)

    # Sidebar for parameters
    st.sidebar.header("📊 Analysis Settings")
    otsu_offset = st.sidebar.slider("Otsu Threshold Offset", 0, 100, 0, step=1)
    lower_hue = st.sidebar.slider("Lower Hue for Diseased Area", 0, 50, 10, step=1)
    upper_hue = st.sidebar.slider("Upper Hue for Diseased Area", 50, 100, 30, step=1)

    # Tabs for interactive visualization
    tab1, tab2, tab3 = st.tabs(["Preprocessing", "Results", "Health Analysis"])

    # Preprocessing Tab
    with tab1:
        st.header("Preprocessing")
        blurred = preprocess_image(image)
        st.image(blurred, caption="Blurred Image", width=200)

    # Results Tab
    with tab2:
        st.header("Leaf Segmentation and Disease Detection")
        segmented = segment_leaf(blurred, otsu_offset)
        diseased = detect_diseased_area(image, segmented, lower_hue, upper_hue)
        results = visualize_results(image, segmented, diseased)
        st.image(results, caption="Processed Results (Original, Segmented, Heatmap)", width=600)

    # Health Analysis Tab
    with tab3:
        st.header("Health Metrics")
        health_percentage, diseased_pixels, total_pixels = calculate_health_metrics(segmented, diseased)
        st.write(f"**Leaf Health Percentage:** {health_percentage:.2f}%")
        st.write(f"**Total Diseased Pixels:** {diseased_pixels}")
        st.write(f"**Total Leaf Pixels:** {total_pixels}")

        # Histogram visualization
        hist_values = cv2.calcHist([diseased], [0], None, [256], [0, 256])
        st.bar_chart(hist_values.flatten())

        # Severity Level
        st.subheader("Severity Level")
        if health_percentage > 80:
            st.success("✅ Healthy Leaf")
        elif 50 <= health_percentage <= 80:
            st.warning("⚠️ Moderate Damage")
        else:
            st.error("❌ Severe Damage")

    # Footer with credits
    st.write("---")
    st.write("🛠️ Project by **Abdul Rafay** & **Tanveer**")
