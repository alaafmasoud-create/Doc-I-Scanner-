import streamlit as st
import cv2
import numpy as np

st.title("📄 Smart Document Scanner")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # تقليل الضجيج
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # نسخة سكانر أخف
    scanned = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    st.subheader("Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img, caption="Original Image", use_container_width=True)

    with col2:
        st.image(enhanced, caption="Enhanced Grayscale", use_container_width=True)

    with col3:
        st.image(scanned, caption="Scanned Version", use_container_width=True)
