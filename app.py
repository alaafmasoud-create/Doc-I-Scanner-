import streamlit as st
import cv2
import numpy as np
from scanner import scan_document
import tempfile

st.title("📄 Smart Document Scanner")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

    cv2.imwrite(temp_input.name, img)

    scan_document(temp_input.name, temp_output.name)

    st.image(img, caption="Original Image")
    st.image(temp_output.name, caption="Scanned Image")
