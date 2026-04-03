import streamlit as st
import cv2
import numpy as np
from scanner import scan_document_from_array

st.set_page_config(page_title="Smart Document Scanner", layout="wide")

st.title("📄 Smart Document Scanner")
st.write("Upload a document photo to automatically straighten it and improve readability.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    warped, scanned, error = scan_document_from_array(img)

    if error:
        st.error(error)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)

        with col2:
            st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Straightened Document", use_container_width=True)

        with col3:
            st.image(scanned, caption="Scanned Output", use_container_width=True)

        # زر تنزيل
        success, buffer = cv2.imencode(".png", scanned)
        if success:
            st.download_button(
                label="Download Scanned Image",
                data=buffer.tobytes(),
                file_name="scanned_document.png",
                mime="image/png"
            )
