import streamlit as st
import cv2
import numpy as np


def process_receipt(file_bytes):
    # Convert uploaded file to OpenCV image
    file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode uploaded image.")

    hh, ww = img.shape[:2]

    # Edge detection
    canny = cv2.Canny(img, 50, 200)

    # Find contours
    contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Filter small contours
    cimg = np.zeros_like(canny)
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area > 20:
            cv2.drawContours(cimg, [cntr], 0, 255, 1)

    # Find points for hull
    points = np.column_stack(np.where(cimg.transpose() > 0))
    if len(points) == 0:
        raise ValueError("No valid contour points found. Try another image.")

    hull = cv2.convexHull(points)

    # Hull image
    himg = img.copy()
    cv2.polylines(himg, [hull], True, (0, 0, 255), 1)

    # Mask
    mask = np.zeros_like(cimg, dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)

    # Apply mask
    mimg = cv2.bitwise_and(img, img, mask=mask)

    # Rotated rectangle
    rotrect = cv2.minAreaRect(hull)
    (center), (width, height), angle = rotrect
    box = cv2.boxPoints(rotrect)
    boxpts = np.intp(box)

    # Rectangle image
    rimg = img.copy()
    cv2.drawContours(rimg, [boxpts], 0, (0, 0, 255), 1)

    # Angle correction
    if angle < -45:
        angle = -(90 + angle)
    else:
        if width > height:
            angle = -(90 + angle)
        else:
            angle = -angle

    neg_angle = -angle

    # Rotate
    M = cv2.getRotationMatrix2D(center, neg_angle, scale=1.0)
    result = cv2.warpAffine(
        mimg,
        M,
        (ww, hh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return {
        "original": img,
        "edges": canny,
        "filtered_edges": cimg,
        "hull": himg,
        "mask": mask,
        "rotated_rect": rimg,
        "result": result,
        "angle": neg_angle,
    }


st.set_page_config(page_title="Smart Receipt Rectifier", layout="wide")

st.title("Smart Receipt Rectifier")
st.write("Upload an image of a receipt or document to detect and straighten it.")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        result = process_receipt(file_bytes)

        st.success(f"Processing completed. Unrotation angle: {result['angle']:.2f}°")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB), use_container_width=True)

            st.subheader("Edges")
            st.image(result["edges"], use_container_width=True, clamp=True)

            st.subheader("Filtered Edges")
            st.image(result["filtered_edges"], use_container_width=True, clamp=True)

        with col2:
            st.subheader("Convex Hull")
            st.image(cv2.cvtColor(result["hull"], cv2.COLOR_BGR2RGB), use_container_width=True)

            st.subheader("Rotated Rectangle")
            st.image(cv2.cvtColor(result["rotated_rect"], cv2.COLOR_BGR2RGB), use_container_width=True)

            st.subheader("Mask")
            st.image(result["mask"], use_container_width=True, clamp=True)

        st.subheader("Final Straightened Result")
        st.image(cv2.cvtColor(result["result"], cv2.COLOR_BGR2RGB), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
