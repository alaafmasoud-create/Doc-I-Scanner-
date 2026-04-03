import streamlit as st
import cv2
import numpy as np


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def detect_document(image):
    original = image.copy()
    ratio = image.shape[0] / 1000.0 if image.shape[0] > 1000 else 1.0

    if ratio != 1.0:
        new_width = int(image.shape[1] / ratio)
        new_height = int(image.shape[0] / ratio)
        image = cv2.resize(image, (new_width, new_height))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blurred, 50, 150)

    # Morphology to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)

    contours = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    screen_cnt = None

    for c in contours[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        area = cv2.contourArea(c)
        if len(approx) == 4 and area > 1000:
            screen_cnt = approx
            break

    # Fallback: use minAreaRect if no perfect 4-corner contour was found
    if screen_cnt is None:
        if not contours:
            raise ValueError("No document contour detected.")

        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        screen_cnt = np.array(box, dtype="float32").reshape(4, 1, 2)

    pts = screen_cnt.reshape(4, 2)

    # Restore coordinates to original size if resized
    pts = pts * ratio

    warped = four_point_transform(original, pts)

    return warped


def clean_document(doc):
    gray = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)

    # Improve scanner-like look
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    cleaned = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    return cleaned


def process_document(file_bytes):
    file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode uploaded image.")

    warped = detect_document(img)
    final = clean_document(warped)

    return final


st.set_page_config(page_title="Smart Document Scanner", layout="centered")

st.title("Smart Document Scanner")
st.write("Upload a document image and get a straight, cropped result.")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        result = process_document(file_bytes)

        st.image(result, caption="Final Result", use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
