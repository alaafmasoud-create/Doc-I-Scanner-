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
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def find_best_document_contour(contours, img_shape):
    img_h, img_w = img_shape[:2]
    img_area = img_h * img_w

    best_quad = None
    best_score = -1

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.15 * img_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w < 50 or h < 50:
            continue

        aspect = max(w, h) / max(1, min(w, h))

        # استبعاد الأشرطة الرفيعة مثل عنوان الصفحة
        if aspect > 2.2:
            continue

        # نفضّل الأشكال الرباعية الكبيرة
        score = area

        if score > best_score:
            best_score = score
            best_quad = approx

    return best_quad


def detect_document(image):
    original = image.copy()

    max_h = 1400
    ratio = 1.0
    if image.shape[0] > max_h:
        ratio = image.shape[0] / max_h
        new_w = int(image.shape[1] / ratio)
        image = cv2.resize(image, (new_w, max_h))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    candidate_contours = []

    # الطريقة 1: Canny + Morphology
    edged = cv2.Canny(blurred, 30, 120)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel1, iterations=2)
    edged = cv2.dilate(edged, kernel1, iterations=1)

    contours1 = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours1 = contours1[0] if len(contours1) == 2 else contours1[1]
    candidate_contours.extend(contours1)

    # الطريقة 2: Adaptive Threshold لعزل الورقة الفاتحة
    thr = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        15
    )

    # بما أن الورقة عادة فاتحة، نعكس حتى تصبح منطقة الورقة واضحة أكثر
    thr_inv = 255 - thr
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thr_inv = cv2.morphologyEx(thr_inv, cv2.MORPH_CLOSE, kernel2, iterations=2)
    thr_inv = cv2.dilate(thr_inv, kernel2, iterations=1)

    contours2 = cv2.findContours(thr_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = contours2[0] if len(contours2) == 2 else contours2[1]
    candidate_contours.extend(contours2)

    # نحاول إيجاد أفضل رباعي يمثل كامل الصفحة
    best_quad = find_best_document_contour(candidate_contours, image.shape)

    # fallback: إذا لم نجد رباعياً واضحًا، نأخذ أكبر contour خارجي منطقي
    if best_quad is None:
        if not candidate_contours:
            raise ValueError("لم أتمكن من اكتشاف المستند كاملًا.")

        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        large_contours = [c for c in candidate_contours if cv2.contourArea(c) > 0.12 * img_area]

        if not large_contours:
            raise ValueError("لم أتمكن من اكتشاف المستند كاملًا.")

        c = max(large_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        best_quad = np.array(box, dtype="float32").reshape(4, 1, 2)

    pts = best_quad.reshape(4, 2) * ratio
    warped = four_point_transform(original, pts)

    return warped


def trim_black_borders(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, thresh = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]


def process_document(file_bytes):
    file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("تعذر قراءة الصورة المرفوعة.")

    warped = detect_document(img)
    warped = trim_black_borders(warped)

    return warped


st.set_page_config(page_title="Smart Document Scanner", layout="centered")

st.title("Smart Document Scanner")
st.write("ارفع صورة المستند، وسيظهر لك المستند كاملًا بشكل مستقيم.")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        result = process_document(file_bytes)

        st.image(
            cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
            caption="Final Document",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error: {e}")
