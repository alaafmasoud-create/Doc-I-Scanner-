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
    max_width = max(int(width_a), int(width_b), 1)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b), 1)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def polygon_area(pts):
    pts = pts.reshape((-1, 1, 2)).astype(np.float32)
    return cv2.contourArea(pts)


def contour_to_quad(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def expand_quad(pts, img_shape, scale=1.04):
    h, w = img_shape[:2]
    center = np.mean(pts, axis=0)
    expanded = center + (pts - center) * scale

    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)

    return expanded.astype(np.float32)


def score_quad(pts, img_shape):
    h, w = img_shape[:2]
    img_area = h * w

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

    if min(width, height) < 80:
        return -1

    area = polygon_area(rect)
    area_ratio = area / img_area

    if area_ratio < 0.20:
        return -1

    aspect = max(width, height) / max(1.0, min(width, height))
    if aspect > 2.2:
        return -1

    # ورق A4 تقريبًا
    paper_ratio = 1.414
    ratio_bonus = 1.0 - min(abs(aspect - paper_ratio), 1.0)

    img_center = np.array([w / 2, h / 2], dtype=np.float32)
    quad_center = np.mean(rect, axis=0)
    center_dist = np.linalg.norm(quad_center - img_center) / np.linalg.norm(img_center)
    center_bonus = 1.0 - min(center_dist, 1.0)

    score = (area_ratio * 100.0) + (ratio_bonus * 15.0) + (center_bonus * 5.0)
    return score


def get_candidate_quads_from_contours(contours, img_shape, max_count=15):
    h, w = img_shape[:2]
    img_area = h * w

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    quads = []

    for c in contours[:max_count]:
        area = cv2.contourArea(c)
        if area < 0.05 * img_area:
            continue

        quad = contour_to_quad(c)
        quads.append(quad)

    return quads


def detect_document(image):
    original = image.copy()

    # تصغير للصورة لتسريع المعالجة مع حفظ النسبة لإرجاع الإحداثيات
    ratio = 1.0
    max_h = 1400
    if image.shape[0] > max_h:
        ratio = image.shape[0] / max_h
        new_w = int(image.shape[1] / ratio)
        image = cv2.resize(image, (new_w, max_h))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    candidate_quads = []

    # 1) أفضل طريقة لهذه الحالة: عزل الورقة الفاتحة عن الخلفية
    _, bright_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
    bright_mask = cv2.erode(bright_mask, kernel, iterations=1)

    contours_bright = cv2.findContours(
        bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_bright = contours_bright[0] if len(contours_bright) == 2 else contours_bright[1]
    candidate_quads.extend(get_candidate_quads_from_contours(contours_bright, image.shape))

    # 2) طريقة احتياطية بالحواف
    edges = cv2.Canny(blurred, 30, 120)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2, iterations=2)
    edges = cv2.dilate(edges, kernel2, iterations=1)

    contours_edges = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_edges = contours_edges[0] if len(contours_edges) == 2 else contours_edges[1]
    candidate_quads.extend(get_candidate_quads_from_contours(contours_edges, image.shape))

    if not candidate_quads:
        raise ValueError("لم أتمكن من اكتشاف حدود المستند كاملًا.")

    best_quad = None
    best_score = -1

    for quad in candidate_quads:
        score = score_quad(quad, image.shape)
        if score > best_score:
            best_score = score
            best_quad = quad

    if best_quad is None:
        raise ValueError("لم أتمكن من تحديد المستند كاملًا.")

    # نوسّع الحدود قليلًا حتى لا تُقص الترويسة أو آخر الصفحة
    best_quad = expand_quad(best_quad, image.shape, scale=1.02)

    # إعادة الإحداثيات لحجم الصورة الأصلي
    best_quad = best_quad * ratio

    warped = four_point_transform(original, best_quad)
    return warped


def smooth_signal(arr, k=21):
    if k < 3:
        return arr
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(arr, kernel, mode="same")


def light_trim(image, threshold=210, pad=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # نستخدم percentile بدل المتوسط حتى لا تؤثر الكتابة داخل الصفحة
    row_signal = smooth_signal(np.percentile(gray, 80, axis=1), 21)
    col_signal = smooth_signal(np.percentile(gray, 80, axis=0), 21)

    top = 0
    while top < len(row_signal) - 1 and row_signal[top] < threshold:
        top += 1

    bottom = len(row_signal) - 1
    while bottom > 0 and row_signal[bottom] < threshold:
        bottom -= 1

    left = 0
    while left < len(col_signal) - 1 and col_signal[left] < threshold:
        left += 1

    right = len(col_signal) - 1
    while right > 0 and col_signal[right] < threshold:
        right -= 1

    top = max(top - pad, 0)
    left = max(left - pad, 0)
    bottom = min(bottom + pad, image.shape[0] - 1)
    right = min(right + pad, image.shape[1] - 1)

    if bottom <= top or right <= left:
        return image

    return image[top:bottom + 1, left:right + 1]

def process_document(file_bytes):
    file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("تعذر قراءة الصورة المرفوعة.")

    warped = detect_document(img)
    warped = light_trim(warped)

    return warped


st.set_page_config(page_title="Smart Document Scanner", layout="centered")

st.title("Smart Document Scanner")
st.write("ارفع صورة المستند وسيظهر لك المستند كاملًا بشكل مستقيم.")

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
