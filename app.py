import streamlit as st
import cv2
import numpy as np


# -----------------------------
# Geometry helpers
# -----------------------------
def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect

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
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def expand_quad(pts, scale, img_shape):
    pts = np.array(pts, dtype=np.float32)
    center = pts.mean(axis=0)
    expanded = center + (pts - center) * scale

    h, w = img_shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)

    return expanded.astype(np.float32)


def contour_to_quad(contour):
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)

    for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


# -----------------------------
# Mask helpers
# -----------------------------
def clear_border_connected(mask):
    cleaned = mask.copy()
    h, w = cleaned.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    for x in range(w):
        if cleaned[0, x] == 255:
            cv2.floodFill(cleaned, flood_mask, (x, 0), 0)
        if cleaned[h - 1, x] == 255:
            cv2.floodFill(cleaned, flood_mask, (x, h - 1), 0)

    for y in range(h):
        if cleaned[y, 0] == 255:
            cv2.floodFill(cleaned, flood_mask, (0, y), 0)
        if cleaned[y, w - 1] == 255:
            cv2.floodFill(cleaned, flood_mask, (w - 1, y), 0)

    return cleaned


def largest_non_border_component(binary_mask, min_area_ratio=0.05):
    h, w = binary_mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)

    best_idx = None
    best_area = 0
    min_area = min_area_ratio * h * w

    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]

        if area < min_area:
            continue

        if x <= 1 or y <= 1 or x + ww >= w - 1 or y + hh >= h - 1:
            continue

        if area > best_area:
            best_area = area
            best_idx = i

    if best_idx is None:
        return None

    comp = np.zeros_like(binary_mask)
    comp[labels == best_idx] = 255
    return comp


def build_candidate_masks(image):
    masks = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    h, w = gray.shape

    # GrabCut
    try:
        gc_mask = np.zeros((h, w), np.uint8)
        rect = (int(w * 0.06), int(h * 0.04), int(w * 0.88), int(h * 0.92))
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)

        grabcut = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        grabcut = cv2.morphologyEx(grabcut, cv2.MORPH_CLOSE, k, iterations=2)

        masks.append(("grabcut", grabcut))
    except:
        pass

    # Bright mask
    _, bright = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = clear_border_connected(bright)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bright = cv2.erode(bright, k, iterations=2)

    comp = largest_non_border_component(bright)
    if comp is not None:
        comp = cv2.dilate(comp, k, iterations=2)
        masks.append(("bright", comp))

    # Edges
    edges = cv2.Canny(gray, 40, 140)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, k2, iterations=2)
    masks.append(("edges", edges))

    return masks


# -----------------------------
# Detection
# -----------------------------
def detect_document(original):

    image = original.copy()
    masks = build_candidate_masks(image)

    best_quad = None
    best_area = 0

    for _, mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < best_area:
                continue

            quad = contour_to_quad(c)
            best_quad = quad
            best_area = area

    if best_quad is None:
        raise ValueError("لم يتم اكتشاف المستند")

    best_quad = expand_quad(best_quad, 1.08, original.shape)

    warped = four_point_transform(original, best_quad)

    return warped


def process_document(file_bytes):
    file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    return detect_document(img)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="A4 Scanner", layout="centered")

st.title("A4 Document Scanner")

uploaded_file = st.file_uploader("Upload image")

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    result = process_document(file_bytes)

    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
