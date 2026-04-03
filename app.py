import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates


# -----------------------------
# Geometry helpers
# -----------------------------
def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

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
    except Exception:
        pass

    _, bright = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = clear_border_connected(bright)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bright = cv2.erode(bright, k, iterations=2)

    comp = largest_non_border_component(bright, min_area_ratio=0.05)
    if comp is not None:
        comp = cv2.dilate(comp, k, iterations=2)
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k, iterations=2)
        masks.append(("bright", comp))

    edges = cv2.Canny(gray, 40, 140)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, k2, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k2, iterations=2)
    masks.append(("edges", edges))

    return masks


# -----------------------------
# Candidate scoring
# -----------------------------
def score_candidate(quad, img_shape, contour_area):
    h, w = img_shape[:2]
    rect = order_points(quad)
    tl, tr, br, bl = rect

    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

    if width < 100 or height < 100:
        return -1e9

    box_area = cv2.contourArea(rect.reshape(-1, 1, 2))
    if box_area <= 1:
        return -1e9

    area_ratio = box_area / (h * w)
    if area_ratio < 0.20 or area_ratio > 0.98:
        return -1e9

    aspect = max(width, height) / max(1.0, min(width, height))
    a4_ratio = 1.414
    aspect_score = max(0.0, 1.0 - abs(aspect - a4_ratio) / 0.8)

    center = rect.mean(axis=0)
    img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    center_dist = np.linalg.norm(center - img_center) / np.linalg.norm(img_center)
    center_score = max(0.0, 1.0 - center_dist)

    fill_ratio = float(np.clip(contour_area / box_area, 0.0, 1.2))

    margin = min(
        rect[:, 0].min(),
        rect[:, 1].min(),
        (w - 1) - rect[:, 0].max(),
        (h - 1) - rect[:, 1].max(),
    )
    margin_score = float(np.clip((margin + 20) / 120.0, 0.0, 1.0))

    score = (
        area_ratio * 95.0
        + aspect_score * 22.0
        + center_score * 8.0
        + margin_score * 4.0
        + fill_ratio * 40.0
    )

    return score


# -----------------------------
# Auto detection
# -----------------------------
def detect_document_auto(original):
    if original.shape[0] > 1400:
        resize_ratio = original.shape[0] / 1400.0
        image = cv2.resize(original, (int(original.shape[1] / resize_ratio), 1400))
    else:
        resize_ratio = 1.0
        image = original.copy()

    masks = build_candidate_masks(image)

    best_quad = None
    best_score = -1e9
    best_source = None
    best_fill_ratio = 1.0

    img_area = image.shape[0] * image.shape[1]

    for source_name, candidate_mask in masks:
        contours = cv2.findContours(
            candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contours[0] if len(contours) == 2 else contours[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

        for c in contours:
            contour_area = cv2.contourArea(c)
            if contour_area < 0.05 * img_area:
                continue

            quad = contour_to_quad(c)
            score = score_candidate(quad, image.shape, contour_area)

            if score > best_score:
                box_area = cv2.contourArea(order_points(quad).reshape(-1, 1, 2))
                fill_ratio = float(np.clip(contour_area / max(box_area, 1.0), 0.0, 1.2))

                best_score = score
                best_quad = quad
                best_source = source_name
                best_fill_ratio = min(fill_ratio, 1.0)

    if best_quad is None:
        raise ValueError("لم أتمكن من اكتشاف الورقة بشكل صحيح.")

    best_quad = best_quad * resize_ratio

    expansion = 1.0 + 0.55 * (1.0 - best_fill_ratio)
    if best_source == "edges":
        expansion += 0.02

    expansion = float(np.clip(expansion, 1.00, 1.18))
    best_quad = expand_quad(best_quad, expansion, original.shape)

    warped = four_point_transform(original, best_quad)
    warped = trim_black_frame(warped)
    return warped


# -----------------------------
# Manual mode helpers
# -----------------------------
def make_preview_for_clicks(image, max_width=900, max_height=1200):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb, scale


def draw_points_on_preview(preview_rgb, points_preview, radius=8):
    canvas = preview_rgb.copy()

    for idx, (x, y) in enumerate(points_preview):
        cv2.circle(canvas, (int(x), int(y)), radius, (255, 0, 0), -1)
        cv2.putText(
            canvas,
            str(idx + 1),
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )
    return canvas


def detect_document_manual(original, points_original):
    pts = np.array(points_original, dtype=np.float32)
    warped = four_point_transform(original, pts)
    warped = trim_black_frame(warped)
    return warped


# -----------------------------
# Post-processing
# -----------------------------
def trim_black_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 6, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]


def decode_uploaded_image(file_bytes):
    file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("تعذر قراءة الصورة المرفوعة.")
    return img


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="A4 Document Scanner", layout="centered")

st.title("A4 Document Scanner")
st.write("ارفع صورة ورقة A4. يمكنك استخدام الوضع التلقائي أو تحديد الزوايا يدويًا بالنقر 4 مرات.")

mode = st.radio("Mode", ["Auto", "Manual"], horizontal=True)

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if "manual_points_preview" not in st.session_state:
    st.session_state.manual_points_preview = []

if "manual_points_original" not in st.session_state:
    st.session_state.manual_points_original = []

if "last_click" not in st.session_state:
    st.session_state.last_click = None

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

if uploaded_file is not None:
    if st.session_state.last_uploaded_name != uploaded_file.name:
        st.session_state.manual_points_preview = []
        st.session_state.manual_points_original = []
        st.session_state.last_click = None
        st.session_state.last_uploaded_name = uploaded_file.name

    file_bytes = uploaded_file.read()
    original = decode_uploaded_image(file_bytes)

    if mode == "Auto":
        try:
            result = detect_document_auto(original)
            st.image(
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                caption="Final Result",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.info("انقر بالترتيب: أعلى يسار، أعلى يمين، أسفل يمين، أسفل يسار.")

        preview_rgb, preview_scale = make_preview_for_clicks(original, max_width=900, max_height=1200)
        preview_with_points = draw_points_on_preview(
            preview_rgb,
            st.session_state.manual_points_preview
        )

        col1, col2 = st.columns([1.3, 1])

        with col1:
            clicked = streamlit_image_coordinates(
                preview_with_points,
                key="manual_click_image"
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Reset points"):
                    st.session_state.manual_points_preview = []
                    st.session_state.manual_points_original = []
                    st.session_state.last_click = None
                    st.rerun()

            with c2:
                if st.button("Undo last point"):
                    if st.session_state.manual_points_preview:
                        st.session_state.manual_points_preview.pop()
                    if st.session_state.manual_points_original:
                        st.session_state.manual_points_original.pop()
                    st.session_state.last_click = None
                    st.rerun()

            if clicked is not None:
                current_click = (clicked["x"], clicked["y"])

                if st.session_state.last_click != current_click:
                    if len(st.session_state.manual_points_preview) < 4:
                        st.session_state.manual_points_preview.append(current_click)

                        ox = int(round(clicked["x"] / preview_scale))
                        oy = int(round(clicked["y"] / preview_scale))

                        ox = max(0, min(ox, original.shape[1] - 1))
                        oy = max(0, min(oy, original.shape[0] - 1))

                        st.session_state.manual_points_original.append((ox, oy))

                    st.session_state.last_click = current_click
                    st.rerun()

        with col2:
            st.write(f"Points selected: {len(st.session_state.manual_points_original)}/4")
            for i, p in enumerate(st.session_state.manual_points_original, start=1):
                st.write(f"{i}: x={p[0]}, y={p[1]}")

            if len(st.session_state.manual_points_original) == 4:
                try:
                    result = detect_document_manual(original, st.session_state.manual_points_original)
                    st.image(
                        cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                        caption="Manual Result",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
