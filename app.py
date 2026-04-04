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


@st.cache_data(show_spinner=False)
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

        cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)

        grabcut = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        grabcut = cv2.morphologyEx(grabcut, cv2.MORPH_CLOSE, k, iterations=2)
        masks.append(("grabcut", grabcut))
    except Exception:
        pass

    # Bright paper mask
    _, bright = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = clear_border_connected(bright)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bright = cv2.erode(bright, k, iterations=2)

    comp = largest_non_border_component(bright, min_area_ratio=0.05)
    if comp is not None:
        comp = cv2.dilate(comp, k, iterations=2)
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k, iterations=2)
        masks.append(("bright", comp))

    # Edges
    edges = cv2.Canny(gray, 40, 140)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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
@st.cache_data(show_spinner=False)
def detect_document_auto(original):
    max_detect_height = 1100

    if original.shape[0] > max_detect_height:
        resize_ratio = original.shape[0] / float(max_detect_height)
        image = cv2.resize(
            original,
            (int(original.shape[1] / resize_ratio), max_detect_height),
            interpolation=cv2.INTER_AREA
        )
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
        raise ValueError("No se pudo detectar el documento correctamente.")

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
@st.cache_data(show_spinner=False)
def make_preview_for_clicks(image, max_width=1000, max_height=1400):
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def decode_uploaded_image(file_bytes):
    file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo leer la imagen subida.")
    return img


@st.cache_data(show_spinner=False)
def image_to_download_bytes(image_bgr, filename="final_result.png"):
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        return None
    return buffer.tobytes()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Escáner de Documentos A4", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero-box {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 22px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 10px 30px rgba(15,23,42,0.08);
        margin-bottom: 1rem;
        backdrop-filter: blur(6px);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: .35rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #475569;
        line-height: 1.65;
        margin-bottom: 0;
    }
    .section-card {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 20px;
        padding: 1rem 1rem 0.8rem 1rem;
        box-shadow: 0 10px 24px rgba(15,23,42,0.05);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: .2rem;
    }
    .section-note {
        font-size: .95rem;
        color: #64748b;
        margin-bottom: .65rem;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 14px !important;
        border: none !important;
        min-height: 3rem;
        font-weight: 800 !important;
        font-size: 1rem !important;
        letter-spacing: 0.01em;
        color: #ffffff !important;
        box-shadow: 0 10px 24px rgba(15,23,42,0.14);
        transition: all .2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(15,23,42,0.18);
        filter: brightness(1.04);
    }
    div[data-testid="column"]:first-child .stButton > button {
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%) !important;
    }
    div[data-testid="column"]:nth-of-type(2) .stButton > button {
        background: linear-gradient(135deg, #9a3412 0%, #ea580c 100%) !important;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    }
    .stFileUploader, div[data-baseweb="select"], .stRadio {
        background: rgba(255,255,255,0.85);
        border-radius: 16px;
        padding: .4rem .55rem;
    }
    div[data-baseweb="select"] {
        border: 2px solid #7c3aed;
        box-shadow: 0 8px 22px rgba(124,58,237,0.14);
        background: linear-gradient(135deg, rgba(245,243,255,0.98) 0%, rgba(238,242,255,0.98) 100%);
    }
    div[data-baseweb="select"] > div {
        color: #312e81 !important;
        font-weight: 700 !important;
    }
    label[data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 1.02rem !important;
    }
    div[role="listbox"] ul {
        background: #ffffff !important;
    }
    div[role="option"] {
        color: #1e1b4b !important;
        font-weight: 700 !important;
    }
    .stRadio [role="radiogroup"] {
        gap: 0.75rem;
        padding-top: 0.2rem;
    }
    .stRadio [role="radiogroup"] label {
        background: linear-gradient(135deg, #ede9fe 0%, #dbeafe 100%);
        border: 1px solid #c4b5fd;
        border-radius: 14px;
        padding: 0.45rem 0.95rem;
        box-shadow: 0 4px 12px rgba(79,70,229,0.10);
    }
    .stRadio [role="radiogroup"] label p {
        color: #312e81;
        font-weight: 700;
    }
    .footer-signature {
        text-align: center;
        margin-top: 2.2rem;
        padding-top: 0.8rem;
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 500;
        opacity: 0.9;
        }
    .stFileUploader {
        border: 2px dashed #38bdf8;
        background: linear-gradient(135deg, rgba(224,242,254,0.9) 0%, rgba(240,249,255,0.96) 100%);
        box-shadow: 0 8px 22px rgba(14,165,233,0.10);
    }
    .stFileUploader label {
        color: #0f172a !important;
        font-weight: 700;
    }
    .stFileUploader section[data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
    }
    .stFileUploader section[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 22px rgba(37,99,235,0.22);
    }
    .stFileUploader section[data-testid="stFileUploaderDropzone"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(37,99,235,0.28);
    }
    div[data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.08);
    }
    .result-label {
        display: inline-block;
        padding: .32rem .65rem;
        border-radius: 999px;
        background: #e0ecff;
        color: #1d4ed8;
        font-size: .83rem;
        font-weight: 700;
        margin-bottom: .5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <div class="hero-title">📄 Escáner de Documentos A4</div>
    <p class="hero-subtitle">
        Sube una imagen de un documento A4. Usa el modo automático predeterminado o ajusta manualmente las esquinas para un recorte más preciso.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Configuración del escáner</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">Elige el modo y sube una o varias imágenes.</div>', unsafe_allow_html=True)

mode = st.radio("Modo", ["Automático", "Manual"], horizontal=True)

uploaded_files = st.file_uploader(
    "Subir imagen",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True
)

st.markdown('</div>', unsafe_allow_html=True)

if "manual_points_preview" not in st.session_state:
    st.session_state.manual_points_preview = []

if "manual_points_original" not in st.session_state:
    st.session_state.manual_points_original = []

if "last_click" not in st.session_state:
    st.session_state.last_click = None

if "last_uploaded_key" not in st.session_state:
    st.session_state.last_uploaded_key = None

if uploaded_files:
    if mode == "Automático":
        for file_index, uploaded_file in enumerate(uploaded_files):
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">RESULTADO AUTOMÁTICO</div>', unsafe_allow_html=True)
            st.subheader(f"{uploaded_file.name}")

            file_bytes = uploaded_file.getvalue()
            original = decode_uploaded_image(file_bytes)

            try:
                result = detect_document_auto(original)
                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="Resultado final",
                    use_container_width=True
                )

                download_bytes = image_to_download_bytes(result)
                if download_bytes is not None:
                    file_base = uploaded_file.name.rsplit(".", 1)[0]
                    st.download_button(
                        label="Descargar resultado final",
                        data=download_bytes,
                        file_name=f"{file_base}_final_result.png",
                        mime="image/png",
                        key=f"download_auto_{file_index}"
                    )

            except Exception as e:
                st.error(f"Error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">MODO MANUAL</div>', unsafe_allow_html=True)

        selected_file_name = st.selectbox(
            "Selecciona una imagen para el modo manual",
            [file.name for file in uploaded_files]
        )

        uploaded_file = next(
            file for file in uploaded_files if file.name == selected_file_name
        )

        upload_key = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.last_uploaded_key != upload_key:
            st.session_state.manual_points_preview = []
            st.session_state.manual_points_original = []
            st.session_state.last_click = None
            st.session_state.last_uploaded_key = upload_key

        file_bytes = uploaded_file.getvalue()
        original = decode_uploaded_image(file_bytes)

        preview_rgb, preview_scale = make_preview_for_clicks(
            original,
            max_width=1000,
            max_height=1400
        )

        preview_with_points = draw_points_on_preview(
            preview_rgb,
            st.session_state.manual_points_preview
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Restablecer puntos"):
                st.session_state.manual_points_preview = []
                st.session_state.manual_points_original = []
                st.session_state.last_click = None
                st.rerun()

        with col_btn2:
            if st.button("Deshacer el último punto"):
                if st.session_state.manual_points_preview:
                    st.session_state.manual_points_preview.pop()
                if st.session_state.manual_points_original:
                    st.session_state.manual_points_original.pop()
                st.session_state.last_click = None
                st.rerun()

        st.markdown('<div class="section-title" style="margin-top:.35rem; color:#1d4ed8;">Haz clic en las 4 esquinas</div>', unsafe_allow_html=True)

        clicked = streamlit_image_coordinates(
            preview_with_points,
            key="manual_click_image"
        )

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

        if len(st.session_state.manual_points_original) == 4:
            try:
                result = detect_document_manual(original, tuple(st.session_state.manual_points_original))
                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="Resultado manual",
                    use_container_width=True
                )

                download_bytes = image_to_download_bytes(result)
                if download_bytes is not None:
                    file_base = uploaded_file.name.rsplit(".", 1)[0]
                    st.download_button(
                        label="Descargar resultado final",
                        data=download_bytes,
                        file_name=f"{file_base}_final_result.png",
                        mime="image/png"
                    )

            except Exception as e:
                st.error(f"Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer-signature">By Alan Masoud</div>', unsafe_allow_html=True)
