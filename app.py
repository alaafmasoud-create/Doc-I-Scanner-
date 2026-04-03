import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Smart Doc Scanner",
    page_icon="📄",
    layout="wide"
)

# =========================
# Load custom CSS
# =========================
def load_css():
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

# =========================
# Helper functions
# =========================
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

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth < 1 or maxHeight < 1:
        return image

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_document_corners(image):
    """Returns 4 points if detected, otherwise None."""
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2) * ratio
            return pts.astype(np.float32)

    return None

def enhance_scanned_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return scanned

def draw_points_on_image(image_rgb, points):
    img = image_rgb.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 10, (255, 0, 0), -1)
        cv2.putText(
            img,
            str(i + 1),
            (int(x) + 12, int(y) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )
    return img

def pil_bytes_from_array(image_array, mode="PNG"):
    if len(image_array.shape) == 2:
        pil_img = Image.fromarray(image_array)
    else:
        pil_img = Image.fromarray(image_array)

    buf = io.BytesIO()
    pil_img.save(buf, format=mode)
    return buf.getvalue()

# =========================
# Session state
# =========================
if "manual_points" not in st.session_state:
    st.session_state.manual_points = []

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# =========================
# Header
# =========================
st.markdown("""
<div class="hero-card">
    <div class="hero-badge">SMART DOC SCANNER</div>
    <h1>Scan and crop A4 documents easily</h1>
    <p>
        Upload an image of an A4 document. Use the default automatic mode,
        or manually adjust the corners for a more accurate crop.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# Upload + options
# =========================
left, right = st.columns([1.2, 1])

with left:
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

with right:
    mode = st.radio(
        "Select Mode",
        ["Automatic", "Manual"],
        horizontal=True
    )

if uploaded_file is not None:
    if st.session_state.last_uploaded_name != uploaded_file.name:
        st.session_state.manual_points = []
        st.session_state.last_uploaded_name = uploaded_file.name

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Unable to read the uploaded image.")
        st.stop()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.markdown('<div class="section-title">Document Workspace</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    final_result = None

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Uploaded Image")

        if mode == "Automatic":
            st.image(image_rgb, use_container_width=True)

        else:
            st.markdown("#### Click on the 4 corners")

            clicked = st.image(
                image_rgb,
                use_container_width=True
            )

            # ملاحظة:
            # Streamlit الأساسي لا يدعم التقاط الإحداثيات من st.image مباشرة.
            # إذا كنت تستخدم streamlit-image-coordinates أو مكوّن مشابه،
            # استبدل هذا الجزء بمنطقك الحالي الذي يعيد click_x و click_y.
            #
            # الكود التالي يحافظ على البنية فقط دون إظهار "Points selected:"
            st.info("Manual corner selection is enabled. Click handling should be connected to your existing coordinate component.")

            # إذا عندك مكوّن جاهز للإحداثيات، استخدمه هنا مثل:
            #
            # from streamlit_image_coordinates import streamlit_image_coordinates
            # value = streamlit_image_coordinates(image_rgb, key="manual_clicks")
            # if value is not None:
            #     point = [value["x"], value["y"]]
            #     if len(st.session_state.manual_points) < 4:
            #         if point not in st.session_state.manual_points:
            #             st.session_state.manual_points.append(point)

            display_image = image_rgb.copy()
            if len(st.session_state.manual_points) > 0:
                display_image = draw_points_on_image(display_image, st.session_state.manual_points)

            st.image(display_image, use_container_width=True)

            # لا نعرض Points selected إطلاقًا

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Reset Points"):
                    st.session_state.manual_points = []
                    st.rerun()

            with btn_col2:
                if len(st.session_state.manual_points) == 4:
                    pts = np.array(st.session_state.manual_points, dtype=np.float32)
                    warped_bgr = four_point_transform(image_bgr, pts)
                    final_result = enhance_scanned_image(warped_bgr)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Final Result")

        if mode == "Automatic":
            auto_pts = detect_document_corners(image_bgr)

            if auto_pts is not None:
                warped_bgr = four_point_transform(image_bgr, auto_pts)
                final_result = enhance_scanned_image(warped_bgr)
                st.image(final_result, use_container_width=True, clamp=True)

                download_bytes = pil_bytes_from_array(final_result, mode="PNG")
                st.download_button(
                    label="Download Final Result",
                    data=download_bytes,
                    file_name="final_result.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.warning("No document corners were detected automatically.")

        else:
            if final_result is not None:
                st.image(final_result, use_container_width=True, clamp=True)

                download_bytes = pil_bytes_from_array(final_result, mode="PNG")
                st.download_button(
                    label="Download Final Result",
                    data=download_bytes,
                    file_name="final_result.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                if len(st.session_state.manual_points) < 4:
                    st.info("Select 4 corner points to generate the final cropped document.")
                else:
                    st.warning("Unable to generate the final result.")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="empty-state">Upload an image to begin scanning.</div>', unsafe_allow_html=True)
