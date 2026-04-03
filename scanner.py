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

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def enhance_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    scanned = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        8
    )

    return enhanced, scanned


def detect_document_contour(image):
    h, w = image.shape[:2]
    scale = 1200.0 / max(h, w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # جرّب طريقتين للحواف
    edged1 = cv2.Canny(blur, 50, 150)
    edged2 = cv2.Canny(blur, 30, 120)

    kernel = np.ones((5, 5), np.uint8)
    edged1 = cv2.dilate(edged1, kernel, iterations=2)
    edged1 = cv2.erode(edged1, kernel, iterations=1)

    edged2 = cv2.dilate(edged2, kernel, iterations=2)
    edged2 = cv2.erode(edged2, kernel, iterations=1)

    candidate_edges = [edged1, edged2]
    image_area = resized.shape[0] * resized.shape[1]

    for edged in candidate_edges:
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:20]:
            area = cv2.contourArea(contour)

            # نتجاهل الصغير جدًا
            if area < image_area * 0.15:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
                pts = pts / scale
                return pts

    return None


def rotate_if_needed(warped):
    h, w = warped.shape[:2]
    # لو خرجت الصفحة أفقية بشكل غير مرغوب، نخليها عمودية
    if w > h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def scan_document_from_array(image):
    original = image.copy()

    contour = detect_document_contour(original)

    if contour is not None:
        warped = four_point_transform(original, contour)
        warped = rotate_if_needed(warped)
        enhanced, scanned = enhance_document(warped)
        return {
            "mode": "detected",
            "original": original,
            "warped": warped,
            "enhanced": enhanced,
            "scanned": scanned,
            "message": None
        }

    # fallback: إذا لم يكتشف الورقة، حسّن الصورة كاملة بدل الفشل
    enhanced, scanned = enhance_document(original)
    return {
        "mode": "fallback",
        "original": original,
        "warped": None,
        "enhanced": enhanced,
        "scanned": scanned,
        "message": "Full document was not detected, so a full-image enhanced scan is shown instead."
    }
