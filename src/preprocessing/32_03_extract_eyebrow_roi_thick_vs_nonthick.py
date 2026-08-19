import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


# =========================
# 1. Path settings
# =========================

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

ALIGNMENT_CSV = BASE_DIR / "outputs" / "features" / "alignment_features.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
ROI_DIR = OUTPUT_DIR / "eyebrow_roi"
QC_DIR = OUTPUT_DIR / "qc" / "eyebrow_roi_overlay"
FEATURE_DIR = OUTPUT_DIR / "features"

ROI_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = FEATURE_DIR / "eyebrow_roi_features.csv"


# =========================
# 2. Settings
# =========================

CANVAS_SIZE = 512

SAVE_QC_OVERLAY = True
MAX_QC_PER_SOURCE = 40

# ROI padding settings
X_PAD_RATIO = 0.45
Y_PAD_UP_RATIO = 1.8
Y_PAD_DOWN_RATIO = 1.2

MIN_X_PAD = 18
MIN_Y_PAD_UP = 22
MIN_Y_PAD_DOWN = 16

# 為了避免 ROI 吃到眼睛，bottom 盡量停在眼睛上緣附近
EYE_GAP_PIXELS = 5


# =========================
# 3. MediaPipe landmark indices
# =========================
# 這裡的 left/right 是「影像中左邊 / 右邊」
# image_left 通常是畫面左側眉毛
# image_right 通常是畫面右側眉毛

IMAGE_LEFT_EYEBROW_IDXS = [
    70, 63, 105, 66, 107,
    55, 65, 52, 53, 46
]

IMAGE_RIGHT_EYEBROW_IDXS = [
    336, 296, 334, 293, 300,
    285, 295, 282, 283, 276
]

IMAGE_LEFT_EYE_IDXS = [
    33, 133, 159, 145, 153, 154, 155, 173, 157, 158
]

IMAGE_RIGHT_EYE_IDXS = [
    362, 263, 386, 374, 380, 381, 382, 398, 384, 385
]


# =========================
# 4. Utility functions
# =========================

def make_safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = text.strip("_")
    return text[:150]


def imread_unicode(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(str(path))
    return success


def clip_int(value, low, high):
    return int(max(low, min(high, value)))


def compute_roi_box(landmarks, eyebrow_idxs, eye_idxs, image_shape):
    """
    Compute eyebrow ROI bounding box from aligned landmarks.

    Return:
        x1, y1, x2, y2
    """
    h, w = image_shape[:2]

    eyebrow_pts = landmarks[eyebrow_idxs, :2]
    eye_pts = landmarks[eye_idxs, :2]

    brow_x_min = float(np.min(eyebrow_pts[:, 0]))
    brow_x_max = float(np.max(eyebrow_pts[:, 0]))
    brow_y_min = float(np.min(eyebrow_pts[:, 1]))
    brow_y_max = float(np.max(eyebrow_pts[:, 1]))

    brow_w = brow_x_max - brow_x_min
    brow_h = brow_y_max - brow_y_min

    if brow_w < 1 or brow_h < 1:
        raise ValueError("invalid_eyebrow_landmark_bbox")

    x_pad = max(MIN_X_PAD, X_PAD_RATIO * brow_w)
    y_pad_up = max(MIN_Y_PAD_UP, Y_PAD_UP_RATIO * brow_h)
    y_pad_down = max(MIN_Y_PAD_DOWN, Y_PAD_DOWN_RATIO * brow_h)

    x1 = brow_x_min - x_pad
    x2 = brow_x_max + x_pad

    y1 = brow_y_min - y_pad_up
    y2_by_brow = brow_y_max + y_pad_down

    # 眼睛最上緣，用來避免 ROI 太往下吃到眼睛
    eye_top_y = float(np.min(eye_pts[:, 1]))
    y2_by_eye = eye_top_y - EYE_GAP_PIXELS

    # 主要使用 brow-based bottom，但如果它超過眼睛太多，就限制在眼睛上緣
    y2 = min(y2_by_brow, y2_by_eye)

    # fallback：如果切太扁，就放寬到 brow_y_max + y_pad_down
    if y2 <= y1 + 18:
        y2 = y2_by_brow

    x1 = clip_int(round(x1), 0, w - 1)
    y1 = clip_int(round(y1), 0, h - 1)
    x2 = clip_int(round(x2), 1, w)
    y2 = clip_int(round(y2), 1, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid_roi_box")

    return x1, y1, x2, y2


def crop_roi(image, box):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2].copy()


def draw_roi_overlay(image, left_box, right_box, landmarks):
    overlay = image.copy()

    # left ROI: green
    lx1, ly1, lx2, ly2 = left_box
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
    cv2.putText(
        overlay, "image_left_eyebrow",
        (lx1, max(15, ly1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    # right ROI: cyan
    rx1, ry1, rx2, ry2 = right_box
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
    cv2.putText(
        overlay, "image_right_eyebrow",
        (rx1, max(15, ry1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA
    )

    # draw eyebrow landmarks
    for idx in IMAGE_LEFT_EYEBROW_IDXS:
        x, y = landmarks[idx, :2]
        cv2.circle(overlay, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)

    for idx in IMAGE_RIGHT_EYEBROW_IDXS:
        x, y = landmarks[idx, :2]
        cv2.circle(overlay, (int(round(x)), int(round(y))), 2, (255, 255, 0), -1)

    return overlay


def is_success_value(x):
    return str(x).lower() in ["1", "true", "yes"]


# =========================
# 5. Load alignment results
# =========================

if not ALIGNMENT_CSV.exists():
    raise FileNotFoundError(f"Cannot find alignment csv: {ALIGNMENT_CSV}")

df = pd.read_csv(ALIGNMENT_CSV)
df = df.fillna("")

df = df[df["alignment_success"].apply(is_success_value)].copy()
df = df.reset_index(drop=True)

print("Images to extract eyebrow ROI:")
print(df.groupby(["source", "label"]).size())


# =========================
# 6. Run ROI extraction
# =========================

records = []

qc_counter = {
    "GMDB": 0,
    "UTKFace": 0
}

for i, row in tqdm(df.iterrows(), total=len(df)):
    source = row["source"]
    label = row["label"]
    filename_stem = Path(row["filename"]).stem

    aligned_image_path = Path(row["aligned_image_path"])
    aligned_landmark_path = Path(row["aligned_landmark_path"])

    safe_stem = make_safe_name(f"{i:06d}_{source}_{label}_{filename_stem}")

    source_roi_dir = ROI_DIR / source
    source_qc_dir = QC_DIR / source

    source_roi_dir.mkdir(parents=True, exist_ok=True)
    source_qc_dir.mkdir(parents=True, exist_ok=True)

    left_roi_path = source_roi_dir / f"{safe_stem}_left_eyebrow_roi.jpg"
    right_roi_path = source_roi_dir / f"{safe_stem}_right_eyebrow_roi.jpg"
    qc_overlay_path = source_qc_dir / f"{safe_stem}_eyebrow_roi_overlay.jpg"

    base_record = row.to_dict()
    base_record.update({
        "roi_success": 0,
        "roi_failure_reason": "",
        "left_roi_path": "",
        "right_roi_path": "",
        "roi_qc_overlay_path": "",

        "left_x1": "",
        "left_y1": "",
        "left_x2": "",
        "left_y2": "",
        "left_roi_width": "",
        "left_roi_height": "",

        "right_x1": "",
        "right_y1": "",
        "right_x2": "",
        "right_y2": "",
        "right_roi_width": "",
        "right_roi_height": "",
    })

    try:
        image = imread_unicode(aligned_image_path)

        if image is None:
            raise ValueError("failed_to_read_aligned_image")

        if not aligned_landmark_path.exists():
            raise ValueError("missing_landmark_file")

        landmarks = np.load(aligned_landmark_path)

        if landmarks.shape[0] < 468:
            raise ValueError("invalid_landmark_shape")

        left_box = compute_roi_box(
            landmarks,
            IMAGE_LEFT_EYEBROW_IDXS,
            IMAGE_LEFT_EYE_IDXS,
            image.shape
        )

        right_box = compute_roi_box(
            landmarks,
            IMAGE_RIGHT_EYEBROW_IDXS,
            IMAGE_RIGHT_EYE_IDXS,
            image.shape
        )

        left_roi = crop_roi(image, left_box)
        right_roi = crop_roi(image, right_box)

        if left_roi.size == 0 or right_roi.size == 0:
            raise ValueError("empty_roi")

        imwrite_unicode(left_roi_path, left_roi)
        imwrite_unicode(right_roi_path, right_roi)

        qc_saved = False
        if SAVE_QC_OVERLAY and qc_counter.get(source, 0) < MAX_QC_PER_SOURCE:
            overlay = draw_roi_overlay(image, left_box, right_box, landmarks)
            imwrite_unicode(qc_overlay_path, overlay)
            qc_counter[source] = qc_counter.get(source, 0) + 1
            qc_saved = True

        lx1, ly1, lx2, ly2 = left_box
        rx1, ry1, rx2, ry2 = right_box

        base_record.update({
            "roi_success": 1,
            "roi_failure_reason": "",

            "left_roi_path": str(left_roi_path),
            "right_roi_path": str(right_roi_path),
            "roi_qc_overlay_path": str(qc_overlay_path) if qc_saved else "",

            "left_x1": lx1,
            "left_y1": ly1,
            "left_x2": lx2,
            "left_y2": ly2,
            "left_roi_width": lx2 - lx1,
            "left_roi_height": ly2 - ly1,

            "right_x1": rx1,
            "right_y1": ry1,
            "right_x2": rx2,
            "right_y2": ry2,
            "right_roi_width": rx2 - rx1,
            "right_roi_height": ry2 - ry1,
        })

    except Exception as e:
        base_record["roi_success"] = 0
        base_record["roi_failure_reason"] = str(e)

    records.append(base_record)


records_df = pd.DataFrame(records)
records_df.to_csv(OUTPUT_CSV, index=False)

print("\nDone.")
print(f"Saved ROI features to: {OUTPUT_CSV}")

print("\nROI extraction summary:")
print(records_df.groupby(["source", "label"])["roi_success"].agg(["count", "sum"]))

failed_df = records_df[records_df["roi_success"] == 0]
print(f"\nFailed ROI cases: {len(failed_df)}")

if len(failed_df) > 0:
    fail_path = FEATURE_DIR / "eyebrow_roi_failed.csv"
    failed_df.to_csv(fail_path, index=False)
    print(f"Saved failed ROI cases to: {fail_path}")
    print("\nFailure reasons:")
    print(failed_df["roi_failure_reason"].value_counts())