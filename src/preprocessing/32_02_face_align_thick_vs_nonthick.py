import os
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from tqdm import tqdm


# =========================
# 1. Path settings
# =========================

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

MANIFEST_PATH = BASE_DIR / "outputs" / "features" / "image_manifest.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
ALIGNED_DIR = OUTPUT_DIR / "aligned"
LANDMARK_DIR = OUTPUT_DIR / "landmarks"
QC_DIR = OUTPUT_DIR / "qc" / "alignment_overlay"
FEATURE_DIR = OUTPUT_DIR / "features"

ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
LANDMARK_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = FEATURE_DIR / "alignment_features.csv"
PARTIAL_CSV = FEATURE_DIR / "alignment_features.partial.csv"


# =========================
# 2. Processing settings
# =========================

# 先測試用：GMDB 全部跑，UTKFace 先抽 1000 張
# 正式跑全 UTKFace 時，把 MAX_UTKFACE 改成 None
MAX_GMDB = None
MAX_UTKFACE = 1000

RANDOM_SEED = 42

CANVAS_SIZE = 512

# 對齊後，左眼與右眼放在固定位置
TARGET_LEFT_EYE = np.array([0.35 * CANVAS_SIZE, 0.40 * CANVAS_SIZE], dtype=np.float32)
TARGET_RIGHT_EYE = np.array([0.65 * CANVAS_SIZE, 0.40 * CANVAS_SIZE], dtype=np.float32)

MIN_DETECTION_CONFIDENCE = 0.5

SAVE_QC_OVERLAY = True
MAX_QC_PER_SOURCE = 30

SAVE_EVERY_N = 200


# =========================
# 3. MediaPipe landmark indices
# =========================

# 這裡的 left/right 是「影像中左邊的眼睛」與「影像中右邊的眼睛」
# MediaPipe FaceMesh 常用眼睛點
IMAGE_LEFT_EYE_IDXS = [33, 133, 159, 145]
IMAGE_RIGHT_EYE_IDXS = [362, 263, 386, 374]


# =========================
# 4. Utility functions
# =========================

def make_safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = text.strip("_")
    return text[:120]


def imread_unicode(path: Path):
    """
    Robust image reader for paths that may contain non-ASCII characters.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: Path, img):
    """
    Robust image writer for paths that may contain non-ASCII characters.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(str(path))
    return success


def detect_landmarks_bgr(image_bgr, face_mesh):
    """
    Detect MediaPipe FaceMesh landmarks.
    Return:
        landmarks: numpy array with shape [N, 3], columns are x, y, z
    """
    h, w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        return None

    face_landmarks = result.multi_face_landmarks[0]

    points = []
    for lm in face_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        z = lm.z * w
        points.append([x, y, z])

    return np.array(points, dtype=np.float32)


def compute_eye_centers(landmarks):
    left_eye = landmarks[IMAGE_LEFT_EYE_IDXS, :2].mean(axis=0)
    right_eye = landmarks[IMAGE_RIGHT_EYE_IDXS, :2].mean(axis=0)
    return left_eye, right_eye


def build_alignment_matrix(src_left_eye, src_right_eye):
    """
    Build similarity transform from original eye points to target eye points.
    """
    src_left_eye = np.array(src_left_eye, dtype=np.float32)
    src_right_eye = np.array(src_right_eye, dtype=np.float32)

    src_mid = (src_left_eye + src_right_eye) / 2.0
    dst_mid = (TARGET_LEFT_EYE + TARGET_RIGHT_EYE) / 2.0

    src_vec = src_right_eye - src_left_eye
    dst_vec = TARGET_RIGHT_EYE - TARGET_LEFT_EYE

    src_dist = np.linalg.norm(src_vec)
    dst_dist = np.linalg.norm(dst_vec)

    if src_dist < 1e-6:
        raise ValueError("eye distance is too small")

    src_angle = math.atan2(src_vec[1], src_vec[0])
    dst_angle = math.atan2(dst_vec[1], dst_vec[0])

    angle = dst_angle - src_angle
    scale = dst_dist / src_dist

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    R = scale * np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ], dtype=np.float32)

    t = dst_mid - R @ src_mid

    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = R
    M[:, 2] = t

    return M, scale, src_dist, dst_dist, math.degrees(angle)


def transform_landmarks(landmarks, M, scale):
    """
    Apply affine transform to landmark x,y.
    """
    xy = landmarks[:, :2]
    ones = np.ones((xy.shape[0], 1), dtype=np.float32)
    xy_h = np.hstack([xy, ones])

    xy_new = xy_h @ M.T

    new_landmarks = landmarks.copy()
    new_landmarks[:, :2] = xy_new
    new_landmarks[:, 2] = landmarks[:, 2] * scale

    return new_landmarks


def draw_alignment_overlay(aligned_img, aligned_landmarks):
    """
    Draw eye centers and eye line for QC.
    """
    overlay = aligned_img.copy()

    left_eye, right_eye = compute_eye_centers(aligned_landmarks)
    left_eye_int = tuple(np.round(left_eye).astype(int))
    right_eye_int = tuple(np.round(right_eye).astype(int))

    cv2.circle(overlay, left_eye_int, 5, (0, 255, 0), -1)
    cv2.circle(overlay, right_eye_int, 5, (0, 255, 0), -1)
    cv2.line(overlay, left_eye_int, right_eye_int, (255, 0, 0), 2)

    cv2.putText(
        overlay,
        "Aligned by eye centers",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    return overlay


def is_true_value(x):
    return str(x).lower() in ["true", "1", "yes"]


# =========================
# 5. Load manifest
# =========================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Cannot find manifest: {MANIFEST_PATH}")

manifest = pd.read_csv(MANIFEST_PATH)
manifest = manifest.fillna("")

# 只處理 readable image
manifest = manifest[manifest["image_readable"].apply(is_true_value)].copy()

gmdb_df = manifest[manifest["source"] == "GMDB"].copy()
utk_df = manifest[manifest["source"] == "UTKFace"].copy()

if MAX_GMDB is not None and len(gmdb_df) > MAX_GMDB:
    gmdb_df = gmdb_df.sample(n=MAX_GMDB, random_state=RANDOM_SEED)

if MAX_UTKFACE is not None and len(utk_df) > MAX_UTKFACE:
    utk_df = utk_df.sample(n=MAX_UTKFACE, random_state=RANDOM_SEED)

work_df = pd.concat([gmdb_df, utk_df], axis=0)
work_df = work_df.sort_values(["source", "filename"]).reset_index(drop=True)

print("Images to process:")
print(work_df.groupby(["source", "label"]).size())


# =========================
# 6. Run alignment
# =========================

records = []
qc_counter = {
    "GMDB": 0,
    "UTKFace": 0
}

try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE
) as face_mesh:

    for i, row in tqdm(work_df.iterrows(), total=len(work_df)):
        image_path = Path(row["image_path"])
        source = row["source"]
        label = row["label"]

        original_filename = Path(row["filename"]).stem
        safe_stem = make_safe_name(f"{i:06d}_{source}_{label}_{original_filename}")

        source_aligned_dir = ALIGNED_DIR / source
        source_landmark_dir = LANDMARK_DIR / source
        source_qc_dir = QC_DIR / source

        source_aligned_dir.mkdir(parents=True, exist_ok=True)
        source_landmark_dir.mkdir(parents=True, exist_ok=True)
        source_qc_dir.mkdir(parents=True, exist_ok=True)

        aligned_path = source_aligned_dir / f"{safe_stem}.jpg"
        landmark_path = source_landmark_dir / f"{safe_stem}.npy"
        qc_path = source_qc_dir / f"{safe_stem}_overlay.jpg"

        record = row.to_dict()
        record.update({
            "alignment_success": 0,
            "failure_reason": "",
            "aligned_image_path": "",
            "aligned_landmark_path": "",
            "qc_overlay_path": "",
            "original_width": "",
            "original_height": "",
            "original_eye_distance": "",
            "target_eye_distance": "",
            "alignment_scale": "",
            "alignment_rotation_degree": "",
            "aligned_eye_distance": "",
        })

        try:
            image_bgr = imread_unicode(image_path)

            if image_bgr is None:
                raise ValueError("failed_to_read_image")

            h, w = image_bgr.shape[:2]
            record["original_width"] = w
            record["original_height"] = h

            landmarks = detect_landmarks_bgr(image_bgr, face_mesh)

            if landmarks is None:
                raise ValueError("no_face_detected")

            left_eye, right_eye = compute_eye_centers(landmarks)

            M, scale, original_eye_distance, target_eye_distance, rotation_degree = build_alignment_matrix(
                left_eye,
                right_eye
            )

            aligned_img = cv2.warpAffine(
                image_bgr,
                M,
                (CANVAS_SIZE, CANVAS_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            aligned_landmarks = transform_landmarks(landmarks, M, scale)

            aligned_left_eye, aligned_right_eye = compute_eye_centers(aligned_landmarks)
            aligned_eye_distance = float(np.linalg.norm(aligned_right_eye - aligned_left_eye))

            imwrite_unicode(aligned_path, aligned_img)
            np.save(landmark_path, aligned_landmarks)

            qc_saved = False
            if SAVE_QC_OVERLAY and qc_counter.get(source, 0) < MAX_QC_PER_SOURCE:
                overlay = draw_alignment_overlay(aligned_img, aligned_landmarks)
                imwrite_unicode(qc_path, overlay)
                qc_counter[source] = qc_counter.get(source, 0) + 1
                qc_saved = True

            record.update({
                "alignment_success": 1,
                "failure_reason": "",
                "aligned_image_path": str(aligned_path),
                "aligned_landmark_path": str(landmark_path),
                "qc_overlay_path": str(qc_path) if qc_saved else "",
                "original_eye_distance": float(original_eye_distance),
                "target_eye_distance": float(target_eye_distance),
                "alignment_scale": float(scale),
                "alignment_rotation_degree": float(rotation_degree),
                "aligned_eye_distance": float(aligned_eye_distance),
            })

        except Exception as e:
            record["alignment_success"] = 0
            record["failure_reason"] = str(e)

        records.append(record)

        if len(records) % SAVE_EVERY_N == 0:
            pd.DataFrame(records).to_csv(PARTIAL_CSV, index=False)

records_df = pd.DataFrame(records)
records_df.to_csv(OUTPUT_CSV, index=False)

print("\nDone.")
print(f"Saved alignment features to: {OUTPUT_CSV}")
print(f"Saved partial backup to: {PARTIAL_CSV}")

print("\nAlignment summary:")
print(records_df.groupby(["source", "label"])["alignment_success"].agg(["count", "sum"]))

failed_df = records_df[records_df["alignment_success"] == 0]
print(f"\nFailed images: {len(failed_df)}")

if len(failed_df) > 0:
    fail_path = FEATURE_DIR / "alignment_failed.csv"
    failed_df.to_csv(fail_path, index=False)
    print(f"Saved failed cases to: {fail_path}")
    print("\nFailure reasons:")
    print(failed_df["failure_reason"].value_counts())