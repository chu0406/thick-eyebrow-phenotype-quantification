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

ROI_CSV = BASE_DIR / "outputs" / "features" / "eyebrow_roi_features.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
MASK_DIR = OUTPUT_DIR / "eyebrow_mask_v2"
OVERLAY_DIR = OUTPUT_DIR / "qc" / "eyebrow_mask_overlay_v2"
DEBUG_DIR = OUTPUT_DIR / "qc" / "eyebrow_mask_debug_v2"
FEATURE_DIR = OUTPUT_DIR / "features"

MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = FEATURE_DIR / "eyebrow_mask_features_v2.csv"


# =========================
# 2. Settings
# =========================

SAVE_OVERLAY = True
SAVE_DEBUG = True

# 眉毛 tube 的膨脹程度
# 太大會吃到皮膚/頭髮，太小會切掉眉毛
TUBE_DILATE_X_RATIO = 0.10
TUBE_DILATE_Y_RATIO = 0.42
MIN_TUBE_DILATE_X = 5
MIN_TUBE_DILATE_Y = 7

# darker-than-skin threshold
MIN_DARKNESS_L = 6

# 若 mask 太少，會稍微放寬
RELAXED_MIN_DARKNESS_L = 3

# 面積限制，避免整片爆掉
MIN_MASK_AREA_RATIO = 0.003
MAX_MASK_AREA_RATIO = 0.32

# connected component
MIN_COMPONENT_AREA_RATIO = 0.0015
MAX_COMPONENT_AREA_RATIO = 0.25


# =========================
# 3. MediaPipe eyebrow landmark indices
# =========================

IMAGE_LEFT_EYEBROW_IDXS = [
    70, 63, 105, 66, 107,
    55, 65, 52, 53, 46
]

IMAGE_RIGHT_EYEBROW_IDXS = [
    336, 296, 334, 293, 300,
    285, 295, 282, 283, 276
]


# =========================
# 4. Utility functions
# =========================

def make_safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = text.strip("_")
    return text[:160]


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


def is_success_value(x):
    return str(x).lower() in ["1", "true", "yes"]


def make_odd(x):
    x = int(round(x))
    x = max(3, x)
    if x % 2 == 0:
        x += 1
    return x


def clip_int(value, low, high):
    return int(max(low, min(high, value)))


def get_side_eyebrow_indices(side: str):
    if side == "left":
        return IMAGE_LEFT_EYEBROW_IDXS
    elif side == "right":
        return IMAGE_RIGHT_EYEBROW_IDXS
    else:
        raise ValueError(f"unknown side: {side}")


def normalize_uint8(x):
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn) * 255.0
    return np.clip(y, 0, 255).astype(np.uint8)


def get_eyebrow_points_in_roi(landmarks, roi_box, side):
    x1, y1, x2, y2 = roi_box
    idxs = get_side_eyebrow_indices(side)

    pts = landmarks[idxs, :2].copy()
    pts[:, 0] -= x1
    pts[:, 1] -= y1

    return pts.astype(np.float32)


def build_landmark_tube_mask(roi_shape, eyebrow_pts):
    """
    建立一個只包住 eyebrow landmark 附近的 narrow tube。
    之後 threshold 只允許發生在這個 tube 裡。
    """
    h, w = roi_shape[:2]

    pts = eyebrow_pts.copy()

    # clip points
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    hull = cv2.convexHull(pts.astype(np.int32))

    core = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(core, hull, 255)

    brow_x_min = float(np.min(pts[:, 0]))
    brow_x_max = float(np.max(pts[:, 0]))
    brow_y_min = float(np.min(pts[:, 1]))
    brow_y_max = float(np.max(pts[:, 1]))

    brow_w = max(1.0, brow_x_max - brow_x_min)
    brow_h = max(1.0, brow_y_max - brow_y_min)

    dilate_x = max(MIN_TUBE_DILATE_X, int(round(brow_w * TUBE_DILATE_X_RATIO)))
    dilate_y = max(MIN_TUBE_DILATE_Y, int(round(brow_h * TUBE_DILATE_Y_RATIO)))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (make_odd(dilate_x), make_odd(dilate_y))
    )

    tube = cv2.dilate(core, kernel, iterations=1)

    return tube


def compute_skin_reference_mask(tube_mask):
    """
    取 tube 外圍一圈作為 skin reference。
    """
    h, w = tube_mask.shape[:2]

    kernel_outer = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (make_odd(w * 0.20), make_odd(h * 0.45))
    )

    outer = cv2.dilate(tube_mask, kernel_outer, iterations=1)

    ring = np.zeros_like(tube_mask, dtype=np.uint8)
    ring[(outer > 0) & (tube_mask == 0)] = 255

    # 避免 reference ring 太少，fallback 用整張 ROI 非 tube 區
    if np.sum(ring > 0) < 20:
        ring = np.zeros_like(tube_mask, dtype=np.uint8)
        ring[tube_mask == 0] = 255

    return ring


def compute_darkness_against_skin(roi_bgr, tube_mask):
    """
    用 Lab 的 L channel。
    眉毛通常比周圍皮膚暗，所以 darkness = skin_L_median - pixel_L。
    """
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # denoise
    L_blur = cv2.GaussianBlur(L, (3, 3), 0)

    skin_ref_mask = compute_skin_reference_mask(tube_mask)

    skin_pixels = L_blur[skin_ref_mask > 0]

    if len(skin_pixels) < 20:
        skin_pixels = L_blur[tube_mask == 0]

    if len(skin_pixels) < 20:
        skin_median = float(np.median(L_blur))
    else:
        skin_median = float(np.median(skin_pixels))

    darkness = skin_median - L_blur
    darkness = np.clip(darkness, 0, 255)

    return L_blur, darkness, skin_ref_mask, skin_median


def otsu_threshold_in_tube(darkness, tube_mask):
    values = darkness[tube_mask > 0]

    if len(values) < 20:
        return MIN_DARKNESS_L

    values_u8 = np.clip(values, 0, 255).astype(np.uint8)

    # 如果 tube 裡幾乎沒有對比，不要讓 Otsu 亂切
    if np.std(values_u8) < 2.0:
        return MIN_DARKNESS_L

    thr, _ = cv2.threshold(
        values_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thr = float(thr)

    # Otsu 有時會太低，至少要比皮膚暗一點
    thr = max(thr, MIN_DARKNESS_L)

    return thr


def clean_mask(mask, roi_shape):
    h, w = roi_shape[:2]

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (make_odd(w * 0.06), make_odd(h * 0.12))
    )

    out = mask.copy()
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)

    return out


def filter_components(mask, tube_mask):
    h, w = mask.shape[:2]
    roi_area = h * w

    min_area = max(4, int(roi_area * MIN_COMPONENT_AREA_RATIO))
    max_area = max(min_area + 1, int(roi_area * MAX_COMPONENT_AREA_RATIO))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    kept = np.zeros_like(mask, dtype=np.uint8)
    kept_count = 0

    for lab_id in range(1, num_labels):
        x, y, ww, hh, area = stats[lab_id]

        if area < min_area:
            continue
        if area > max_area:
            continue

        comp = labels == lab_id

        # component 必須大部分落在 eyebrow tube 裡
        overlap = np.sum(comp & (tube_mask > 0))
        overlap_ratio = overlap / max(1, area)

        if overlap_ratio < 0.65:
            continue

        kept[comp] = 255
        kept_count += 1

    return kept, kept_count


def generate_mask_v2(roi_bgr, landmarks, roi_box, side):
    h, w = roi_bgr.shape[:2]

    eyebrow_pts = get_eyebrow_points_in_roi(
        landmarks=landmarks,
        roi_box=roi_box,
        side=side
    )

    tube_mask = build_landmark_tube_mask(
        roi_shape=roi_bgr.shape,
        eyebrow_pts=eyebrow_pts
    )

    L_blur, darkness, skin_ref_mask, skin_median = compute_darkness_against_skin(
        roi_bgr,
        tube_mask
    )

    thr = otsu_threshold_in_tube(darkness, tube_mask)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[(darkness >= thr) & (tube_mask > 0)] = 255

    mask = clean_mask(mask, roi_bgr.shape)
    mask, comp_count = filter_components(mask, tube_mask)

    area_ratio = np.sum(mask > 0) / float(h * w)

    # 如果太少，放寬一次
    if area_ratio < MIN_MASK_AREA_RATIO:
        relaxed_thr = RELAXED_MIN_DARKNESS_L
        mask2 = np.zeros((h, w), dtype=np.uint8)
        mask2[(darkness >= relaxed_thr) & (tube_mask > 0)] = 255
        mask2 = clean_mask(mask2, roi_bgr.shape)
        mask2, comp_count2 = filter_components(mask2, tube_mask)

        area_ratio2 = np.sum(mask2 > 0) / float(h * w)

        if area_ratio2 > area_ratio:
            mask = mask2
            thr = relaxed_thr
            comp_count = comp_count2
            area_ratio = area_ratio2

    # 如果仍然爆掉，改成 tube 裡最暗的前 35%，但仍只限 tube
    if area_ratio > MAX_MASK_AREA_RATIO:
        values = darkness[tube_mask > 0]
        fallback_thr = np.percentile(values, 65)
        fallback_thr = max(fallback_thr, MIN_DARKNESS_L)

        mask3 = np.zeros((h, w), dtype=np.uint8)
        mask3[(darkness >= fallback_thr) & (tube_mask > 0)] = 255
        mask3 = clean_mask(mask3, roi_bgr.shape)
        mask3, comp_count3 = filter_components(mask3, tube_mask)

        mask = mask3
        thr = fallback_thr
        comp_count = comp_count3

    darkness_u8 = normalize_uint8(darkness)

    return {
        "mask": mask,
        "tube_mask": tube_mask,
        "skin_ref_mask": skin_ref_mask,
        "darkness_map": darkness_u8,
        "eyebrow_pts": eyebrow_pts,
        "threshold": float(thr),
        "component_count": int(comp_count),
        "skin_median_L": float(skin_median),
    }


def draw_overlay(roi_bgr, mask, tube_mask, eyebrow_pts):
    overlay = roi_bgr.copy()

    # tube：藍色淡淡顯示
    tube_color = np.zeros_like(roi_bgr)
    tube_color[:, :, 0] = 255
    tube_bool = tube_mask > 0

    overlay[tube_bool] = cv2.addWeighted(
        overlay[tube_bool],
        0.80,
        tube_color[tube_bool],
        0.20,
        0
    )

    # mask：綠色顯示
    mask_color = np.zeros_like(roi_bgr)
    mask_color[:, :, 1] = 255
    mask_bool = mask > 0

    overlay[mask_bool] = cv2.addWeighted(
        overlay[mask_bool],
        0.40,
        mask_color[mask_bool],
        0.60,
        0
    )

    # landmarks：紅點
    for x, y in eyebrow_pts:
        cv2.circle(
            overlay,
            (int(round(x)), int(round(y))),
            2,
            (0, 0, 255),
            -1
        )

    return overlay


def make_debug_image(roi_bgr, tube_mask, skin_ref_mask, darkness_map, mask):
    h, w = roi_bgr.shape[:2]

    tube_vis = cv2.cvtColor(tube_mask, cv2.COLOR_GRAY2BGR)
    skin_vis = cv2.cvtColor(skin_ref_mask, cv2.COLOR_GRAY2BGR)
    dark_vis = cv2.cvtColor(darkness_map, cv2.COLOR_GRAY2BGR)
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # resize to same height if needed
    panels = [
        roi_bgr,
        tube_vis,
        skin_vis,
        dark_vis,
        mask_vis
    ]

    debug = np.hstack(panels)

    cv2.putText(debug, "ROI", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(debug, "Tube", (w + 10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(debug, "SkinRef", (2*w + 10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(debug, "Darkness", (3*w + 10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(debug, "Mask", (4*w + 10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return debug


def compute_features(roi_bgr, mask, tube_mask, darkness_map):
    h, w = mask.shape[:2]
    roi_area = h * w

    mask_bool = mask > 0
    tube_bool = tube_mask > 0

    mask_area = int(np.sum(mask_bool))
    tube_area = int(np.sum(tube_bool))

    mask_area_ratio = mask_area / float(roi_area) if roi_area > 0 else 0.0
    mask_tube_ratio = mask_area / float(tube_area) if tube_area > 0 else 0.0

    if mask_area > 0:
        ys, xs = np.where(mask_bool)
        mask_x1 = int(np.min(xs))
        mask_x2 = int(np.max(xs)) + 1
        mask_y1 = int(np.min(ys))
        mask_y2 = int(np.max(ys)) + 1
        mask_width = int(mask_x2 - mask_x1)
        mask_height = int(mask_y2 - mask_y1)

        mean_darkness = float(np.mean(darkness_map[mask_bool]))
        median_darkness = float(np.median(darkness_map[mask_bool]))
    else:
        mask_x1 = mask_x2 = mask_y1 = mask_y2 = ""
        mask_width = 0
        mask_height = 0
        mean_darkness = ""
        median_darkness = ""

    return {
        "mask_area": mask_area,
        "tube_area": tube_area,
        "mask_area_ratio": mask_area_ratio,
        "mask_tube_ratio": mask_tube_ratio,
        "mask_x1": mask_x1,
        "mask_y1": mask_y1,
        "mask_x2": mask_x2,
        "mask_y2": mask_y2,
        "mask_width": mask_width,
        "mask_height": mask_height,
        "mean_darkness": mean_darkness,
        "median_darkness": median_darkness,
    }


# =========================
# 5. Load ROI results
# =========================

if not ROI_CSV.exists():
    raise FileNotFoundError(f"Cannot find ROI csv: {ROI_CSV}")

df = pd.read_csv(ROI_CSV)
df = df.fillna("")

df = df[df["roi_success"].apply(is_success_value)].copy()
df = df.reset_index(drop=True)

print("Images to generate eyebrow masks v2:")
print(df.groupby(["source", "label"]).size())


# =========================
# 6. Run mask generation
# =========================

records = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    source = row["source"]
    label = row["label"]
    filename_stem = Path(row["filename"]).stem

    landmark_path = Path(row["aligned_landmark_path"])
    landmarks = np.load(landmark_path)

    for side in ["left", "right"]:
        if side == "left":
            roi_path = Path(row["left_roi_path"])
            roi_box = (
                int(row["left_x1"]),
                int(row["left_y1"]),
                int(row["left_x2"]),
                int(row["left_y2"]),
            )
        else:
            roi_path = Path(row["right_roi_path"])
            roi_box = (
                int(row["right_x1"]),
                int(row["right_y1"]),
                int(row["right_x2"]),
                int(row["right_y2"]),
            )

        safe_stem = make_safe_name(
            f"{i:06d}_{source}_{label}_{filename_stem}_{side}"
        )

        source_mask_dir = MASK_DIR / source
        source_overlay_dir = OVERLAY_DIR / source
        source_debug_dir = DEBUG_DIR / source

        source_mask_dir.mkdir(parents=True, exist_ok=True)
        source_overlay_dir.mkdir(parents=True, exist_ok=True)
        source_debug_dir.mkdir(parents=True, exist_ok=True)

        mask_path = source_mask_dir / f"{safe_stem}_mask.png"
        overlay_path = source_overlay_dir / f"{safe_stem}_overlay.jpg"
        debug_path = source_debug_dir / f"{safe_stem}_debug.jpg"

        record = row.to_dict()
        record.update({
            "side": side,
            "side_roi_path": str(roi_path),
            "mask_success": 0,
            "mask_failure_reason": "",
            "mask_path": "",
            "mask_overlay_path": "",
            "debug_path": "",
            "threshold_L": "",
            "component_count": "",
            "skin_median_L": "",
        })

        try:
            roi_bgr = imread_unicode(roi_path)

            if roi_bgr is None:
                raise ValueError("failed_to_read_roi")

            result = generate_mask_v2(
                roi_bgr=roi_bgr,
                landmarks=landmarks,
                roi_box=roi_box,
                side=side
            )

            mask = result["mask"]
            tube_mask = result["tube_mask"]
            skin_ref_mask = result["skin_ref_mask"]
            darkness_map = result["darkness_map"]
            eyebrow_pts = result["eyebrow_pts"]

            imwrite_unicode(mask_path, mask)

            if SAVE_OVERLAY:
                overlay = draw_overlay(
                    roi_bgr=roi_bgr,
                    mask=mask,
                    tube_mask=tube_mask,
                    eyebrow_pts=eyebrow_pts
                )
                imwrite_unicode(overlay_path, overlay)

            if SAVE_DEBUG:
                debug_img = make_debug_image(
                    roi_bgr=roi_bgr,
                    tube_mask=tube_mask,
                    skin_ref_mask=skin_ref_mask,
                    darkness_map=darkness_map,
                    mask=mask
                )
                imwrite_unicode(debug_path, debug_img)

            features = compute_features(
                roi_bgr=roi_bgr,
                mask=mask,
                tube_mask=tube_mask,
                darkness_map=darkness_map
            )

            record.update({
                "mask_success": 1,
                "mask_failure_reason": "",
                "mask_path": str(mask_path),
                "mask_overlay_path": str(overlay_path) if SAVE_OVERLAY else "",
                "debug_path": str(debug_path) if SAVE_DEBUG else "",
                "threshold_L": result["threshold"],
                "component_count": result["component_count"],
                "skin_median_L": result["skin_median_L"],
                **features
            })

        except Exception as e:
            record["mask_success"] = 0
            record["mask_failure_reason"] = str(e)

        records.append(record)


records_df = pd.DataFrame(records)
records_df.to_csv(OUTPUT_CSV, index=False)

print("\nDone.")
print(f"Saved mask features v2 to: {OUTPUT_CSV}")

print("\nMask generation v2 summary:")
print(records_df.groupby(["source", "label", "side"])["mask_success"].agg(["count", "sum"]))

failed_df = records_df[records_df["mask_success"] == 0]
print(f"\nFailed mask cases: {len(failed_df)}")

if len(failed_df) > 0:
    fail_path = FEATURE_DIR / "eyebrow_mask_failed_v2.csv"
    failed_df.to_csv(fail_path, index=False)
    print(f"Saved failed mask cases to: {fail_path}")
    print("\nFailure reasons:")
    print(failed_df["mask_failure_reason"].value_counts())

print("\nMask area ratio summary:")
success_df = records_df[records_df["mask_success"] == 1].copy()
print(success_df.groupby(["source", "label", "side"])["mask_area_ratio"].describe())

print("\nMask / tube ratio summary:")
print(success_df.groupby(["source", "label", "side"])["mask_tube_ratio"].describe())