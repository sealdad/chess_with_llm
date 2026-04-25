# chess_algo.py
import json
import numpy as np
import cv2

import torch
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image

# =========================
# Model paths / parameters
# =========================
DETECT_MODEL   = "/home/sealdad/sealdad_project/runs/detect/train_hardcase/weights/best.pt"
CLS_CKPT       = "/home/sealdad/sealdad_project/runs/chess_cls/mobilenetv3_small_v5/best.pt"
CLS_MAPPING    = "/home/sealdad/sealdad_project/runs/chess_cls/mobilenetv3_small_v5/class_mapping.json"
SEGMENT_MODEL  = "/home/sealdad/sealdad_project/runs/segment/board_seg_exp7/weights/best.pt"

# YOLO inference params
DETECT_IMSZ = 960
DETECT_CONF = 0.20
PAD         = 0.08
MIN_SIDE    = 20
DEDUP_PX    = 30       # merge detections whose centers are within this many pixels (~10 mm)

# Warp/grid params (from infer_and_draw_mask.py)
WARP_SIZE = 900

# Class -> letter
CLASS_LETTER = {
    "white_king":   "K",
    "white_queen":  "Q",
    "white_rook":   "R",
    "white_bishop": "B",
    "white_knight": "N",
    "white_pawn":   "P",
    "black_king":   "K",
    "black_queen":  "Q",
    "black_rook":   "R",
    "black_bishop": "B",
    "black_knight": "N",
    "black_pawn":   "P",
    "piece":        "X",
    "white_piece":  "W",
    "black_piece":  "B",
}


# =========================
# Piece classifier (stage-2)
# =========================
def load_classifier(device: torch.device):
    """Load stage-2 classifier + preprocessing, return (model, class_names, tf)."""
    with open(CLS_MAPPING, "r", encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta["classes"]
    num_classes = len(class_names)

    model = models.mobilenet_v3_small(pretrained=False)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)

    ckpt = torch.load(CLS_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, class_names, tf


@torch.no_grad()
def classify_patch(model, tf, class_names, device: torch.device, patch_bgr: np.ndarray):
    """Classify a single BGR patch. Return (cls_name, conf)."""
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(patch_rgb)

    x = tf(pil_img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)

    return class_names[idx.item()], float(conf.item())


def draw_box_and_label(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, cls_name: str):
    """Draw bbox + single-letter label."""
    label = CLASS_LETTER.get(cls_name, "?")
    # Color code: green=white piece, red=black piece, yellow=unknown
    if cls_name.startswith("white"):
        color = (0, 255, 0)
    elif cls_name.startswith("black"):
        color = (0, 0, 255)
    else:
        color = (0, 255, 255)
    thickness = 2

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    font_scale = 0.7
    font_thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    text_x1 = x1
    text_y1 = y1 - text_h - 6
    if text_y1 < 0:
        text_y1 = y1 + 6
    text_x2 = text_x1 + text_w + 6
    text_y2 = text_y1 + text_h + 6

    cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color, -1)
    cv2.putText(
        img,
        label,
        (text_x1 + 3, text_y2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        font_thickness,
        lineType=cv2.LINE_AA,
    )


# =========================
# Models helpers (YOLO)
# =========================
def load_piece_detector(model_path: str = DETECT_MODEL):
    """Load YOLO detector once (don't create inside the loop)."""
    return YOLO(model_path)

def load_board_segmenter(model_path: str = SEGMENT_MODEL):
    """Load YOLO segmenter once (don't create inside the function)."""
    return YOLO(model_path)


def mask_board_region(img_bgr: np.ndarray, mask_01: np.ndarray, dilate_px: int = 30) -> np.ndarray:
    """Apply board mask to image: keep board region, black out everything else.

    Dilates the mask slightly so pieces on the board edge aren't clipped.

    Args:
        img_bgr: original BGR image
        mask_01: binary mask (H,W) uint8 {0,1} from segment_chessboard
        dilate_px: pixels to dilate the mask boundary

    Returns:
        Masked BGR image (same shape as input)
    """
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        dilated = cv2.dilate(mask_01, kernel, iterations=1)
    else:
        dilated = mask_01
    return cv2.bitwise_and(img_bgr, img_bgr, mask=dilated)


# =========================
# Chessboard segmentation
# =========================
def segment_chessboard(seg_model: YOLO, color_img_bgr: np.ndarray):
    """
    Input BGR image -> return mask (H,W) uint8 {0,1}, or None.
    Mask is resized to match the original image resolution.
    """
    result = seg_model(color_img_bgr)[0]
    if (not result.masks) or (result.masks.data is None) or (result.masks.data.shape[0] == 0):
        return None
    mask = result.masks.data[0].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)

    # Resize mask to original image resolution so quad coords match piece coords
    h, w = color_img_bgr.shape[:2]
    mh, mw = mask.shape[:2]
    if (mh, mw) != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask

def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    c = pts.mean(axis=0)

    angles = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])
    order = np.argsort(angles)  # CCW
    pts = pts[order]

    # 找 TL 作為起點：y最小，若相同取x最小
    tl_idx = np.lexsort((pts[:,0], pts[:,1]))[0]
    pts = np.roll(pts, -tl_idx, axis=0)

    # 現在 pts 是 [TL, ?, ?, ?] 且 CCW
    # 讓它變成 [TL, TR, BR, BL]：CCW 時第二個通常是 BL
    # 所以如果你需要順序固定，可用 x 判斷第二點是 TR 還 BL
    if pts[1,0] < pts[3,0]:
        # 代表 pts[1] 在左邊 => 是 BL，把順序轉成 [TL,TR,BR,BL]
        pts = pts[[0,3,2,1]]

    return pts


def _quad_is_valid(quad: np.ndarray,
                   min_dist: float = 20.0,
                   min_area: float = 200.0) -> bool:
    q = quad.astype(np.float32)

    # 1) distinct points
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(q[i] - q[j]) < min_dist:
                return False

    # 2) non-degenerate area
    area = 0.5 * abs(np.cross(q[1]-q[0], q[2]-q[0]) + np.cross(q[2]-q[0], q[3]-q[0]))
    if area < min_area:
        return False

    # 3) convexity (TL,TR,BR,BL order assumed)
    def cross_z(a, b, c):
        return np.cross(b - a, c - b)

    zs = []
    for i in range(4):
        a = q[i]
        b = q[(i+1) % 4]
        c = q[(i+2) % 4]
        zs.append(cross_z(a, b, c))
    zs = np.array(zs)

    # all same sign (allow tiny numerical noise)
    if not (np.all(zs > -1e-6) or np.all(zs < 1e-6)):
        return False

    return True



def _find_4_corners_from_hull(hull: np.ndarray) -> np.ndarray:
    """
    Find the 4 extreme corners of a convex hull that best represent a quadrilateral.
    Uses the farthest point method to find corners that preserve the actual shape.
    Returns None if 4 unique corners cannot be found.
    """
    pts = hull.reshape(-1, 2).astype(np.float32)
    if len(pts) < 4:
        return None

    # Find the 4 extreme points: top-left, top-right, bottom-right, bottom-left
    # TL: min(x + y), TR: min(y - x), BR: max(x + y), BL: max(y - x)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 1] - pts[:, 0]

    idx_tl = np.argmin(s)
    idx_br = np.argmax(s)
    idx_tr = np.argmin(d)
    idx_bl = np.argmax(d)

    # Check all 4 indices are unique
    indices = [idx_tl, idx_tr, idx_br, idx_bl]
    if len(set(indices)) != 4:
        return None

    tl = pts[idx_tl]
    tr = pts[idx_tr]
    br = pts[idx_br]
    bl = pts[idx_bl]

    return np.array([tl, tr, br, bl], dtype=np.float32)

import numpy as np
import cv2

def get_quad_from_mask(mask_u8_255: np.ndarray):
    """
    mask_u8_255: uint8 mask in {0,255}
    Return quad 4x2 float32 (ordered TL,TR,BR,BL), or None.
    Extracts the actual quadrilateral shape from the mask, not a bounding rectangle.
    """

    # -------------------------
    # helpers (local, robust)
    # -------------------------
    def order_points_ccw_tl(pts: np.ndarray) -> np.ndarray:
        """Robustly order 4 points as [TL, TR, BR, BL]."""
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        pts = pts[np.argsort(ang)]  # CCW
        tl = np.lexsort((pts[:, 0], pts[:, 1]))[0]  # smallest y then x
        pts = np.roll(pts, -tl, axis=0)
        # ensure [TL,TR,BR,BL]
        if pts[1, 0] < pts[3, 0]:
            pts = pts[[0, 3, 2, 1]]
        return pts.astype(np.float32)

    def quad_is_valid_strict(quad: np.ndarray, min_dist: float = 20.0, min_area: float = 200.0) -> bool:
        q = np.asarray(quad, dtype=np.float32).reshape(4, 2)

        # distinct points
        for i in range(4):
            for j in range(i + 1, 4):
                if np.linalg.norm(q[i] - q[j]) < min_dist:
                    return False

        # non-degenerate area
        area = float(abs(cv2.contourArea(q.reshape(-1, 1, 2))))
        if area < min_area:
            return False

        # convex & non-self-cross
        if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
            return False

        return True

    def score_quad(quad: np.ndarray, hull_xy: np.ndarray) -> float:
        """Higher is better. Prefer quads that cover the hull reasonably."""
        q_area = float(abs(cv2.contourArea(quad.reshape(-1, 1, 2))))
        h_area = float(abs(cv2.contourArea(hull_xy.reshape(-1, 1, 2))))
        if h_area <= 1e-6:
            return q_area
        ratio = q_area / (h_area + 1e-6)

        penalty = 0.0
        if ratio < 0.4:
            penalty += (0.4 - ratio) * 1000.0
        if ratio > 1.8:
            penalty += (ratio - 1.8) * 1000.0

        return q_area - penalty

    def pick_4_by_turning(pts: np.ndarray) -> np.ndarray:
        """
        pts: ordered polygon vertices (Nx2) in boundary order (CW/CCW).
        Pick 4 vertices with largest turning (corner sharpness proxy).
        """
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        n = p.shape[0]
        if n <= 4:
            return p[:4]

        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            a = p[(i - 1) % n]
            b = p[i]
            c = p[(i + 1) % n]
            v1 = a - b
            v2 = c - b
            n1 = np.linalg.norm(v1) + 1e-6
            n2 = np.linalg.norm(v2) + 1e-6
            cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            scores[i] = 1.0 - cosang  # bigger => sharper
        idx = np.argsort(scores)[::-1][:4]
        return p[idx]

    # -------------------------
    # main
    # -------------------------
    cnts, _ = cv2.findContours(mask_u8_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)

    # Use convex hull to fill concavities from occlusions
    hull = cv2.convexHull(cnt)  # (N,1,2) in boundary order
    hull_xy = hull.reshape(-1, 2).astype(np.float32)

    arc = cv2.arcLength(hull, True)
    if arc < 1e-6:
        return None

    candidates = []

    # Try approxPolyDP; accept 4 directly, or 5~6 then reduce to 4 by turning score
    for eps_factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20]:
        epsilon = eps_factor * arc
        approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2).astype(np.float32)
        k = approx.shape[0]

        if k == 4:
            quad = order_points_ccw_tl(approx)
            if quad_is_valid_strict(quad):
                candidates.append(quad)

        elif 4 < k <= 6:
            picked = pick_4_by_turning(approx)
            if picked.shape[0] == 4:
                quad = order_points_ccw_tl(picked)
                if quad_is_valid_strict(quad):
                    candidates.append(quad)

        elif k < 4:
            # further increasing epsilon will only reduce points more
            break

    if candidates:
        best = max(candidates, key=lambda q: score_quad(q, hull_xy))
        return best.astype(np.float32)

    # Fallback #1: your extreme-corners-from-hull (preserves perspective shape)
    quad = _find_4_corners_from_hull(hull)
    if quad is not None:
        quad = order_points_ccw_tl(np.asarray(quad, dtype=np.float32).reshape(4, 2))
        if quad_is_valid_strict(quad, min_dist=10.0, min_area=200.0):
            return quad

    # Fallback #2: minAreaRect (always 4 points, last resort)
    rect = cv2.minAreaRect(hull)  # (center,(w,h),angle)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = order_points_ccw_tl(box)
    if quad_is_valid_strict(box, min_dist=10.0, min_area=200.0):
        return box

    # Debug info
    epsilon = 0.15 * arc
    approx_dbg = cv2.approxPolyDP(hull, epsilon, True)

    corner_dists = []
    if quad is not None:
        q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        labels = ["TL", "TR", "BR", "BL"]
        for i in range(4):
            for j in range(i + 1, 4):
                d = np.linalg.norm(q[i] - q[j])
                corner_dists.append(f"{labels[i]}-{labels[j]}:{d:.1f}")

    raise ValueError(
        f"get_quad_from_mask: could not extract valid quad from mask. "
        f"Hull has {len(hull_xy)} points. approxPolyDP(0.15*arc) gave {approx_dbg.shape[0]} points. "
        f"Extreme corners: {q.tolist() if quad is not None else None}. "
        f"Corner distances: {corner_dists}"
    )



def warp_board(img_bgr: np.ndarray, quad_4x2: np.ndarray, out_size: int = WARP_SIZE):
    dst = np.array([[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]], np.float32)
    H = cv2.getPerspectiveTransform(quad_4x2.astype(np.float32), dst)
    Hinv = cv2.getPerspectiveTransform(dst, quad_4x2.astype(np.float32))
    warped = cv2.warpPerspective(img_bgr, H, (out_size, out_size), flags=cv2.INTER_LINEAR)
    return warped, H, Hinv


def make_grid_9x9_in_warp(size: int = WARP_SIZE):
    """Generate 9x9 intersection points (8x8 cells) in warp-space."""
    xs = np.linspace(0, size - 1, 9, dtype=np.float32)
    ys = np.linspace(0, size - 1, 9, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)  # (9,9,2)
    return grid


def project_points(Hinv: np.ndarray, pts_xy: np.ndarray):
    pts = pts_xy.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)
    return out


def draw_grid_lines(img_bgr: np.ndarray, grid_9x9: np.ndarray,
                    color_h=(200, 200, 0), color_v=(0, 200, 200), thickness: int = 2):
    g = grid_9x9.astype(int)
    for r in range(9):
        for c in range(8):
            cv2.line(img_bgr, tuple(g[r, c]), tuple(g[r, c + 1]), color_h, thickness)
    for c in range(9):
        for r in range(8):
            cv2.line(img_bgr, tuple(g[r, c]), tuple(g[r + 1, c]), color_v, thickness)
    return img_bgr


def draw_corners(img_bgr: np.ndarray, grid_9x9: np.ndarray, color=(0, 0, 255), radius: int = 3):
    g = grid_9x9.astype(int)
    for r in range(9):
        for c in range(9):
            cv2.circle(img_bgr, tuple(g[r, c]), radius, color, -1)
    return img_bgr


def refine_corners_in_warp(warped_bgr: np.ndarray, grid_9x9: np.ndarray, win: int = 11, warp_size: int = WARP_SIZE):
    """
    Optional: refine each predicted intersection in warp-space using Harris response,
    with morphology-open to suppress pieces details.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    k = max(21, warp_size // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    base = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    harris = cv2.cornerHarris(np.float32(base), 2, 3, 0.04)
    harris = cv2.dilate(harris, None)

    refined = grid_9x9.copy()
    half = win // 2
    for r in range(9):
        for c in range(9):
            x, y = refined[r, c]
            xi, yi = int(round(x)), int(round(y))
            x0, x1 = max(0, xi - half), min(warp_size - 1, xi + half)
            y0, y1 = max(0, yi - half), min(warp_size - 1, yi + half)

            patch = harris[y0:y1 + 1, x0:x1 + 1]
            if patch.size == 0:
                continue
            dy, dx = np.unravel_index(np.argmax(patch), patch.shape)
            refined[r, c] = (x0 + dx, y0 + dy)

    return refined


def grid_from_mask_and_image(img_bgr: np.ndarray, mask_01: np.ndarray,
                             warp_size: int = WARP_SIZE, do_refine: bool = False, refine_win: int = 11):
    """
    Convenience pipeline:
      mask(0/1) -> quad -> warp -> 9x9 grid -> (optional) refine -> back-project to original

    Return dict or None:
      {
        "quad": (4,2),
        "warped": warped_img,
        "H": H,
        "Hinv": Hinv,
        "grid_warp": (9,9,2),
        "grid_orig": (9,9,2),
      }
    """
    if mask_01 is None:
        return None
    mask_u8 = (mask_01 > 0).astype(np.uint8) * 255

    quad = get_quad_from_mask(mask_u8)
    if quad is None:
        return None

    warped, H, Hinv = warp_board(img_bgr, quad, out_size=warp_size)

    grid_warp = make_grid_9x9_in_warp(warp_size)
    if do_refine:
        grid_warp = refine_corners_in_warp(warped, grid_warp, win=refine_win, warp_size=warp_size)

    grid_orig = project_points(Hinv, grid_warp).reshape(9, 9, 2)

    # Sanity check: dst corners projected back via Hinv should match quad
    dst_corners = np.array([[0, 0], [warp_size - 1, 0],
                            [warp_size - 1, warp_size - 1], [0, warp_size - 1]], dtype=np.float32)
    reprojected_corners = project_points(Hinv, dst_corners).reshape(4, 2)
    corner_errors = np.linalg.norm(reprojected_corners - quad, axis=1)
    max_corner_err = float(corner_errors.max())
    mean_corner_err = float(corner_errors.mean())
    if max_corner_err > 5.0:
        import logging
        logging.getLogger(__name__).warning(
            "grid_from_mask_and_image: reprojection error too large! "
            "max=%.2f mean=%.2f px (corners: %s vs %s)",
            max_corner_err, mean_corner_err,
            reprojected_corners.tolist(), quad.tolist(),
        )

    return {
        "quad": quad,
        "reprojected_corners": reprojected_corners,
        "warped": warped,
        "H": H,
        "Hinv": Hinv,
        "grid_warp": grid_warp,
        "grid_orig": grid_orig,
        "reprojection_error": {"max_px": max_corner_err, "mean_px": mean_corner_err},
    }


# =========================
# Pieces detection pipeline (single frame)
# =========================
def _dedup_detections(detections, dist_px=DEDUP_PX):
    """Remove near-duplicate detections. Keep the one with higher det_conf."""
    if len(detections) <= 1:
        return detections
    # Sort by det_conf descending so we keep higher-confidence first
    ranked = sorted(detections, key=lambda d: d["det_conf"], reverse=True)
    keep = []
    for det in ranked:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        too_close = False
        for kept in keep:
            kx1, ky1, kx2, ky2 = kept["bbox"]
            kcx, kcy = (kx1 + kx2) / 2.0, (ky1 + ky2) / 2.0
            dist = ((cx - kcx) ** 2 + (cy - kcy) ** 2) ** 0.5
            if dist < dist_px:
                too_close = True
                break
        if not too_close:
            keep.append(det)
    return keep


def classify_piece_color(img_bgr: np.ndarray, bbox: list) -> str:
    """Classify a detected piece as white or black based on pixel intensity.

    Crops the center region of the bounding box, converts to grayscale,
    and uses average intensity to determine color.

    Args:
        img_bgr: Full image in BGR format
        bbox: [x1, y1, x2, y2] bounding box

    Returns:
        "white_piece" or "black_piece"
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # Use center 50% of the bbox to avoid board color bleeding in from edges
    cx1 = x1 + bw // 4
    cy1 = y1 + bh // 4
    cx2 = x2 - bw // 4
    cy2 = y2 - bh // 4

    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2

    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "piece"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(gray.mean())

    # Threshold calibrated from 249 real samples across 8 captures:
    #   White pieces: mean=123, range 91-176
    #   Black pieces: mean=64,  range 46-98
    #   Threshold 89 → 99.2% accuracy
    return "white_piece" if mean_intensity > 89 else "black_piece"


def detect_pieces(
    det_model: YOLO,
    color_img_bgr: np.ndarray,
    imgsz: int = DETECT_IMSZ,
    conf: float = DETECT_CONF,
    pad: float = PAD,
    min_side: int = MIN_SIDE,
    dedup_px: float = DEDUP_PX,
):
    """
    Run YOLO detect -> dedup.  No classification — detection alone is enough
    for occupancy-based move tracking. Post-processes with color detection
    to distinguish white vs black pieces.

    Return list of dict:
      [{"bbox":[x1,y1,x2,y2], "cls_name":"white_piece"|"black_piece", "cls_conf":1.0, "det_conf":...}, ...]
    """
    h, w = color_img_bgr.shape[:2]
    r = det_model(color_img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]

    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    det_confs = r.boxes.conf.cpu().numpy()
    cls_indices = r.boxes.cls.cpu().numpy().astype(int)

    for xyxy, det_conf, cls_idx in zip(boxes_xyxy, det_confs, cls_indices):
        x1, y1, x2, y2 = map(int, xyxy)

        dx = int(pad * (x2 - x1))
        dy = int(pad * (y2 - y1))
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w, x2 + dx)
        y2 = min(h, y2 + dy)

        if x2 <= x1 or y2 <= y1:
            continue
        bw, bh = (x2 - x1), (y2 - y1)
        if bw < min_side or bh < min_side:
            continue

        # Post-process: classify piece color (white vs black) from image crop
        cls_name = classify_piece_color(color_img_bgr, [x1, y1, x2, y2])

        out.append({
            "bbox": [x1, y1, x2, y2],
            "cls_name": cls_name,
            "cls_conf": float(det_conf),
            "det_conf": float(det_conf),
        })

    # Remove near-duplicate detections (centers within dedup_px pixels)
    if dedup_px > 0:
        out = _dedup_detections(out, dedup_px)

    return out


def detect_and_classify_pieces(
    det_model: YOLO,
    cls_model,
    cls_tf,
    class_names,
    device: torch.device,
    color_img_bgr: np.ndarray,
    imgsz: int = DETECT_IMSZ,
    conf: float = DETECT_CONF,
    pad: float = PAD,
    min_side: int = MIN_SIDE,
    dedup_px: float = DEDUP_PX,
):
    """
    Run YOLO detect -> crop -> classify -> dedup.
    Return list of dict:
      [{"bbox":[x1,y1,x2,y2], "cls_name":..., "cls_conf":..., "det_conf":...}, ...]
    """
    h, w = color_img_bgr.shape[:2]
    r = det_model(color_img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]

    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    det_confs = r.boxes.conf.cpu().numpy()

    for xyxy, det_conf in zip(boxes_xyxy, det_confs):
        x1, y1, x2, y2 = map(int, xyxy)

        dx = int(pad * (x2 - x1))
        dy = int(pad * (y2 - y1))
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w, x2 + dx)
        y2 = min(h, y2 + dy)

        if x2 <= x1 or y2 <= y1:
            continue
        bw, bh = (x2 - x1), (y2 - y1)
        if bw < min_side or bh < min_side:
            continue

        patch = color_img_bgr[y1:y2, x1:x2].copy()
        if patch.size == 0:
            continue

        cls_name, cls_conf = classify_patch(cls_model, cls_tf, class_names, device, patch)

        out.append({
            "bbox": [x1, y1, x2, y2],
            "cls_name": cls_name,
            "cls_conf": cls_conf,
            "det_conf": float(det_conf),
        })

    # Remove near-duplicate detections (centers within dedup_px pixels)
    if dedup_px > 0:
        out = _dedup_detections(out, dedup_px)

    return out


# =========================
# Debug visualization helpers
# =========================
def draw_segmentation_overlay(img_bgr: np.ndarray, mask_01: np.ndarray, grid_info: dict = None):
    """Draw segmentation mask overlay + hull + quad and reprojected corners. Returns annotated copy."""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # Green mask overlay + hull
    if mask_01 is not None:
        mh, mw = mask_01.shape[:2]
        if (mh, mw) != (h, w):
            mask_resized = cv2.resize(mask_01.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask_01.astype(np.uint8)
        overlay = vis.copy()
        overlay[mask_resized > 0] = [0, 200, 0]
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # Draw convex hull from mask — magenta
        mask_255 = (mask_resized > 0).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            hull = cv2.convexHull(cnt)
            cv2.drawContours(vis, [hull], -1, (255, 0, 255), 2)
            # Draw hull vertices
            for pt in hull.reshape(-1, 2):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (255, 0, 255), -1)
            cv2.putText(vis, f"Hull: {len(hull)} pts", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    if grid_info is not None:
        quad = grid_info.get("quad")
        reprojected_corners = grid_info.get("reprojected_corners")
        reproj_error = grid_info.get("reprojection_error")

        # Quad (from mask) — cyan
        if quad is not None:
            pts = quad.astype(int).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
            for i, corner in enumerate(quad):
                label = ["TL", "TR", "BR", "BL"][i]
                cx, cy = int(corner[0]), int(corner[1])
                cv2.circle(vis, (cx, cy), 8, (0, 255, 255), 2)
                cv2.putText(vis, f"Q:{label}({cx},{cy})", (cx + 10, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # Reprojected corners (dst via Hinv) — green
        if reprojected_corners is not None:
            rpts = reprojected_corners.astype(int).reshape(-1, 1, 2)
            cv2.polylines(vis, [rpts], True, (0, 255, 0), 2)
            for i, corner in enumerate(reprojected_corners):
                label = ["TL", "TR", "BR", "BL"][i]
                rx, ry = int(corner[0]), int(corner[1])
                cv2.circle(vis, (rx, ry), 8, (0, 255, 0), -1)
                cv2.putText(vis, f"R:{label}({rx},{ry})", (rx + 10, ry + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Reprojection error
        if reproj_error:
            cv2.putText(vis, f"Reproj err: max={reproj_error['max_px']:.1f} mean={reproj_error['mean_px']:.1f} px",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Legend
        cv2.putText(vis, "Magenta=hull  Cyan=quad  Green=reprojected", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


def draw_detection_boxes(img_bgr: np.ndarray, det_boxes: np.ndarray, det_confs: np.ndarray):
    """Draw raw YOLO detection boxes with confidence. Returns annotated copy."""
    vis = img_bgr.copy()
    if det_boxes is None or len(det_boxes) == 0:
        cv2.putText(vis, "No detections", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return vis
    for xyxy, conf in zip(det_boxes, det_confs):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Detected: {len(det_boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return vis


def draw_classification_results(img_bgr: np.ndarray, pieces: list):
    """Draw classified pieces with color-coded boxes. Returns annotated copy."""
    vis = img_bgr.copy()
    if not pieces:
        cv2.putText(vis, "No pieces classified", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return vis
    for p in pieces:
        x1, y1, x2, y2 = p["bbox"]
        cls_name = p["cls_name"]
        cls_conf = p["cls_conf"]
        color = (255, 150, 0) if cls_name.startswith("white_") else (0, 80, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        letter = CLASS_LETTER.get(cls_name, "?")
        side = "W" if cls_name.startswith("white_") else "B"
        label = f"{side}{letter} {cls_conf:.2f}"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    white_count = sum(1 for p in pieces if p["cls_name"].startswith("white_"))
    black_count = len(pieces) - white_count
    cv2.putText(vis, f"White: {white_count}  Black: {black_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return vis


def draw_final_board(img_bgr: np.ndarray, board_state: dict, grid_9x9: np.ndarray, fen: str = ""):
    """Draw final board state with grid, piece labels, and FEN. Returns annotated copy."""
    vis = img_bgr.copy()
    if grid_9x9 is not None:
        draw_grid_lines(vis, grid_9x9, thickness=1)
    for square, info in board_state.items():
        bbox = info["bbox"]
        cls_name = info["piece"]
        color = (255, 150, 0) if cls_name.startswith("white_") else (0, 80, 255)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        letter = CLASS_LETTER.get(cls_name, "?")
        label = f"{square}:{letter}"
        cv2.putText(vis, label, (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    if fen:
        fen_short = fen.split(" ")[0] if " " in fen else fen
        cv2.putText(vis, fen_short, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Pieces: {len(board_state)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    return vis
