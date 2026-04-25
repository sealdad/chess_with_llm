"""
calibration.py ── 傳統棋盤格 (Chessboard) 相機內參校正工具
for RX-200 RealSense 視覺管線。

新增函式
------------
calibrate_camera_chessboard(...)
    使用一般棋盤格影像計算相機內參與畸變係數，可輸出 YAML。

get_chessboard_corners(...)
    從單張影像中提取棋盤格角點座標，用於手眼校正。

已存在函式
------------
calibrate_hand_eye(...)
    保留之前的手眼校正 (eye-to-hand) 功能，無需改動。
"""

import glob
import os
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
import yaml


# ──────────────────────────────────────────────────────────────
#  Camera intrinsic calibration (Chessboard)
# ──────────────────────────────────────────────────────────────
def calibrate_camera_chessboard(
    images_glob: str,
    *,
    board_width: int = 9,
    board_height: int = 6,
    square_size: float = 0.025,
    output_path: Optional[str] = None,
    show_extraction: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    使用棋盤格影像計算相機內參與畸變參數。

    Args
    ----
    images_glob:       glob pattern，例 "calib_images/chess_*.png"
    board_width:       棋盤格內部角點橫向數量 (columns-1)，例如 9 表示有 9 個內部交點橫向
    board_height:      棋盤格內部角點縱向數量 (rows-1)，例如 6 表示有 6 個內部交點縱向
    square_size:       每格邊長 (公尺)，例如 0.025 表示 25 mm
    output_path:       若提供檔名，校正結果會存為 YAML (包含 camera_matrix、dist_coeffs)
    show_extraction:   是否在每張影像上顯示所偵測到的角點 (True/False)

    Returns
    -------
    camera_matrix:     3×3 相機內參矩陣
    dist_coeffs:       1×5 的畸變係數 (k1,k2,p1,p2,k3)
    rvecs:             每張影像的旋轉向量
    tvecs:             每張影像的平移向量
    """

    # 1. 準備棋盤格要對應的「世界座標」(object points)
    #    每個內部交點在 (x,y,0)，z=0；x = (0,1,...,board_width-1)*square_size
    objp = np.zeros((board_height * board_width, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
    objp *= square_size

    # 2. 讀取所有影像、準備儲存每張影像的 object points 與 image points
    objpoints: List[np.ndarray] = []  # 3D points in real world space
    imgpoints: List[np.ndarray] = []  # 2D points in image plane

    image_files = sorted(glob.glob(images_glob))
    if not image_files:
        raise RuntimeError(f"No images found for pattern: {images_glob}")

    # 3. 檢查每張影像，提取棋盤格角點
    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"[Warning] Unable to read {fname}, skip")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (width, height)

        # 3.1 找 input chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            (board_width, board_height),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                  | cv2.CALIB_CB_NORMALIZE_IMAGE
                  | cv2.CALIB_CB_FAST_CHECK
        )

        if not ret:
            print(f"[{fname}] 找不到 {board_width}×{board_height} 棋盤角點，跳過")
            continue

        # 3.2 對初步找到的角點做子像素精煉 (cornerSubPix)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=criteria
        )

        if show_extraction:
            vis = cv2.drawChessboardCorners(
                img.copy(),
                (board_width, board_height),
                corners_subpix,
                True
            )
            cv2.imshow(f"Chessboard Corners: {os.path.basename(fname)}", vis)
            cv2.waitKey(500)
            cv2.destroyWindow(f"Chessboard Corners: {os.path.basename(fname)}")

        # 3.3 如果成功提取，就把 object points、image points 加入列表
        objpoints.append(objp)
        imgpoints.append(corners_subpix)

    # 確保至少 5–10 張成功提取，否則難以收斂
    if len(objpoints) < 5:
        raise RuntimeError(f"有效棋盤格圖像 < 5 ({len(objpoints)})，無法校正")

    # 4. 呼叫 calibrateCamera
    #    flags 可選 cv2.CALIB_RATIONAL_MODEL 之類，這裡只用基本參數
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=0
    )

    if not ret:
        raise RuntimeError("cv2.calibrateCamera 未能收斂")

    # 5. 選擇性把結果存成 YAML
    if output_path:
        yaml.safe_dump(
            {
                "camera_matrix": camera_matrix.tolist(),
                "dist_coeffs": dist_coeffs.ravel().tolist()
            },
            open(output_path, "w"),
        )

    print(f"校正完成：使用了 {len(objpoints)} 張有效影像")
    return camera_matrix, dist_coeffs, rvecs, tvecs


# ──────────────────────────────────────────────────────────────
#  Hand–Eye (eye-to-hand) calibration (沿用以前)
# ──────────────────────────────────────────────────────────────
def calibrate_hand_eye(
    r_gripper2base: List[np.ndarray],
    t_gripper2base: List[np.ndarray],
    r_target2cam: List[np.ndarray],
    t_target2cam: List[np.ndarray],
    *,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenCV 手眼校正 (eye-to-hand)。

    r_gripper2base / t_gripper2base : 末端→基座
    r_target2cam   / t_target2cam   : 標定板→相機

    Returns
    -------
    R_cam2gripper, t_cam2gripper : 相機→末端剛性變換
    """
    R_cam2grip, t_cam2grip = cv2.calibrateHandEye(
        R_gripper2base=r_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=r_target2cam,
        t_target2cam=t_target2cam,
        method=method,
    )

    if output_path:
        yaml.safe_dump(
            {
                "R_cam2gripper": R_cam2grip.tolist(),
                "t_cam2gripper": t_cam2grip.tolist(),
                "method": int(method),
            },
            open(output_path, "w"),
        )
    return R_cam2grip, t_cam2grip


def get_chessboard_corners(
    image: Union[str, np.ndarray],
    *,
    board_width: int = 9,
    board_height: int = 6,
    square_size: float = 0.025,
    show_corners: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    從單張影像中提取棋盤格角點座標，用於手眼校正。

    Args
    ----
    image:          影像路徑或 numpy array
    board_width:    棋盤格內部角點橫向數量 (columns-1)
    board_height:   棋盤格內部角點縱向數量 (rows-1)
    square_size:    每格邊長 (公尺)
    show_corners:   是否顯示角點偵測結果

    Returns
    -------
    corners:        角點影像座標 (N×1×2 的 numpy array，N = board_width × board_height)
    objp:           角點世界座標 (N×3 的 numpy array，z=0)
    vis_image:      可視化結果 (如果 show_corners=True)
    """
    # 1. 讀取影像
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            print(f"[Error] Unable to read image: {image}")
            return None, None, None
    else:
        img = image.copy()

    # 2. 轉換為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 找棋盤格角點
    ret, corners = cv2.findChessboardCorners(
        gray,
        (board_width, board_height),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
              | cv2.CALIB_CB_NORMALIZE_IMAGE
              | cv2.CALIB_CB_FAST_CHECK
    )

    if not ret:
        print(f"[Error] 找不到 {board_width}×{board_height} 棋盤角點")
        return None, None, None

    # 4. 子像素精煉
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_subpix = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=criteria
    )

    # 5. 準備世界座標
    objp = np.zeros((board_height * board_width, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
    objp *= square_size

    # 6. 可選：顯示結果
    vis_image = None
    if show_corners:
        vis_image = cv2.drawChessboardCorners(
            img.copy(),
            (board_width, board_height),
            corners_subpix,
            True
        )
        cv2.imshow("Chessboard Corners", vis_image)
        cv2.waitKey(0)
        cv2.destroyWindow("Chessboard Corners")

    return corners_subpix, objp, vis_image


# ──────────────────────────────────────────────────────────────
#  Example: 如果直接執行此檔案，做棋盤格校正
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) 修改為你放置棋盤格影像的路徑
    #    (請確保檔名符合 "calib_images/chess_01.png"、"chess_02.png" 等模式)
    images_pattern = "calib_images/chess_*.png"

    # 2) 棋盤格參數 (內部角點數)
    board_w = 9   # 9 內部角點 → 10 格子橫向
    board_h = 6   # 6 內部角點 → 7 格子縱向
    square_size = 0.025  # 25 mm

    # 3) 執行校正
    K, D, rvecs, tvecs = calibrate_camera_chessboard(
        images_glob=images_pattern,
        board_width=board_w,
        board_height=board_h,
        square_size=square_size,
        output_path="camera_intrinsics_chess.yaml",
        show_extraction=True  # 顯示每張影像的角點提取結果
    )

    # 4) 印出結果
    print("=== 相機內參 (Chessboard) 校正完成 ===")
    print("Camera matrix:\n", K)
    print("Distortion coeffs:\n", D.ravel())
    print("已將結果存為 camera_intrinsics_chess.yaml")
