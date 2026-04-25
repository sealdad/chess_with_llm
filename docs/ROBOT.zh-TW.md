[← 回 README](../README.zh-TW.md) ・ [English Version](ROBOT.md)

# Physical Robot Mode — 實體機械手臂模式

把螢幕上的西洋棋搬到真實的棋盤上。這篇文件涵蓋 **RX-200 實體版** 從零開始所需要的全部資訊：硬體準備、視覺管線、手眼校準、64 格子教學、IK 策略、吃子/升變的實作細節、所有 YAML 設定檔結構、REST API 契約,以及把這套架構接到 **其他機械手臂** 的最小契約。

> 軟體本身不綁死在 RX-200 上。如果你想接 UR5、DOBOT、自組臂,只要實作下面 [REST API 契約](#rest-api-契約) 列出的少數幾個 endpoint,Agent 跟 Vision 就能無痛接上。

---

## 目錄

- [整體架構](#整體架構)
- [硬體清單](#硬體清單)
- [首次啟動流程](#首次啟動流程)
- [視覺 6 階段管線](#視覺-6-階段管線)
- [AprilTag 手眼校準](#apriltag-手眼校準)
- [64 格子教學](#64-格子教學)
- [Board Surface Z 校準](#board-surface-z-校準)
- [IK 搜尋策略](#ik-搜尋策略)
- [Grasp 高度計算](#grasp-高度計算)
- [吃子與升變處理](#吃子與升變處理)
- [YAML 設定檔參考](#yaml-設定檔參考)
- [REST API 契約](#rest-api-契約)
- [接到自己的機器人](#接到自己的機器人)
- [常見問題](#常見問題)

---

## 整體架構

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Agent :8000   │ ──▶│   Vision :8001  │    │   Robot :8002   │
│   (LangGraph)   │    │  (RealSense +   │    │  (ROS Noetic +  │
│                 │ ──▶│   YOLO + AprT)  │    │   Interbotix)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       ▲                      ▲
        └───────────────────────┴──────────────────────┘
                       HTTP REST + JSON
```

- **Vision** 把實體棋盤拍下來,輸出 FEN + 每顆棋子的 3D 位置。
- **Robot** 把 UCI 走法 (例如 `e2e4`) 轉成手臂的 pick / place / capture 動作。
- **Agent** 負責整合:讀視覺 → 判斷人類走了哪一步 → LLM/Stockfish 決定回應 → 呼叫 Robot 走棋。

啟動方式:

```bash
docker-compose -f docker-compose.physical.yaml up --build
```

> 第一次 build robot image 會跑很久 (~30 min),因為要 clone + compile 整個 Interbotix ROS workspace。Vision image 也會比較久 (PyTorch + RealSense SDK)。後續啟動會用 cache,只要幾秒。

---

## 硬體清單

| 元件 | 規格 | 備註 |
|------|------|------|
| **機械手臂** | Interbotix RX-200 (5-DOF) | 透過 USB serial 連接,通常掛在 `/dev/ttyUSB0` |
| **相機** | Intel RealSense D4xx (D435 / D435i / D455) | 支援 RGB + 深度 + 對齊 |
| **棋盤** | 8×8 標準棋盤 | 預設每格 50mm (`BOARD_SQUARE_SIZE = 0.05`) |
| **棋子** | 任意西洋棋子 | 用 YOLO 訓練識別,高度建議 35–80mm |
| **AprilTag** | tag36h11 family,45mm | 用於手眼校準,需要列印 |
| **電腦** | Linux (Ubuntu 20.04+) + Docker | ROS Noetic 透過 `network_mode: host` 跑 |

**棋盤座標系**(robot base frame,單位:公尺):

```
BOARD_ORIGIN     = [0.30, -0.15, 0.02]   # a1 角的 XYZ
BOARD_SQUARE_SIZE = 0.05                  # 每格 5cm
```

把這幾個常數調整成你的物理擺設(在 `services/robot/main.py` 頭部)。

---

## 首次啟動流程

整套校準流程依序為:

```text
1. 連接硬體
       │
       ▼
2. 相機內參 (camera intrinsics) ── 可直接用 RealSense 出廠值
       │
       ▼
3. AprilTag 手眼校準 ──▶ camera_robot_calibration.yaml
       │
       ▼
4. Board Surface Z 校準 ──▶ board_surface_z.yaml
       │
       ▼
5. 64 格子教學(教 4 角內插即可)──▶ square_positions.yaml
       │
       ▼
6. 教 work / vision waypoint ──▶ waypoints.yaml
       │
       ▼
7. 開始下棋
```

每一步在 Web UI 的 **Robot tab** 都有對應按鈕。

---

## 視覺 6 階段管線

實作在 `chess_vision/vision_pipeline.py:analyze_image()`,輸入是一張 RGB image,輸出是 `BoardStateResult` 物件。

```text
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: 棋盤分割 (segment_chessboard)                        │
│   YOLO segmentation → binary mask of board                    │
├──────────────────────────────────────────────────────────────┤
│ Stage 2: 9×9 角點抽取 (grid_from_mask_and_image)              │
│   mask → 4 corners → homography → 9×9 grid corners            │
├──────────────────────────────────────────────────────────────┤
│ Stage 3: 棋子偵測 (detect_pieces)                             │
│   YOLO detection on full image → bbox + colour                │
│   分類器吐出 white_piece / black_piece (89/255 threshold)     │
├──────────────────────────────────────────────────────────────┤
│ Stage 4: 棋子 → 格子映射 (map_pieces_to_squares)               │
│   bbox bottom-center 透過 homography 投影到 8×8 棋盤格         │
├──────────────────────────────────────────────────────────────┤
│ Stage 5: 結果封裝 (BoardStateResult)                           │
│   board_state, grid_9x9, orientation, image_shape             │
├──────────────────────────────────────────────────────────────┤
│ Stage 6: 輸出格式 (FEN / ASCII / LLM-friendly)                 │
│   get_fen() / get_ascii_board() / get_llm_description()       │
└──────────────────────────────────────────────────────────────┘
```

**關鍵常數**(chess_vision/chess_algo.py 頭部):

| 常數 | 預設值 | 用途 |
|------|--------|------|
| `DETECT_MODEL` | `runs/detect/.../best.pt` | YOLO 偵測模型 |
| `SEGMENT_MODEL` | `runs/segment/.../best.pt` | YOLO 分割模型 |
| `DETECT_CONF` | 0.25 | 偵測信心門檻 |
| `WARP_SIZE` | 512 | 透視變換後的影像大小 |

**顏色分類**:每顆 bbox 取中央區域,計算平均亮度 vs 89/255 門檻,> threshold 為 white,反之為 black。這個門檻是用 249 張樣本實測校準出來的。

---

## AprilTag 手眼校準

目的:求出 **camera → robot** 的剛體變換 (4×4 matrix),讓 Vision 看到的座標可以轉成 Robot 能執行的座標。

### 校準方法

實作在 `chess_vision/handeye_calibration.py`,使用 OpenCV 的 `cv2.calibrateHandEye()`,支援 5 種演算法:

- **TSAI**(預設)
- PARK
- HORAUD
- ANDREFF
- DANIILIDIS

### 工作流程

1. 把 AprilTag(tag36h11,45mm)貼到夾爪末端或其他 EE-fixed 位置。
2. 在 Robot tab 點 **Add Calibration Sample**,系統會:
   - 用 `vision/detect/apriltag` 偵測 tag,得到相機座標 (rvec, tvec)
   - 透過 robot FK 取得目前 EE 在 robot base 的 (R, t)
   - 存成一筆樣本
3. 移動手臂到不同姿態,重複 **至少 3 次**(5–10 次效果更好)。
4. 點 **Compute Calibration**,系統用 SVD 解最小平方:

```python
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,   # 來自 robot FK
    R_target2cam, t_target2cam,       # 來自 AprilTag 偵測
    method=cv2.CALIB_HAND_EYE_TSAI,
)
```

### 校準結果格式

存到 `hand_eye_calibration.yaml`:

```yaml
rotation:
  - [0.99989, -0.00765, 0.01281]
  - [0.00770,  0.99996, -0.00415]
  - [-0.01278, 0.00425, 0.99991]
translation:
  - -0.6255
  - 0.6127
  - 0.7912
method: TSAI
reprojection_error: 0.00342   # 公尺
num_samples: 8
```

**驗收標準**:reprojection_error < 5mm 算可用,< 2mm 算很好。誤差超過 1cm 就要重新校準。

---

## 64 格子教學

每一格的 (x, y, z, pitch) 用來決定夾爪該怎麼下到那格上方。完整教 64 格很煩,所以提供 **4 角內插** 模式。

### 三種教學方式

#### 方式 A:逐格手動記錄

最精確但最費工:

1. Jog 手臂到目標格子正上方
2. `POST /square_positions/record` `{ "square": "e2" }`
3. 系統讀取 `robot.arm.get_ee_pose()` 的 (x, y, z, pitch)
4. 存到 `square_positions.yaml`

#### 方式 B:直接寫 API

如果已經有座標(例如從 CAD / 模擬算出):

```http
POST /square_positions/set
{
  "square": "e2",
  "x": -0.03711,
  "y":  0.41604,
  "z":  0.1276,
  "pitch": 0.96641
}
```

#### 方式 C:4 角內插(推薦)

只教 a1, a8, h1, h8 四個角,其餘 60 格自動算:

```http
POST /square_positions/interpolate
```

內插公式(雙線性):

```text
For each square (file f ∈ a..h, rank r ∈ 1..8):
  u = file_index / 7.0    # 0=a-file, 1=h-file
  v = rank_index / 7.0    # 0=rank1, 1=rank8

  bottom = val_a1 * (1-u) + val_h1 * u
  top    = val_a8 * (1-u) + val_h8 * u
  value  = bottom * (1-v) + top * v
```

四個角的 pitch 會被 **平均** 後套用到所有格子。X / Y / Z 各自雙線性內插。

### `square_positions.yaml` 格式

```yaml
e2:
  x: -0.03711
  y: 0.41604
  z: 0.1276
  pitch: 0.96641   # 弧度,約 55°
e4:
  x: -0.03711
  y: 0.31604
  z: 0.1276
  pitch: 0.96641
# ... 共 64 條
```

---

## Board Surface Z 校準

這是 **棋盤平面的 Z 高度**,用來算 grasp 高度時的基準(因為 64 個 square positions 的 Z 是夾爪預設的 hover 高度,不是棋盤本身的高度)。

```http
POST /board_surface_z/record
```

操作:把夾爪輕觸棋盤表面 → 呼叫 record → 系統存 `ee_pose[2,3]` 到 `board_surface_z.yaml`:

```yaml
board_surface_z: 0.04374   # 公尺
```

**為什麼需要這個**:看 [Grasp 高度計算](#grasp-高度計算) 那一節。

---

## IK 搜尋策略

5-DOF 手臂的工作空間有限,直接給 (x, y, z, pitch) 經常解不出 IK。系統的 fallback 策略是 **多角度搜尋**。

### 距離分區策略

`services/robot/main.py:find_reachable_pitch()`:

```python
distance = sqrt(x² + y²)   # 從 base 到目標的水平距離

if distance <= 0.20:
    candidates = [90°, 80°, 70°, 60°]      # 近距離,陡向下
elif distance <= 0.30:
    candidates = [80°, 70°, 60°, 55°, 45°]
else:
    candidates = [55°, 45°, 60°, 70°]      # 遠距離,平向前
```

對每個候選 pitch:

```python
result = robot.arm.set_ee_pose_components(
    x, y, z, pitch,
    blocking=False, execute=False    # 只解 IK,不執行
)
if result.success:
    return pitch       # 第一個解得到的就用
```

### 容忍度搜尋(進階)

當 caller 指定 `pitch_tolerance` 時(例如 `/arm/move_to_xyz`),會在 preferred pitch 附近 ±45° 內 **每 5° 步進** 搜尋:

```python
PITCH_TOL = radians(45)
step = radians(5)
candidates = [preferred_pitch]
for k in range(1, int(PITCH_TOL/step)+1):
    candidates += [preferred_pitch + k*step,
                   preferred_pitch - k*step]
```

**找不到解的情況**:會記 warning 但不阻擋,改用第一個候選 pitch 硬執行(此時手臂可能達不到目標,需要 caller 自己處理)。

---

## Grasp 高度計算

實作在 `services/robot/main.py:manual_pick`(約 line 2738)。

### 公式

```python
board_z = board_surface_z if board_surface_z is not None else xyz[2]

# 不同棋子用不同 grasp ratio (夾哪個高度比例)
if piece_type == "knight":
    grasp_ratio = 0.35 if rank <= 3 else 0.25   # 馬比較矮,夾低一點
else:
    grasp_ratio = 0.75                           # 大部分夾子身上半段

grasp_z = board_z + piece_height * grasp_ratio

# 近端 (rank 1-3) 因手臂角度問題,需要額外抬一點
if rank == 1:
    grasp_z += 0.030    # +30mm
elif rank <= 3:
    grasp_z += 0.020    # +20mm
```

### 棋子預設高度

```python
PIECE_HEIGHTS = {
    "pawn":   0.035,
    "rook":   0.040,
    "knight": 0.060,
    "bishop": 0.055,
    "queen":  0.070,
    "king":   0.080,
}
```

可以在 `services/robot/main.py` 頂部修改,單位是 公尺。

### 深度感測補償

如果有 RealSense 深度,Vision 端的 `chess_vision/depth_utils.py:compute_piece_heights()` 會:

1. 對每顆棋子的 bbox 取上方中央 30% 區域
2. 用 **第 10 百分位深度** 當棋子頂端(避免噪聲)
3. 對空格採樣中位數當棋盤表面深度
4. `piece_height = board_z - piece_z`,夾在 [0.5cm, 15cm]

這個動態量測會回傳給 Robot service,讓 grasp 高度根據實際棋子高度調整,不必死信 `PIECE_HEIGHTS` 表。

---

## 吃子與升變處理

### 吃子

`POST /move {"uci_move": "e4d5", "fen": "..."}` 偵測到 `board.piece_at(to_square) is not None` 時,流程:

```text
1. 移到被吃棋子上方            (square_positions[d5])
2. Pick 被吃的棋子
3. 移到 capture zone           (waypoint with tag "capture_zone")
4. Drop 被吃的棋子
5. 移到原本要走的棋子           (square_positions[e4])
6. Pick + 移到 d5 + Place
```

特殊情況:

- **En passant**:被吃的兵在 `to_square` 後方一格(由 `compute_en_passant_pawn_square()` 算)
- **Castling**:王走兩格 + 城堡跳過王(內建偵測)

`capture_zone` 不是檔案,而是 `waypoints.yaml` 裡 tag 為 `capture_zone` 的 waypoint。

### 升變

當 `move.promotion` 不為 None(預設升變成 Queen):

```text
1. Pick 兵 (待升變那一格)
2. 移到 capture zone, drop 兵
3. 移到 promotion_queen_position  (額外備用 Queen 的位置)
4. Pick 那顆 Queen
5. 移到目標格 (例如 e8), drop Queen
```

`promotion_queen_position.yaml` 需要事先用 `POST /promotion_queen/set` 教好。

---

## YAML 設定檔參考

| 檔名 | 用途 | 寫入端 | 讀取端 |
|------|------|--------|--------|
| `agent_settings.yaml` | LLM/STT/TTS 設定,UI 編輯 | API Settings 面板 | Agent boot |
| `characters.yaml` | 角色性格清單 | 手動編輯 | UI + Agent prompt |
| `waypoints.yaml` | 命名 + 標籤的關節姿態 | Robot tab teach | Robot move |
| `square_positions.yaml` | 64 格的 (x,y,z,pitch) | Robot tab teach / interpolate | manual_pick / move |
| `board_surface_z.yaml` | 棋盤表面 Z 高度 | Robot tab teach | grasp 計算 |
| `camera_robot_calibration.yaml` | 5+ 點對加 4×4 transform | Vision/Robot 校準流程 | 座標系轉換 |
| `hand_eye_calibration.yaml` | 3×3 R + 3×1 t (camera→gripper) | hand-eye calibration | 上層校準 |
| `gestures.yaml` | 自訂手勢 frame 序列 | UI 錄製 | 表演用 |

**注意**:這些 YAML 都是 **gitignored**,因為內容是各人物理擺設特有的。從 GitHub clone 下來的人需要自己跑校準/教學流程產生這些檔案。

### 關鍵 YAML 範例

**`waypoints.yaml`**:

```yaml
work:
  joints:
    waist: 1.776
    shoulder: -0.892
    elbow: 1.234
    wrist_angle: 0.557
    wrist_rotate: 0.0
  gripper: 0.5
  tag: work_position        # ← 用 tag 取代獨立的 work_position.yaml
  timestamp: 2026-04-25T15:42:00
vision:
  joints: { ... }
  tag: vision_position
capture_zone:
  joints: { ... }
  tag: capture_zone
```

**`camera_robot_calibration.yaml`**:

```yaml
points_camera: [[x1, y1, z1], [x2, y2, z2], ...]   # 相機座標
points_robot:  [[x1, y1, z1], [x2, y2, z2], ...]   # 對應的機器人座標
transform:                                          # 4×4 rigid transform
  - [r11, r12, r13, tx]
  - [r21, r22, r23, ty]
  - [r31, r32, r33, tz]
  - [0,   0,   0,   1]
timestamp: 2026-04-25T16:00:00
```

---

## REST API 契約

完整 API 列表很長,這裡列 **Agent 會直接呼叫的關鍵 endpoint**。其他全部在 `services/{vision,robot}/main.py` 的 `@app.post()` / `@app.get()` decorator 裡。

### Vision Service (`localhost:8001`)

| Endpoint | Method | Request | Response 重點欄位 |
|----------|--------|---------|--------------------|
| `/health` | GET | — | `{status, camera_connected, pipeline_ready}` |
| `/capture` | POST | — | `{success, fen, ascii_board, piece_positions, depth_info}` |
| `/capture/occupancy` | POST | — | `{success, occupied_squares, piece_positions, shots}` |
| `/capture/occupancy_init` | POST | — | `{success, occupied_squares, white_side: "bottom"`\|`"top"}` |
| `/detect/apriltag` | POST | `{return_image: bool}` | `{success, detections: [{tag_id, tvec, rvec}]}` |
| `/calibration/detect_tags` | POST | `{return_image: bool}` | `{success, points: [{tag_id, position, pixel}]}` |
| `/camera/stream/yolo` | GET | — | MJPEG (帶偵測 box) |

### Robot Service (`localhost:8002`)

**Connection**

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/health` | GET | — | `{status, robot_connected}` |
| `/connect` | POST | `{mock: bool}` | `{success, connected, mock}` |
| `/disconnect` | POST | — | `{success, connected}` |

**Chess moves**

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/move` | POST | `{uci_move: "e2e4", fen}` | `{success, was_capture, debug}` |
| `/manual_pick` | POST | `{square, piece_type}` | `{success, message}` |
| `/manual_place` | POST | `{square, piece_type}` | `{success, message}` |
| `/pickup_from_promotion_queen` | POST | `{piece_type: "queen"}` | `{success, message}` |

**Arm motion**

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/arm/work` | POST | — | `{success}` |
| `/arm/vision` | POST | — | `{success}` |
| `/arm/home` | POST | — | `{success}` |
| `/arm/sleep` | POST | — | `{success}` |
| `/arm/move_to_xyz` | POST | `{x, y, z, pitch, pitch_tolerance, moving_time}` | `{success, error}` |
| `/arm/positions` | GET | — | `{joints: {...}, ee_pose: {...}}` |

**Teaching / calibration**

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/square_positions/record` | POST | `{square}` | `{success, position}` |
| `/square_positions/interpolate` | POST | — | `{success, count: 64}` |
| `/board_surface_z/record` | POST | — | `{success, board_surface_z}` |
| `/calibration/add_point` | POST | `{camera_point, tag_id}` | `{success, num_points}` |
| `/calibration/compute` | POST | — | `{success, transform: 4×4}` |
| `/waypoints/save` | POST | `{name, tag}` | `{success, waypoint}` |
| `/waypoints/load/{name}` | POST | — | `{success}` |

**Gripper**

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/gripper/open` | POST | — | `{success}` |
| `/gripper/close` | POST | — | `{success}` |
| `/gripper/set` | POST | `{position: 0..1}` | `{success}` |

完整 schema 看 `services/robot/main.py` 頂部的 Pydantic models 區塊。

---

## 接到自己的機器人

整套架構不綁 RX-200。如果你想接 UR5、DOBOT、自組臂,只要實作下面這些 endpoint,Agent 跟 Vision 不必動:

### 最小可行契約

```text
GET  /health                        → {status, robot_connected}
POST /connect                       → {success, connected}

POST /arm/work                      → {success}        # 安全工作位
POST /arm/vision                    → {success}        # 給相機看的位置
POST /arm/move_to_xyz   {x,y,z,pitch}→ {success}        # 直接 XYZ 移動

POST /manual_pick   {square, piece_type} → {success}
POST /manual_place  {square, piece_type} → {success}
POST /move          {uci_move, fen}      → {success}   # 高層走法

POST /gripper/open                  → {success}
POST /gripper/close                 → {success}

POST /calibration/add_point  {camera_point, tag_id} → {success}
POST /calibration/compute            → {success, transform}
```

### 實作要點

1. **座標系**:所有 (x, y, z) 是 robot base frame,單位 公尺。Pitch / roll 弧度。
2. **IK**:你的硬體 SDK 必須能在 **不執行** 模式下檢查 IK 是否有解(類似 Interbotix 的 `set_ee_pose_components(execute=False)`)。沒有的話,你得自己實作 fallback 邏輯。
3. **Gripper 範圍**:在 main.py 頂部定義 `GRIPPER_CLOSED_M / GRIPPER_OPEN_M`,單位通常是 公尺(夾爪開度)。如果你的夾爪是 0–1 比例,自己對應一下。
4. **Square coordinates**:你可以選擇要不要支援 `square_positions.yaml`,或者在 `/manual_pick` 內部直接從 chess square 算 XYZ。
5. **Mock 模式**:`/connect {mock: true}` 應該讓 service 不真的連硬體,所有 endpoint 回 success — 方便沒手臂時做開發。

### 可以省略的部分

- ROS:只有 RX-200 用 ROS Noetic,因為 Interbotix SDK 是 ROS-based。其他臂直接用 Python SDK 就好,不必扛 ROS。
- AprilTag 校準:可以省,前提是 camera→robot 變換你用其他方式校。
- 64 格子教學:可以省,前提是 `/manual_pick` 內部自己算 XYZ。

---

## 常見問題

**Q: 為什麼 manual_pick 失敗?**
- 先確認 `square_positions.yaml` 有那個 square
- 看 robot 容器的 log,通常是 IK 解不到 — 試試擴大 pitch_tolerance,或重新教 4 個角讓內插值在工作空間內
- 確認 `/dev/ttyUSB0` 權限 (`ls -l /dev/ttyUSB0` 看是不是 dialout group)

**Q: AprilTag 偵測不到?**
- tag36h11 family、45mm 大小、印 **黑白** (彩印失敗率高)
- 光源要均勻,不能反光
- 相機距離 tag 0.3–0.8m 最穩
- 確認 `pupil-apriltags` 有裝(`apriltag_detector.py` 會 fallback,但 `pupil-apriltags` 速度最快)

**Q: 校準誤差很大?**
- 加更多 sample(8–10 個),姿態要分散(不要全部都很相似)
- 重新確認相機內參 (`camera_intrinsics_chess.yaml`)
- Reprojection error 應該 < 5mm

**Q: 夾爪頂到棋子但抓不起來?**
- 重新教 `board_surface_z`(`POST /board_surface_z/record`)
- 確認 `PIECE_HEIGHTS[piece_type]` 對你的棋子是正確的
- 看 robot log 裡的 grasp_z 是不是合理(應該大概是 board_z + 0.5×piece_height)

**Q: 手臂走到一半卡住?**
- 通常是中途某個 waypoint IK 不通
- 用 `/arm/jog/joint` 試手動轉,看哪個關節碰到限位
- `dynamixel motor reboot`(`POST /motors/reboot/wrist`)有時可以救過熱保護

---

## 相關文件

- [README (中文)](../README.zh-TW.md) — 專案總覽
- `services/vision/main.py` — Vision FastAPI 完整實作
- `services/robot/main.py` — Robot FastAPI 完整實作
- `chess_vision/vision_pipeline.py` — 視覺管線
- `chess_vision/handeye_calibration.py` — 手眼校準演算法

---

MIT License
