[← Back to README](../README.md) ・ [繁體中文版本](ROBOT.zh-TW.md)

# Physical Robot Mode

Take chess off the screen and onto a real board. This document covers everything you need for the **physical RX-200 build** from scratch: hardware, vision pipeline, hand-eye calibration, 64-square teaching, IK strategy, capture/promotion implementation, every YAML config schema, the REST API contract, and the minimum contract for porting this stack to **any other robot arm**.

> The software isn't bound to RX-200. To wire in a UR5, DOBOT, or homebrew arm, just implement the few endpoints listed under [REST API Contract](#rest-api-contract) — the Agent and Vision services don't need to change.

---

## Contents

- [Architecture](#architecture)
- [Hardware](#hardware)
- [First-Time Setup Flow](#first-time-setup-flow)
- [The 6-Stage Vision Pipeline](#the-6-stage-vision-pipeline)
- [AprilTag Hand-Eye Calibration](#apriltag-hand-eye-calibration)
- [64-Square Teaching](#64-square-teaching)
- [Board Surface Z Calibration](#board-surface-z-calibration)
- [IK Search Strategy](#ik-search-strategy)
- [Grasp Height Calculation](#grasp-height-calculation)
- [Capture and Promotion](#capture-and-promotion)
- [YAML Config Reference](#yaml-config-reference)
- [REST API Contract](#rest-api-contract)
- [Porting to Your Own Robot](#porting-to-your-own-robot)
- [Troubleshooting](#troubleshooting)

---

## Architecture

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

- **Vision** photographs the physical board and outputs FEN + 3D coordinates of every piece.
- **Robot** turns UCI moves (e.g. `e2e4`) into pick/place/capture motions.
- **Agent** ties them together: read vision → identify the human's move → LLM/Stockfish picks the response → call Robot to execute.

Start it up:

```bash
docker-compose -f docker-compose.physical.yaml up --build
```

> The first robot image build takes ~30 minutes (it clones and compiles the entire Interbotix ROS workspace). The vision image is also slow first time (PyTorch + RealSense SDK). Subsequent runs hit the build cache and start in seconds.

---

## Hardware

| Component | Spec | Notes |
|-----------|------|-------|
| **Robot arm** | Interbotix RX-200 (5-DOF) | USB serial, usually `/dev/ttyUSB0` |
| **Camera** | Intel RealSense D4xx (D435 / D435i / D455) | RGB + depth, with alignment |
| **Chess board** | Standard 8×8 | Default 50mm per square (`BOARD_SQUARE_SIZE = 0.05`) |
| **Pieces** | Any chess pieces | Trained via YOLO; 35–80mm tall works best |
| **AprilTag** | tag36h11, 45mm | For hand-eye calibration; print and mount |
| **Computer** | Linux (Ubuntu 20.04+) + Docker | Robot service uses `network_mode: host` for ROS |

**Board coordinate system** (in robot base frame, metres):

```
BOARD_ORIGIN     = [0.30, -0.15, 0.02]   # XYZ of the a1 corner
BOARD_SQUARE_SIZE = 0.05                  # 5cm per square
```

Adjust these to match your physical setup at the top of `services/robot/main.py`.

---

## First-Time Setup Flow

The full calibration sequence:

```text
1. Connect hardware
       │
       ▼
2. Camera intrinsics ── you can use the RealSense factory values directly
       │
       ▼
3. AprilTag hand-eye calibration ──▶ camera_robot_calibration.yaml
       │
       ▼
4. Board Surface Z ──▶ board_surface_z.yaml
       │
       ▼
5. Teach 64 squares (4-corner interpolation is fine) ──▶ square_positions.yaml
       │
       ▼
6. Teach work / vision waypoints ──▶ waypoints.yaml
       │
       ▼
7. Start playing
```

Every step has a corresponding button in the **Robot tab** of the web UI.

---

## The 6-Stage Vision Pipeline

Implemented in `chess_vision/vision_pipeline.py:analyze_image()`. Input: an RGB image. Output: a `BoardStateResult` object.

```text
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: Board segmentation (segment_chessboard)              │
│   YOLO segmentation → binary mask of the board                │
├──────────────────────────────────────────────────────────────┤
│ Stage 2: 9×9 grid extraction (grid_from_mask_and_image)       │
│   mask → 4 corners → homography → 9×9 grid corners            │
├──────────────────────────────────────────────────────────────┤
│ Stage 3: Piece detection (detect_pieces)                      │
│   YOLO detection on full image → bbox + colour                │
│   Classifier outputs white_piece / black_piece (89/255 thr)   │
├──────────────────────────────────────────────────────────────┤
│ Stage 4: Piece → square mapping (map_pieces_to_squares)       │
│   Project bbox bottom-centre via homography to 8×8 squares    │
├──────────────────────────────────────────────────────────────┤
│ Stage 5: Result assembly (BoardStateResult)                   │
│   board_state, grid_9x9, orientation, image_shape             │
├──────────────────────────────────────────────────────────────┤
│ Stage 6: Output formatting (FEN / ASCII / LLM-friendly)       │
│   get_fen() / get_ascii_board() / get_llm_description()       │
└──────────────────────────────────────────────────────────────┘
```

**Key constants** (top of `chess_vision/chess_algo.py`):

| Constant | Default | Purpose |
|----------|---------|---------|
| `DETECT_MODEL` | `runs/detect/.../best.pt` | YOLO detection weights |
| `SEGMENT_MODEL` | `runs/segment/.../best.pt` | YOLO segmentation weights |
| `DETECT_CONF` | 0.25 | Detection confidence threshold |
| `WARP_SIZE` | 512 | Warped board image size |

**Colour classification**: each piece bbox is sampled in the centre region, mean luminance is compared against the 89/255 threshold. Above threshold = white, below = black. The threshold was empirically calibrated from 249 samples.

---

## AprilTag Hand-Eye Calibration

Goal: compute the **camera → robot** rigid-body transform (4×4 matrix) so that coordinates Vision sees can be turned into coordinates Robot can move to.

### Method

Implemented in `chess_vision/handeye_calibration.py` using OpenCV's `cv2.calibrateHandEye()`. Five algorithms supported:

- **TSAI** (default)
- PARK
- HORAUD
- ANDREFF
- DANIILIDIS

### Workflow

1. Mount the AprilTag (tag36h11, 45mm) on the gripper end-effector or another EE-fixed location.
2. In the Robot tab, click **Add Calibration Sample**. The system:
   - Calls `vision/detect/apriltag` → tag in camera frame (rvec, tvec)
   - Reads robot FK → EE pose in robot base frame (R, t)
   - Saves the pair as one sample
3. Move the arm to varied poses and repeat **at least 3 times** (5–10 is better).
4. Click **Compute Calibration**. The system runs SVD-based least squares:

```python
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,   # from robot FK
    R_target2cam, t_target2cam,       # from AprilTag detection
    method=cv2.CALIB_HAND_EYE_TSAI,
)
```

### Result Format

Saved to `hand_eye_calibration.yaml`:

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
reprojection_error: 0.00342   # metres
num_samples: 8
```

**Acceptance**: reprojection_error < 5mm is usable, < 2mm is great. Anything over 1cm needs to be redone.

---

## 64-Square Teaching

Each square needs its own (x, y, z, pitch) so the gripper knows how to descend onto it. Teaching all 64 manually is tedious, so the system supports **4-corner interpolation**.

### Three Teaching Methods

#### Method A: Per-square manual recording

Most accurate, most work:

1. Jog the arm to the target position above the square.
2. `POST /square_positions/record { "square": "e2" }`
3. The system reads `robot.arm.get_ee_pose()` → (x, y, z, pitch).
4. Stored in `square_positions.yaml`.

#### Method B: Direct API write

If you already have coordinates (e.g. computed from CAD or simulation):

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

#### Method C: 4-corner interpolation (recommended)

Teach only a1, a8, h1, h8; the other 60 are computed:

```http
POST /square_positions/interpolate
```

Bilinear interpolation formula:

```text
For each square (file f ∈ a..h, rank r ∈ 1..8):
  u = file_index / 7.0    # 0 = a-file, 1 = h-file
  v = rank_index / 7.0    # 0 = rank 1, 1 = rank 8

  bottom = val_a1 * (1-u) + val_h1 * u
  top    = val_a8 * (1-u) + val_h8 * u
  value  = bottom * (1-v) + top * v
```

Pitch from the four corners is **averaged** and applied to all squares. X / Y / Z are interpolated independently.

### `square_positions.yaml` schema

```yaml
e2:
  x: -0.03711
  y: 0.41604
  z: 0.1276
  pitch: 0.96641   # radians, ~55°
e4:
  x: -0.03711
  y: 0.31604
  z: 0.1276
  pitch: 0.96641
# ... 64 entries total
```

---

## Board Surface Z Calibration

This is the **height of the board's flat surface**. The 64 square positions store the gripper's hover height for each square — but to compute the actual grasp height, the system needs the board surface Z separately.

```http
POST /board_surface_z/record
```

Procedure: lower the gripper until it just touches the board → call the endpoint → the system saves `ee_pose[2,3]` to `board_surface_z.yaml`:

```yaml
board_surface_z: 0.04374   # metres
```

**Why this matters** — see [Grasp Height Calculation](#grasp-height-calculation).

---

## IK Search Strategy

A 5-DOF arm has a limited workspace; many (x, y, z, pitch) targets have no IK solution. The system's fallback is **multi-angle search**.

### Distance-based candidates

`services/robot/main.py:find_reachable_pitch()`:

```python
distance = sqrt(x² + y²)   # horizontal distance from base

if distance <= 0.20:
    candidates = [90°, 80°, 70°, 60°]      # close — steep down
elif distance <= 0.30:
    candidates = [80°, 70°, 60°, 55°, 45°]
else:
    candidates = [55°, 45°, 60°, 70°]      # far — angled forward
```

For each candidate pitch:

```python
result = robot.arm.set_ee_pose_components(
    x, y, z, pitch,
    blocking=False, execute=False    # IK check only, don't move
)
if result.success:
    return pitch       # use the first one that solves
```

### Tolerance search (advanced)

When the caller specifies `pitch_tolerance` (e.g. on `/arm/move_to_xyz`), the system searches around the preferred pitch in **5° steps up to ±45°**:

```python
PITCH_TOL = radians(45)
step = radians(5)
candidates = [preferred_pitch]
for k in range(1, int(PITCH_TOL/step)+1):
    candidates += [preferred_pitch + k*step,
                   preferred_pitch - k*step]
```

**No solution found**: a warning is logged but the move isn't blocked — it executes with the first candidate pitch. The caller may then have to handle a missed target.

---

## Grasp Height Calculation

Implemented in `services/robot/main.py:manual_pick` (around line 2738).

### Formula

```python
board_z = board_surface_z if board_surface_z is not None else xyz[2]

# Different pieces use different grasp ratios (where on the piece to grip)
if piece_type == "knight":
    grasp_ratio = 0.35 if rank <= 3 else 0.25   # knights are short
else:
    grasp_ratio = 0.75                           # most pieces grip upper half

grasp_z = board_z + piece_height * grasp_ratio

# Near-side ranks (1–3) need a bit of lift due to arm geometry
if rank == 1:
    grasp_z += 0.030    # +30mm
elif rank <= 3:
    grasp_z += 0.020    # +20mm
```

### Default piece heights

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

Edit at the top of `services/robot/main.py` if your pieces differ. Units are metres.

### Depth-based dynamic height

When RealSense depth is available, `chess_vision/depth_utils.py:compute_piece_heights()`:

1. For each piece bbox, samples the upper-centre 30%.
2. Uses the **10th percentile depth** as the piece top (rejects noise).
3. Samples empty squares with median to get the board surface depth.
4. `piece_height = board_z - piece_z`, clamped to [0.5cm, 15cm].

This dynamic measurement is sent to Robot, so the grasp height adapts to actual piece heights instead of trusting `PIECE_HEIGHTS` blindly.

---

## Capture and Promotion

### Captures

When `POST /move {"uci_move": "e4d5", "fen": "..."}` detects `board.piece_at(to_square) is not None`, the flow is:

```text
1. Move to above the captured piece    (square_positions[d5])
2. Pick the captured piece
3. Move to capture zone                (waypoint with tag "capture_zone")
4. Drop the captured piece
5. Move to the moving piece            (square_positions[e4])
6. Pick + move to d5 + place
```

Special cases:

- **En passant** — the captured pawn is on the rank behind `to_square` (computed by `compute_en_passant_pawn_square()`).
- **Castling** — king moves two squares; rook jumps over (built-in detection).

`capture_zone` isn't a separate file — it's a waypoint in `waypoints.yaml` tagged `capture_zone`.

### Promotion

When `move.promotion` is set (defaults to Queen):

```text
1. Pick the pawn (the one being promoted)
2. Move to capture zone, drop the pawn
3. Move to promotion_queen_position    (the spare Queen's parking spot)
4. Pick the spare Queen
5. Move to the destination square (e.g. e8), drop the Queen
```

`promotion_queen_position.yaml` must be taught beforehand via `POST /promotion_queen/set`.

---

## YAML Config Reference

| File | Purpose | Written by | Read by |
|------|---------|-----------|---------|
| `agent_settings.yaml` | LLM/STT/TTS settings, edited via UI | API Settings panel | Agent boot |
| `characters.yaml` | Character preset list | Manual edit | UI + Agent prompts |
| `waypoints.yaml` | Named, tagged joint poses | Robot tab teach | Robot motion |
| `square_positions.yaml` | (x,y,z,pitch) for 64 squares | Robot tab teach / interpolate | manual_pick / move |
| `board_surface_z.yaml` | Board surface Z height | Robot tab teach | Grasp calc |
| `camera_robot_calibration.yaml` | 5+ point pairs + 4×4 transform | Vision/Robot calibration flow | Frame conversion |
| `hand_eye_calibration.yaml` | 3×3 R + 3×1 t (camera → gripper) | Hand-eye calibration | Upper-level calibration |
| `gestures.yaml` | Custom gesture frame sequences | UI recording | Performance |

**Note**: these YAMLs are all **gitignored** because their contents are specific to your physical setup. Anyone cloning from GitHub will need to run their own calibration/teaching to produce them.

### Sample YAML Files

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
  tag: work_position        # ← tag-based; supersedes a separate work_position.yaml
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
points_camera: [[x1, y1, z1], [x2, y2, z2], ...]   # camera-frame coords
points_robot:  [[x1, y1, z1], [x2, y2, z2], ...]   # corresponding robot-frame coords
transform:                                          # 4×4 rigid transform
  - [r11, r12, r13, tx]
  - [r21, r22, r23, ty]
  - [r31, r32, r33, tz]
  - [0,   0,   0,   1]
timestamp: 2026-04-25T16:00:00
```

---

## REST API Contract

The full API surface is large; below are the **key endpoints the Agent calls**. Everything else is in the `@app.post()` / `@app.get()` decorators of `services/{vision,robot}/main.py`.

### Vision Service (`localhost:8001`)

| Endpoint | Method | Request | Response (key fields) |
|----------|--------|---------|------------------------|
| `/health` | GET | — | `{status, camera_connected, pipeline_ready}` |
| `/capture` | POST | — | `{success, fen, ascii_board, piece_positions, depth_info}` |
| `/capture/occupancy` | POST | — | `{success, occupied_squares, piece_positions, shots}` |
| `/capture/occupancy_init` | POST | — | `{success, occupied_squares, white_side: "bottom"`\|`"top"}` |
| `/detect/apriltag` | POST | `{return_image: bool}` | `{success, detections: [{tag_id, tvec, rvec}]}` |
| `/calibration/detect_tags` | POST | `{return_image: bool}` | `{success, points: [{tag_id, position, pixel}]}` |
| `/camera/stream/yolo` | GET | — | MJPEG stream with detection boxes |

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

Full schemas live in the Pydantic models at the top of `services/robot/main.py`.

---

## Porting to Your Own Robot

The architecture isn't tied to RX-200. To wire in a UR5, DOBOT, or your own arm, implement these endpoints — Agent and Vision don't change:

### Minimum Viable Contract

```text
GET  /health                        → {status, robot_connected}
POST /connect                       → {success, connected}

POST /arm/work                      → {success}        # safe work position
POST /arm/vision                    → {success}        # camera-viewing position
POST /arm/move_to_xyz   {x,y,z,pitch}→ {success}        # direct XYZ move

POST /manual_pick   {square, piece_type} → {success}
POST /manual_place  {square, piece_type} → {success}
POST /move          {uci_move, fen}      → {success}   # high-level move

POST /gripper/open                  → {success}
POST /gripper/close                 → {success}

POST /calibration/add_point  {camera_point, tag_id} → {success}
POST /calibration/compute            → {success, transform}
```

### Implementation Notes

1. **Frames**: all (x, y, z) are in robot base frame, metres. Pitch / roll in radians.
2. **IK**: your hardware SDK needs an "IK check without execute" mode (like Interbotix's `set_ee_pose_components(execute=False)`). If it doesn't have one, you'll need to implement the fallback search yourself.
3. **Gripper range**: define `GRIPPER_CLOSED_M / GRIPPER_OPEN_M` at the top of `main.py`, typically in metres (gripper opening). If your gripper is 0–1 normalized, map appropriately.
4. **Square coordinates**: optional — you can support `square_positions.yaml`, or compute XYZ from the chess square inside `/manual_pick`.
5. **Mock mode**: `/connect {mock: true}` should let the service skip real-hardware calls and return `success` from every endpoint — handy for development without an arm attached.

### What You Can Skip

- **ROS** — only the RX-200 build uses ROS Noetic, because Interbotix's SDK is ROS-based. Other arms with native Python SDKs don't need ROS at all.
- **AprilTag calibration** — skip if you have another way to compute camera→robot transform.
- **64-square teaching** — skip if `/manual_pick` computes XYZ from the chess square internally.

---

## Troubleshooting

**Q: `manual_pick` keeps failing.**
- Check `square_positions.yaml` actually contains that square.
- Check the robot container logs — usually it's an IK failure. Try increasing `pitch_tolerance` or re-teach the four corners so the interpolated values land in the workspace.
- Confirm `/dev/ttyUSB0` permissions (`ls -l /dev/ttyUSB0` — should be in `dialout` group).

**Q: AprilTag isn't being detected.**
- tag36h11 family, 45mm size, **black-and-white** print (colour print fails often).
- Even lighting, no glare.
- Camera distance 0.3–0.8m from the tag is most reliable.
- Make sure `pupil-apriltags` is installed (`apriltag_detector.py` falls back to others, but `pupil-apriltags` is fastest).

**Q: Calibration error is too large.**
- Add more samples (8–10), with varied poses (don't take all of them from similar angles).
- Verify the camera intrinsics (`camera_intrinsics_chess.yaml`).
- Reprojection error should be < 5mm.

**Q: Gripper hits the piece but can't grip it.**
- Re-teach `board_surface_z` (`POST /board_surface_z/record`).
- Verify `PIECE_HEIGHTS[piece_type]` matches your actual pieces.
- Check the robot log for the chosen `grasp_z` — should be roughly `board_z + 0.5×piece_height`.

**Q: Arm freezes mid-motion.**
- Usually one of the intermediate waypoints has no IK solution.
- Use `/arm/jog/joint` to manually rotate joints and find which one's hitting the limit.
- Dynamixel motor reboot (`POST /motors/reboot/wrist`) sometimes recovers from over-temperature shutdowns.

---

## Related Docs

- [README](../README.md) — project overview
- `services/vision/main.py` — full Vision FastAPI implementation
- `services/robot/main.py` — full Robot FastAPI implementation
- `chess_vision/vision_pipeline.py` — vision pipeline
- `chess_vision/handeye_calibration.py` — hand-eye calibration algorithm

---

MIT License
