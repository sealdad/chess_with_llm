import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import yaml

from interbotix_xs_modules.arm import InterbotixManipulatorXS

# Load hand-eye calibration results
def load_hand_eye_calibration():
    with open('hand_eye_calibration.yaml', 'r') as f:
        calib_data = yaml.safe_load(f)
    R_cam2grip = np.array(calib_data['R_cam2gripper'])
    t_cam2grip = np.array(calib_data['t_cam2gripper']).reshape(3)
    return R_cam2grip, t_cam2grip

# Transform camera coordinates to robot coordinates
def camera_to_robot_coords(camera_coords, R_cam2grip, t_cam2grip):
    # Convert camera coordinates to robot coordinates
    robot_coords = R_cam2grip @ camera_coords + t_cam2grip
    return robot_coords

# 初始化 Interbotix RX-200 操作對象（透過ROS2與機械臂連線）
# 這會啟動後台的 ROS2 節點以準備接受運動指令&#8203;:contentReference[oaicite:13]{index=13}
try:
    bot = InterbotixManipulatorXS("rx200", "arm", "gripper")  # 建立機械臂控制實例&#8203;:contentReference[oaicite:14]{index=14}
except Exception as e:
    print("機械臂控制接口初始化失敗，請確認ROS 2環境是否啟動: ", e)
    raise

# Load hand-eye calibration
try:
    R_cam2grip, t_cam2grip = load_hand_eye_calibration()
except Exception as e:
    print("無法載入手眼校正結果: ", e)
    raise

# 建立主視窗
root = tk.Tk()
root.title("Interbotix RX-200 控制面板")

# Create notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(padx=10, pady=5, fill='both', expand=True)

# Create frames for each tab
robot_frame = ttk.Frame(notebook)
camera_frame = ttk.Frame(notebook)
notebook.add(robot_frame, text='機器人座標')
notebook.add(camera_frame, text='相機座標')

# Robot coordinates tab
tk.Label(robot_frame, text="X座標 (m):").grid(row=0, column=0, padx=5, pady=5)
robot_x_entry = tk.Entry(robot_frame, width=10)
robot_x_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(robot_frame, text="Y座標 (m):").grid(row=1, column=0, padx=5, pady=5)
robot_y_entry = tk.Entry(robot_frame, width=10)
robot_y_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Label(robot_frame, text="Z座標 (m):").grid(row=2, column=0, padx=5, pady=5)
robot_z_entry = tk.Entry(robot_frame, width=10)
robot_z_entry.grid(row=2, column=1, padx=5, pady=5)

# Camera coordinates tab
tk.Label(camera_frame, text="X座標 (m):").grid(row=0, column=0, padx=5, pady=5)
camera_x_entry = tk.Entry(camera_frame, width=10)
camera_x_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(camera_frame, text="Y座標 (m):").grid(row=1, column=0, padx=5, pady=5)
camera_y_entry = tk.Entry(camera_frame, width=10)
camera_y_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Label(camera_frame, text="Z座標 (m):").grid(row=2, column=0, padx=5, pady=5)
camera_z_entry = tk.Entry(camera_frame, width=10)
camera_z_entry.grid(row=2, column=1, padx=5, pady=5)

def move_to_robot_position():
    """當按下移動按鈕時調用，讀取輸入的機器人座標並控制機械臂移動"""
    try:
        # 從輸入框獲取座標並轉換為浮點數
        x = float(robot_x_entry.get())
        y = float(robot_y_entry.get())
        z = float(robot_z_entry.get())
    except ValueError:
        # 若輸入無效，提示使用者
        messagebox.showerror("輸入錯誤", "請輸入有效的數值座標 (X, Y, Z)！")
        return

    # 呼叫機械臂API移動到指定位置 (以 base_link 為座標系)&#8203;:contentReference[oaicite:15]{index=15}
    # 指定 roll=0, pitch=0, yaw 不指定 (RX-200 無第6軸yaw)&#8203;:contentReference[oaicite:16]{index=16}
    try:
        joints, success = bot.arm.set_ee_pose_components(x=x, y=y, z=z)
    except Exception as e:
        messagebox.showerror("運動錯誤", f"發送運動指令時出現錯誤: {e}")
        return

    if not success:
        # 若逆運動學無解或超出範圍，提示使用者
        messagebox.showerror("無法到達", f"機械臂無法到達座標 ({x:.2f}, {y:.2f}, {z:.2f})，請檢查座標是否在工作範圍內。")
    else:
        # 運動成功，提示完成
        messagebox.showinfo("執行完成", f"機械臂已移動至座標 ({x:.2f}, {y:.2f}, {z:.2f})。")

def move_to_camera_position():
    """當按下移動按鈕時調用，讀取輸入的相機座標，轉換為機器人座標後控制機械臂移動"""
    try:
        # 從輸入框獲取相機座標並轉換為浮點數
        cam_x = float(camera_x_entry.get())
        cam_y = float(camera_y_entry.get())
        cam_z = float(camera_z_entry.get())
    except ValueError:
        messagebox.showerror("輸入錯誤", "請輸入有效的數值座標 (X, Y, Z)！")
        return

    # 轉換相機座標為機器人座標
    camera_coords = np.array([cam_x, cam_y, cam_z])
    robot_coords = camera_to_robot_coords(camera_coords, R_cam2grip, t_cam2grip)
    
    try:
        joints, success = bot.arm.set_ee_pose_components(
            x=robot_coords[0],
            y=robot_coords[1],
            z=robot_coords[2]
        )
    except Exception as e:
        messagebox.showerror("運動錯誤", f"發送運動指令時出現錯誤: {e}")
        return

    if not success:
        messagebox.showerror("無法到達", 
            f"機械臂無法到達轉換後的座標 ({robot_coords[0]:.2f}, {robot_coords[1]:.2f}, {robot_coords[2]:.2f})，"
            f"請檢查座標是否在工作範圍內。")
    else:
        messagebox.showinfo("執行完成", 
            f"相機座標 ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) 已轉換為機器人座標 "
            f"({robot_coords[0]:.2f}, {robot_coords[1]:.2f}, {robot_coords[2]:.2f})，"
            f"機械臂已移動至目標位置。")

# Create buttons for each tab
robot_move_button = tk.Button(robot_frame, text="移動", command=move_to_robot_position)
robot_move_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

camera_move_button = tk.Button(camera_frame, text="移動", command=move_to_camera_position)
camera_move_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

# 啟動Tkinter事件循環
root.mainloop()

