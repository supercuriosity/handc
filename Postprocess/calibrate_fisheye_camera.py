import cv2
import numpy as np
import os
import glob

# =================================================================================
# 1. 定义棋盘格和标定参数
# =================================================================================
# 棋盘格内部角点的数量 (corners = (cols-1, rows-1))
CHECKERBOARD = (6, 9) # 例如，一个 7x10 的棋盘格，内部角点是 6x9
# 每个方格的物理尺寸（例如，毫米）。
square_size = 25.0
# 标定所需的最少有效帧数
MIN_FRAMES = 15

# =================================================================================
# 2. 准备对象点 (3D 世界坐标系中的点)
# =================================================================================
# 创建一个 (0,0,0), (1,0,0), ..., (8,5,0) 这样的坐标网格
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size # 乘以每个格子的实际大小

# =================================================================================
# 3. 从摄像头实时捕获图像并提取角点
# =================================================================================
# 存储所有图片的对象点和图像点
objpoints = [] # 3D点
imgpoints = [] # 2D点

# 打开默认摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误: 无法打开摄像头。请检查摄像头是否连接并正常工作。")
    exit()

print("\n摄像头已打开。请将棋盘格展示在摄像头前。")
print(f"说明: 按 's' 保存当前帧用于标定 (至少需要 {MIN_FRAMES} 帧), 按 'c' 开始标定, 按 'q' 退出。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法从摄像头读取帧。")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 寻找棋盘格角点
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    display_frame = frame.copy()
    
    # 如果找到了角点
    if ret_corners:
        # 绘制角点
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret_corners)

    # 在画面上显示提示信息
    text = f"Saved frames: {len(objpoints)}/{MIN_FRAMES}. Press 's' to save, 'c' to calibrate, 'q' to quit."
    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Camera Calibration', display_frame)
    
    key = cv2.waitKey(1) & 0xFF

    # 按 'q' 退出
    if key == ord('q'):
        print("程序退出。")
        break
    
    # 按 's' 保存帧
    elif key == ord('s'):
        if ret_corners:
            print(f"帧 {len(objpoints)+1} 已保存: 成功找到角点!")
            objpoints.append(objp)
            # 提高角点检测的精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
        else:
            print("警告: 当前帧未找到角点，无法保存。")

    # 按 'c' 开始标定
    elif key == ord('c'):
        if len(objpoints) >= MIN_FRAMES:
            print(f"\n已收集 {len(objpoints)} 帧，开始进行鱼眼相机标定...")
            break # 退出循环，进入标定步骤
        else:
            print(f"错误: 标定需要至少 {MIN_FRAMES} 帧，当前只有 {len(objpoints)} 帧。请继续采集。")


# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()

if len(objpoints) < MIN_FRAMES:
    print("\n未收集到足够的帧进行标定，程序已终止。")
    exit()

# =================================================================================
# 4. 执行鱼眼相机标定
# =================================================================================
N_OK = len(objpoints)
K = np.zeros((3, 3)) # 内参矩阵
D = np.zeros((4, 1)) # 畸变系数 (k1, k2, k3, k4)
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

# 获取最后一帧的图像尺寸用于标定
# 注意：这里假设所有帧的尺寸都一样
h, w = gray.shape[:2]

# 标定函数
# 注意：我们使用的是 cv2.fisheye.calibrate，而不是 cv2.calibrateCamera
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
# 注意：gray.shape[::-1] 已被替换为 (w, h)
ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    (w, h),
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print("\n标定完成！")
print(f"重投影误差 (Reprojection Error): {ret}")
print("="*30)
print(f"相机内参矩阵 K:\n{K}")
print(f"\n相机畸变系数 D:\n{D}")
print("="*30)

# =================================================================================
# 5. 保存标定结果
# =================================================================================
output_file = "fisheye_calibration_data.npz"
np.savez(output_file, camera_matrix=K, dist_coeffs=D, reprojection_error=ret)
print(f"\n标定结果已保存到 '{output_file}'")

# =================================================================================
# 6. (可选) 显示矫正后的效果
# =================================================================================
print("\n按任意键关闭预览窗口并结束程序。")
cap = cv2.VideoCapture(0) # 重新打开摄像头以显示效果
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 矫正图像
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w,h), cv2.CV_16SC2)
    undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 并排显示原图和矫正图
    combined_view = np.hstack((frame, undistorted_img))
    cv2.putText(combined_view, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_view, "Undistorted", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Original vs Undistorted", combined_view)

    if cv2.waitKey(1) != -1:
        break

cap.release()
cv2.destroyAllWindows()