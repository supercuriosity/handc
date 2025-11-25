"""Check the device camera availability and functionality.
"""

import os
import sys
import cv2
sys.path.append('/root/workspace/handcap')

from config import TACTILE_CAMERA, WRIST_CAMERA

def save_check_tactile():
    for cam_name, cam_path in TACTILE_CAMERA.items():
        cap = cv2.VideoCapture(cam_path)
        if not cap.isOpened():
            print(f"Error: Could not open tactile camera {cam_name} at {cam_path}")
            continue

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from tactile camera {cam_name} at {cam_path}")
            cap.release()
            continue
        
        height, width = frame.shape[:2]

        crop_width = int(width * 3 / 5)
        crop_height = int(height * 3 / 5)
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2

        colored_image = frame[start_y:start_y+crop_height, 
                                    start_x:start_x+crop_width]
        
        output_path = f"checked_images/check_tactile_{cam_name}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, colored_image)
        print(f"Saved frame from tactile camera {cam_name} to {output_path}")
        cap.release()

def save_check_wrist():
    for cam_name, cam_path in WRIST_CAMERA.items():
        cap = cv2.VideoCapture(cam_path)
        if not cap.isOpened():
            print(f"Error: Could not open wrist camera {cam_name} at {cam_path}")
            continue

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from wrist camera {cam_name} at {cam_path}")
            cap.release()
            continue

        output_path = f"checked_images/check_wrist_{cam_name}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
        print(f"Saved frame from wrist camera {cam_name} to {output_path}")
        cap.release()

def save_check_video_wrist():
    for cam_name, cam_path in WRIST_CAMERA.items():
        cap = cv2.VideoCapture(cam_path)
        if not cap.isOpened():
            print(f"Error: Could not open wrist camera {cam_name} at {cam_path}")
            continue

        # 获取视频尺寸
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30.0  # 设定帧率

        output_path = f"checked_images/check_wrist_{cam_name}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 初始化视频写入器 (使用 mp4v 编码)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Recording 3 seconds video for wrist camera {cam_name}...")

        # 录制约 3 秒 (90帧)
        for _ in range(600):
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame from wrist camera {cam_name} at {cam_path}")
                break
            out.write(frame)

        print(f"Saved video from wrist camera {cam_name} to {output_path}")
        cap.release()
        out.release()


def main():
    save_check_tactile()
    save_check_wrist()
    # save_check_video_wrist()

if __name__ == "__main__":
    main()


