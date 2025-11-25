import os
import cv2
import time
from pyzbar import pyzbar
import numpy as np

def run_qr_scanner(camera_index=4, output_filename="qr_scan_video.avi"):
    """
    Start a QR code scanner that reads time stamps from QR codes,
    displays the video feed with annotations, and saves the result to a video file.
    """
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print("错误：无法打开摄像头。请检查摄像头是否被其他程序占用。")
        return

    # --- 视频保存设置 ---
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20  # 为输出视频设置一个合适的帧率

    # 定义编解码器并创建 VideoWriter 对象
    # 对于 .avi 文件，'XVID' 是一个很好的选择
    # 对于 .mp4 文件，可以尝试 'mp4v' 或 'avc1'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, float(fps), (frame_width, frame_height))
    
    print(f"摄像头已启动，视频将保存为 '{output_filename}'")
    print("请将摄像头对准另一台设备上的时间二维码...")
    print("按 'q' 键退出程序并保存视频。")

    last_decoded_data = None
    
    total_time_difference = []
    
    
    while True:
        scanner_time_ms = int(time.time() * 1000)
        ret, frame = camera.read()
        
        output_path = f"checked_images/wrist_{camera_index}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if not ret:
            print("错误：无法读取视频帧。")
            break

        decoded_objects = pyzbar.decode(thresh)
        for obj in decoded_objects:
            qr_data_str = obj.data.decode('utf-8')
            
            if qr_data_str == last_decoded_data:
                continue
            last_decoded_data = qr_data_str

            try:
                generator_time_ms = int(qr_data_str)
                time_difference = scanner_time_ms - generator_time_ms
                
                # --- 在控制台打印结果 ---
                print("----------------------------------------")
                print(f"二维码解码成功")
                print(f"  生成器时间戳 (A): {generator_time_ms} ms")
                print(f"  扫描器时间戳 (B): {scanner_time_ms} ms")
                print(f"  时间差 (B - A): {time_difference} ms")
                print("----------------------------------------\n")

                # --- 在视频帧上绘制结果 ---
                points = obj.polygon
                if len(points) > 3:
                    # 将 pyzbar 返回的点转换为 NumPy 数组以供 cv2.polylines 使用
                    pts = np.array([(p.x, p.y) for p in points], dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

                text = f"Delay: {time_difference} ms"
                rect = obj.rect
                cv2.putText(frame, text, (rect.left, rect.top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            
            except ValueError:
                # 忽略无法转换为整数的二维码数据
                # passq
                print("Please move the camera to QR code with time stamp...")
            
            
            total_time_difference.append(time_difference)
            
            mean_delay = np.mean(total_time_difference)
            print(f"当前平均延迟: {mean_delay:.2f} ms")
            
            if len(total_time_difference) >= 100:
                exit()

        # 将处理后的帧写入输出视频文件
        out.write(frame)
        
        # 仍然显示窗口，以便用户可以看到正在录制的内容
        # cv2.imshow("Python QR Code Time Scanner (Recording...)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("正在关闭...")
    # 释放所有资源
    camera.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频已成功保存到 '{output_filename}'")

if __name__ == "__main__":
    # 调用函数，并可以指定输出文件名
    run_qr_scanner(output_filename="qr_scan_video.mp4")