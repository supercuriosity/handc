import os
import cv2
import time
import datetime  # <-- 1. 引入 datetime 模块
from pyzbar import pyzbar
import numpy as np

def reconstruct_timestamp(simplified_time_str: str) -> datetime.datetime:
    """
    从新的简化时间字符串（HHMMSSsss）重构出完整的datetime对象。
    
    Args:
        simplified_time_str: 从二维码解码出的字符串，格式应为 "时时分分秒秒毫毫毫"。

    Returns:
        一个完整的datetime对象，如果输入格式错误则返回None。
    """
    # 2. 检查数据格式是否为9位数字
    if not simplified_time_str.isdigit() or len(simplified_time_str) != 9:
        return None  # 不是我们期望的格式

    try:
        # 3. 获取解码这一刻的“不变”信息（年、月、日）
        now = datetime.datetime.now()

        # 4. 从解码的字符串中解析出 "HHMMSSsss"
        hh = int(simplified_time_str[0:2])
        mm = int(simplified_time_str[2:4])
        ss = int(simplified_time_str[4:6])
        ms = int(simplified_time_str[6:9])
        us = ms * 1000  # 转换为微秒

        # 5. 使用.replace()方法，将完整的时间“嫁接”到当前日期上
        reconstructed_dt = now.replace(hour=hh, minute=mm, second=ss, microsecond=us)

        # 6. 处理跨日的边界情况 (例如，在 00:00:01 解码了前一天 23:59:59 生成的码)
        # 如果重构出的时间比当前时间晚了超过12小时，那么它很可能属于前一天。
        if reconstructed_dt > now and (reconstructed_dt - now).total_seconds() > 12 * 3600:
             reconstructed_dt -= datetime.timedelta(days=1)
            
        return reconstructed_dt

    except ValueError:
        return None # 字符串中的数字无效 (e.g., "999999999")

def run_qr_scanner(camera_index=4, output_filename="qr_scan_video.mp4"): # <-- 改为 .mp4
    """
    启动一个QR码扫描器，它能从简化的时间码中重构完整时间戳，
    计算延迟，显示带注释的视频，并将结果保存到视频文件。
    """
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print("错误：无法打开摄像头。请检查摄像头是否被其他程序占用。")
        return

    # --- 视频保存设置 ---
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # 20-30 帧是一个合理的范围

    # 定义编解码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 'mp4v' 适用于 .mp4
    out = cv2.VideoWriter(output_filename, fourcc, float(fps), (frame_width, frame_height))
    
    print(f"摄像头已启动，视频将保存为 '{output_filename}'")
    print("请将摄像头对准另一台设备上的时间二维码...")
    print("按 'q' 键退出程序并保存视频。")
    
    last_decoded_data = None
    total_time_difference = []
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("错误：无法读取视频帧。")
            break

        output_path = f"checked_images/wrist_{camera_index}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)

        # 图像预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        decoded_objects = pyzbar.decode(thresh)
        
        # 默认文本，当没有有效二维码时显示
        display_text = ""

        for obj in decoded_objects:
            qr_data_str = obj.data.decode('utf-8')
            
            if qr_data_str == last_decoded_data:
                continue
            last_decoded_data = qr_data_str

            # --- 7. 这是新的核心解码逻辑 ---
            reconstructed_dt = reconstruct_timestamp(qr_data_str)
            
            # 只有在成功重构时间后（即，它是一个有效的HHMMSSsss码）
            if reconstructed_dt:
                # 在计算差值前立刻获取扫描器时间，以保证最高精度
                scanner_time = datetime.datetime.now()
                scanner_time_ms = int(scanner_time.timestamp() * 1000)
                
                # 将重构后的datetime对象转换为毫秒级Unix时间戳
                generator_time_ms = int(reconstructed_dt.timestamp() * 1000)
                time_difference = scanner_time_ms - generator_time_ms
                
                # --- 在控制台打印结果 ---
                print("----------------------------------------")
                print(f"二维码解码成功: '{qr_data_str}'")
                print(f"  生成器时间 (A): {reconstructed_dt.strftime('%H:%M:%S.%f')[:-3]}")
                print(f"  扫描器时间 (B): {scanner_time.strftime('%H:%M:%S.%f')[:-3]}")
                print(f"  时间差 (B - A): {time_difference} ms")

                # --- 在视频帧上绘制结果 ---
                points = obj.polygon
                pts = np.array([(p.x, p.y) for p in points], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

                display_text = f"Delay: {time_difference} ms"
                rect = obj.rect
                cv2.putText(frame, display_text, (rect.left, rect.top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # --- 8. (Bug修复) 只有成功后才将数据计入统计 ---
                total_time_difference.append(time_difference)
                mean_delay = np.mean(total_time_difference)
                print(f"  当前平均延迟:   {mean_delay:.2f} ms ({len(total_time_difference)} samples)")
                print("----------------------------------------\n")
            
                if len(total_time_difference) >= 100:
                    break # 退出内层循环
            
            else:
                # 如果解码的数据不是 HHMMSSsss 格式
                print(f"解码到无效数据: '{qr_data_str}'，请将摄像头对准正确的时间二维码...")
        
        # 检查是否需要退出外层循环
        if len(total_time_difference) >= 100:
            print("已收集100个样本，程序将自动退出。")
            break

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
    
    # 9. 打印最终的统计结果
    if total_time_difference:
        final_mean_delay = np.mean(total_time_difference)
        print(f"\n最终平均延迟 (基于 {len(total_time_difference)} 个样本): {final_mean_delay:.2f} ms")

if __name__ == "__main__":
    # 调用函数，并指定为 .mp4 文件
    run_qr_scanner(output_filename="qr_scan_video.mp4")