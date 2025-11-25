import cv2
from pyzbar import pyzbar
import datetime
import platform
import os
import sys
import signal
import threading
import time


def set_system_time(dt_object: datetime.datetime):
    """
    根据操作系统设置系统时间。
    需要管理员/sudo权限。
    """
    system = platform.system()
    
    try:
        if system == "Linux" or system == "Darwin": # Darwin 是 macOS
            # 使用 date 命令设置时间
            # 格式: "YYYY-MM-DD HH:MM:SS"
            time_str = dt_object.strftime('%Y-%m-%d %H:%M:%S')
            
            # 必须使用 sudo
            print(f"\n正在尝试使用 'sudo' 设置系统时间为: {time_str}")
            print("您可能需要在终端中输入密码。")
            os.system(f"sudo date -s \"{time_str}\"")
            print(f"✅ 命令已执行。请检查您的系统时间是否已更新。")

        else:
            print(f"不支持的操作系统: {system}")
            
    except Exception as e:
        print(f"\n❌ 设置系统时间失败! 错误: {e}")
        print("请确保您是以管理员(Windows)或使用sudo(Linux/macOS)运行此脚本！")


def test_x11_display():
    """
    全面测试X11显示环境
    """
    print("=== X11显示环境诊断 ===")
    
    # 检查环境变量
    display = os.environ.get('DISPLAY', None)
    print(f"DISPLAY环境变量: {display}")
    
    ssh_client = os.environ.get('SSH_CLIENT', None)
    print(f"SSH_CLIENT: {ssh_client}")
    
    # 检查xauth
    xauth_result = os.system("xauth list > /dev/null 2>&1")
    print(f"xauth状态: {'✅ 正常' if xauth_result == 0 else '❌ 异常'}")
    
    # 测试简单的X11应用
    print("测试X11转发...")
    test_result = os.system("timeout 2 xeyes > /dev/null 2>&1 &")
    
    if display and display.startswith(':'):
        print(f"检测到显示: {display}")
        # 检查是否是本地显示
        if display.startswith(':0') or display.startswith(':10') or display.startswith(':11'):
            print("检测到SSH X11转发显示")
            return True, False
        elif display.startswith(':99'):
            print("检测到虚拟显示 (xvfb)")
            return True, True
    
    return False, False


def setup_opencv_display():
    """
    设置OpenCV显示环境
    """
    try:
        # 强制使用X11后端
        cv2.namedWindow('opencv_test', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('opencv_test')
        print("✅ OpenCV X11后端测试成功")
        return True
    except Exception as e:
        print(f"❌ OpenCV X11后端测试失败: {e}")
        
        # 尝试设置后端
        try:
            import os
            os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'
            # 可能需要重新导入cv2
            print("尝试重新配置OpenCV后端...")
            return False
        except:
            return False


class KeyboardListener:
    """处理键盘输入的线程类"""
    def __init__(self):
        self.should_exit = False
        self.thread = None
    
    def start(self):
        self.thread = threading.Thread(target=self._listen_for_input, daemon=True)
        self.thread.start()
    
    def _listen_for_input(self):
        print("输入 'q' 并按回车退出程序...")
        while not self.should_exit:
            try:
                user_input = input().strip().lower()
                if user_input == 'q':
                    print("收到退出命令...")
                    self.should_exit = True
                    break
                elif user_input == 'test':
                    # 添加测试命令
                    print("正在测试X11显示...")
                    os.system("xeyes &")
            except (EOFError, KeyboardInterrupt):
                self.should_exit = True
                break


def signal_handler(signum, frame):
    """处理Ctrl+C信号"""
    print("\n收到中断信号，正在退出...")
    sys.exit(0)


def main():
    """
    通过摄像头扫描二维码，解析时间并设置系统时钟。
    """
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 全面测试X11环境
    use_display, is_virtual = test_x11_display()
    
    if not use_display:
        print("❌ X11显示环境不可用")
        print("请检查SSH X11转发配置:")
        print("1. 使用 ssh -Y 或 ssh -X 连接")
        print("2. 确认服务端 /etc/ssh/sshd_config 中 X11Forwarding yes")
        print("3. 尝试运行 xeyes 测试X11转发")
        print("\n程序将在无显示模式下继续运行...")
    elif is_virtual:
        print("检测到虚拟显示环境，无图像显示")
    else:
        print("✅ X11显示环境可用")
        # 测试OpenCV
        opencv_ok = setup_opencv_display()
        if not opencv_ok:
            print("❌ OpenCV显示可能有问题，但程序会继续尝试")
    
    # 尝试打开摄像头
    print("\n正在检测摄像头设备...")
    cap = None
    
    # 首先尝试指定的设备
    for device in ["/dev/video4", "/dev/video0", "/dev/video1", "/dev/video2", 0, 1, 2, 3, 4]:
        try:
            print(f"尝试打开摄像头: {device}")
            cap = cv2.VideoCapture(device)
            if cap.isOpened():
                # 测试是否能读取帧
                ret, frame = cap.read()
                if ret:
                    print(f"✅ 成功打开摄像头设备: {device}")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                cap = None
        except Exception as e:
            print(f"设备 {device} 打开失败: {e}")
            cap = None
    
    if cap is None:
        print("❌ 无法打开任何摄像头设备")
        return

    print("摄像头已启动... 将镜头对准时间二维码。")
    print("成功扫描后会请求确认。")
    
    # 启动键盘监听器
    keyboard_listener = KeyboardListener()
    keyboard_listener.start()
    print("输入 'test' 测试X11显示，输入 'q' 退出程序")
    
    last_qr_data = ""
    frame_count = 0
    last_status_time = time.time()

    # 创建窗口（如果支持显示）
    window_created = False
    if use_display and not is_virtual:
        try:
            cv2.namedWindow('Time Sync Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Time Sync Camera', 800, 600)
            window_created = True
            print("✅ 图像窗口已创建")
        except Exception as e:
            print(f"❌ 创建窗口失败: {e}")
            window_created = False

    try:
        while True:
            # 检查是否应该退出
            if keyboard_listener.should_exit:
                break
            
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            frame_count += 1
            current_time = time.time()
            
            # 每3秒输出一次状态信息
            if current_time - last_status_time >= 3.0:
                print(f"正在扫描... 帧数: {frame_count} | 分辨率: {frame.shape[1]}x{frame.shape[0]}")
                last_status_time = current_time

            # 二维码检测
            try:
                qrcodes = pyzbar.decode(frame)
            except Exception as e:
                print(f"二维码解码错误: {e}")
                continue
            
            # 在图像上添加状态信息
            if not qrcodes:
                cv2.putText(frame, "Scanning for QR Code...", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for qr in qrcodes:
                qr_data = qr.data.decode('utf-8')

                # 只有在扫描到新的二维码内容时才处理
                if qr_data != last_qr_data:
                    last_qr_data = qr_data
                    print(f"\n🎯 检测到二维码内容: {qr_data}")
                    
                    try:
                        # 解析时间戳
                        decoded_time = datetime.datetime.fromisoformat(qr_data)
                        time_str = decoded_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        print(f"📅 扫描到时间: {time_str}")
                        
                        # 用户确认
                        choice = input("是否将本机系统时间设置为此时间? (y/n): ").lower()
                        if choice == 'y':
                            set_system_time(decoded_time)
                        else:
                            print("操作已取消。")

                    except ValueError as e:
                        print(f"❌ 二维码内容不是有效的时间格式: {e}")
                        cv2.putText(frame, "Invalid Time QR Code", 
                                   (qr.rect.left, qr.rect.top - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        continue

                # 绘制二维码边框
                (x, y, w, h) = qr.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, "Time QR Detected", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示画面
            if window_created:
                try:
                    cv2.imshow('Time Sync Camera', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except Exception as e:
                    print(f"显示画面失败: {e}")
                    window_created = False
            else:
                # 无显示模式，添加延迟
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    finally:
        if cap:
            cap.release()
        if window_created:
            cv2.destroyAllWindows()
        print("程序已退出。")


if __name__ == '__main__':
    print("=== 时间同步程序 ===")
    print(f"当前系统时间: {datetime.datetime.now()}")
    main()