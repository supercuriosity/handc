# subscriber_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
import cv2
import time
import numpy as np
from cv_bridge import CvBridge
import threading

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from audio_common_msgs.msg import AudioData
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque

class MacSubscriber(Node):
    def __init__(self):
        super().__init__('mac_subscriber')
        
        # --- 数据存储 ---
        # 用于存储从各个话题接收到的最新数据
        self.image_raw = None
        self.tactile_left_image = None
        self.tactile_right_image = None
        self.current_angle = -10086
        self.left_force = -10086
        self.right_force = -10086
        self.bridge = CvBridge()
        # 创建一个锁来防止多线程访问图像数据时发生冲突
        
        self.declare_parameter('sample_rate', 44100)
        self.SAMPLE_RATE = self.get_parameter('sample_rate').get_parameter_value().integer_value
        
        # --- 数据存储和线程安全 ---
        self.audio_chunk = None
        
        self.lock = threading.Lock()
        
        self.max_history = 100
        self.angle_history = deque(maxlen=self.max_history)
        self.left_force_history = deque(maxlen=self.max_history)
        self.right_force_history = deque(maxlen=self.max_history)
        self.time_history = deque(maxlen=self.max_history)

        # --- 订阅者设置 ---
        
        # 2. 订阅正确的音频话题和消息类型
        self.audio_subscription = self.create_subscription(
            AudioData,
            '/audio/audio_raw',
            self.audio_callback,
            10) # QoS profile depth
        
        
        # 订阅 /camera/image_raw
        self.image_raw_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_raw_callback,
            10)
        
        # 订阅 /camera/tactile_left
        self.tactile_left_subscription = self.create_subscription(
            Image,
            '/camera/tactile_left',
            self.tactile_left_callback,
            10)

        self.tactile_right_subscription = self.create_subscription(
            Image,
            '/camera/tactile_right',
            self.tactile_right_callback,
            10)
        
        # force 
        self.left_force_subscription = self.create_subscription(
            Float64,
            '/sensor/force_left',
            self.left_force_callback,
            10)
        
        self.right_force_subscription = self.create_subscription(
            Float64,
            '/sensor/force_right',
            self.right_force_callback,
            10)

        # 订阅 /sensor/as5600/angle
        self.angle_subscription = self.create_subscription(
            Float64,
            '/sensor/as5600/angle',
            self.angle_callback,
            10)
        
        self.get_logger().info("订阅节点已启动，等待所有话题数据...")
        
        # --- 创建一个定时器来刷新显示 ---
        # 使用定时器可以解耦图像的接收和处理，让显示更流畅
        # 每秒刷新30次 (1/30s)
        self.timer = self.create_timer(1.0/20.0, self.update_display)

    def audio_callback(self, msg):
        audio_data = np.array(msg.data, dtype=np.int16)
        
        with self.lock:
            self.audio_chunk = audio_data
    
    def image_raw_callback(self, msg):
        """处理 /camera/image_raw 的回调函数"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.image_raw = cv_image
        except Exception as e:
            self.get_logger().error(f'CvBridge 错误 (image_raw): {e}')

    def tactile_left_callback(self, msg):
        """处理 /camera/tactile_left 的回调函数"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.tactile_left_image = cv_image
        except Exception as e:
            self.get_logger().error(f'CvBridge 错误 (tactile_left): {e}')

    def tactile_right_callback(self, msg):
        """处理 /camera/tactile_right 的回调函数"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.tactile_right_image = cv_image
        except Exception as e:
            self.get_logger().error(f'CvBridge 错误 (tactile_right): {e}')

    def left_force_callback(self, msg):
        """处理 /sensor/force_left 的回调函数"""
        self.get_logger().info(f'左侧力传感器数据: {msg.data}')
        self.left_force = msg.data

    def right_force_callback(self, msg):
        """处理 /sensor/force_right 的回调函数"""
        self.get_logger().info(f'右侧力传感器数据: {msg.data}')
        self.right_force = msg.data

    def angle_callback(self, msg):
        """处理角度信息的回调函数"""
        # 简单地更新角度值
        self.current_angle = msg.data
    
    def update_data_history(self):
        """更新时间序列数据"""
        current_time = time.time()
        self.timestamp_history.append(current_time)
        self.angle_history.append(self.current_angle)
        self.left_force_history.append(self.left_force)
        self.right_force_history.append(self.right_force)

    def update_display(self):
        """
        定时器触发的函数，创建6个子图的可视化界面。
        """
        with self.lock:
            # 复制图像以避免在处理时被回调函数修改
            img_raw = self.image_raw
            img_tactile_left = self.tactile_left_image
            img_tactile_right = self.tactile_right_image
            audio_chunk = self.audio_chunk

        # 确保所有图像都已经接收到
        if img_raw is None or img_tactile_left is None or img_tactile_right is None:
            self.get_logger().info("正在等待所有摄像头的图像数据...", throttle_duration_sec=5)
            return

        # 更新时间序列数据
        current_time = time.time()
        self.time_history.append(current_time)
        self.angle_history.append(self.current_angle)
        self.left_force_history.append(self.left_force)
        self.right_force_history.append(self.right_force)

        # 创建matplotlib图形
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Real-time Data Visualization', fontsize=16)

        # 上排三个图像子图
        # 子图1: img_raw
        axes[0, 0].imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Camera Raw Image')
        axes[0, 0].axis('off')

        # 子图2: img_tactile_left
        axes[0, 1].imshow(cv2.cvtColor(img_tactile_left, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Tactile Left')
        axes[0, 1].axis('off')

        # 子图3: img_tactile_right
        axes[0, 2].imshow(cv2.cvtColor(img_tactile_right, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Tactile Right')
        axes[0, 2].axis('off')

        # 下排三个数据子图
        # 子图4: current_angle 折线图
        if len(self.angle_history) > 1:
            time_points = list(range(len(self.angle_history)))
            
            # 过滤掉无效的初始值-10086
            valid_angles = [angle for angle in self.angle_history if angle != -10086]

            if not valid_angles:
                axes[1, 0].text(0.5, 0.5, 'Waiting for angle data...', ha='center', va='center')
                axes[1, 0].set_title('Current Angle (Time Series)')
                axes[1, 0].set_xlim(0, self.max_history-1)
                axes[1, 0].set_ylim(-100, 100) # 设置一个默认范围
            else:
                # 仍然使用原始历史数据绘图，以保持时间轴的连续性
                # 但无效值会被绘制在0的位置
                y_upper = [84 - n if n != -10086 else 0 for n in self.angle_history]
                y_lower = [n-84 if n != -10086 else 0 for n in self.angle_history]
                
                axes[1, 0].plot(time_points, y_upper, 'b-', linewidth=2, marker='o', label='Upper Bound')
                axes[1, 0].plot(time_points, y_lower, 'r-', linewidth=2, marker='o', label='Lower Bound')
                axes[1, 0].set_title('Current Angle (Time Series)')
                axes[1, 0].set_xlabel('Frame')
                axes[1, 0].set_ylabel('Angle Range')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_xlim(0, self.max_history-1)
                
                # 根据有效数据的最大值动态设置Y轴范围
                axes[1, 0].set_ylim(-84, 84)
                axes[1, 0].legend()

        # 子图5: left_force 柱状图
        if len(self.left_force_history) > 0:
            time_points = list(range(len(self.left_force_history)))
            bars = axes[1, 1].bar(time_points, list(self.left_force_history), 
                                color='green', alpha=0.7, width=0.6)
            axes[1, 1].set_title('Left Force (Bar Chart)')
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('Force')
            axes[1, 1].set_xlim(-0.5, self.max_history-0.5)
            axes[1, 1].set_ylim(0, 10)  # 设置Y轴范围：最小值0，最大值10
            
            # 添加数值标签
            # for i, bar in enumerate(bars):
            #     height = bar.get_height()
            #     if height != -10086:  # 只在有效数据时显示
            #         axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
            #                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
  
        # 子图6: right_force 柱状图
        if len(self.right_force_history) > 0:
            time_points = list(range(len(self.right_force_history)))
            bars = axes[1, 2].bar(time_points, list(self.right_force_history), 
                                color='red', alpha=0.7, width=0.6)
            axes[1, 2].set_title('Right Force (Bar Chart)')
            axes[1, 2].set_xlabel('Frame')
            axes[1, 2].set_ylabel('Force')
            axes[1, 2].set_xlim(-0.5, self.max_history-0.5)
            axes[1, 2].set_ylim(0, 10)  # 设置Y轴范围：最小值0，最大值10
            
            # 添加数值标签
            # for i, bar in enumerate(bars):
            #     height = bar.get_height()
            #     if height != -10086:  # 只在有效数据时显示
            #         axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
            #                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        # ### ADDED ###: 第三排，第7个子图用于音频频谱 (Row 2)
        if audio_chunk is not None and len(audio_chunk) > 0:
            N = len(audio_chunk)
            yf = np.fft.fft(audio_chunk)
            xf = np.fft.fftfreq(N, 1 / self.SAMPLE_RATE)
            
            positive_mask = xf >= 0
            xf_positive = xf[positive_mask]
            yf_magnitude = np.abs(yf[positive_mask]) / N

            # 绘制到第7个子图 (axes[2, 0])
            axes[2, 0].plot(xf_positive, yf_magnitude, color='cyan')
            axes[2, 0].set_title('Audio Spectrum')
            axes[2, 0].set_xlabel('Frequency (Hz)')
            axes[2, 0].set_ylabel('Amplitude')
            axes[2, 0].set_xlim(0, self.SAMPLE_RATE / 2)
            axes[2, 0].set_ylim(0, 150) # 可根据实际情况调整
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'Waiting for audio data...', ha='center', va='center')
            axes[2, 0].set_title('Audio Spectrum')
        
        # ### ADDED ###: 隐藏第8和第9个不用的子图
        # axes[2, 0].axis('off')
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应主标题

        # --- 可视化：Matplotlib 绘图 ---
        # fig, ax = plt.subplots(figsize=(8, 4))
        
        # ax.plot(xf_positive, yf_magnitude, color='cyan')
        
        
        # # 调整子图间距
        # plt.tight_layout()

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        try:
            # 新版本matplotlib的方法
            buf = canvas.buffer_rgba()
            # 转换为numpy数组
            surf = np.asarray(buf)
            # 转换为RGB格式（去掉alpha通道）
            surf = surf[:, :, :3]
        except AttributeError:
            # 兼容旧版本matplotlib
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            size = canvas.get_width_height()
            surf = np.frombuffer(raw_data, dtype=np.uint8)
            surf = surf.reshape((size[1], size[0], 3))
        
        # 转换为BGR格式以供OpenCV显示
        final_canvas = cv2.cvtColor(surf, cv2.COLOR_RGB2BGR)
        
        # 显示最终图像
        cv2.imshow("Six-Panel Visualization", final_canvas)
        cv2.waitKey(1)   
        # 关闭matplotlib图形以释放内存
        plt.close(fig)               

def main(args=None):
    rclpy.init(args=args)
    node = MacSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("节点被手动终止")
    finally:
        # 确保在退出时销毁所有资源
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()