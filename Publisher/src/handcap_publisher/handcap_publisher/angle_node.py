
import math
import smbus
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64 # 使用标准浮点数消息
from scipy.optimize import fsolve

max_width = 84
sensor_range = 596
delta_rad = sensor_range / 4096 * 360 * math.pi / 180

def l(theta):
    return 27*np.cos(theta) + np.sqrt(39**2 - (11 + 27*np.sin(theta))**2)

def equation(theta):
    return l(theta - delta_rad) - l(theta) - max_width / 2

class AnglePublisher(Node):
    def __init__(self):
        super().__init__('sensor_node')
        
        self.max_width = 84  # max width of the gripper(mm)

        self.sensor_range = 596
        self.theta_0 = fsolve(equation, 0.6)
        self.b = - 2*l(self.theta_0)
        self.bus = smbus.SMBus(2)
        
        initial_read = -1
        read = self.ReadRawAngle()
        if read > initial_read:
            self.initial_read = read
        
        self.publisher_ = self.create_publisher(Float64, '/sensor/as5600/angle', 10)
        timer_period = 0.1 
        self.timer = self.create_timer(timer_period, self.timer_callback)

    
    def ReadRawAngle(self): # Read angle (0-360 represented as 0-4096)
        DEVICE_AS5600 = 0x36 # Default device I2C address
        read_bytes = self.bus.read_i2c_block_data(DEVICE_AS5600, 0x0C, 2)
        x = (read_bytes[0]<<8) | read_bytes[1];
        # if x<100:
        #     x = 4096
        return float(x)

    def timer_callback(self):
        angle = self.ReadRawAngle()  # 读取角度
        num = self.initial_read - angle
        theta = num / 4096 * 360 / 180 * math.pi #/ 4096 * 10 * 2 * math.pi
        width = 2 * l(self.theta_0 - theta) + self.b
        num = width


        # 2. output: visualization
        if num < 0:
            num = 0
        alpha = 0.8  # limiting the max width of the visualization
        num = int(int(num * alpha) * 2)
        max_num = int(int(max_width * alpha) * 2)
        side = (max_num - num) // 2
        msg = Float64()
        msg.data = float(side)
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing angle: {angle}')

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    angle_publisher = AnglePublisher()
    try:
        rclpy.spin(angle_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        angle_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()