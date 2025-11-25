"""
夹爪宽度读取代码
需要先运行gripper_calibration.py进行校准

使用方法：
实例化WidthReader时传入配置文件路径
随后调用get_gripper_width()获取夹爪宽度，默认单位为cm
"""

import json
import smbus
import time
import math
from scipy.optimize import fsolve
import numpy as np

DEVICE_AS5600 = 0x36  #0x36  # Default device I2C address
bus = smbus.SMBus(2)

margin = 500

def ReadRawAngle(): # Read angle (0-360 represented as 0-4096)
    try:
        read_bytes = bus.read_i2c_block_data(DEVICE_AS5600, 0x0C, 2)
        return (read_bytes[0] << 8) | read_bytes[1]
    except Exception as e:
        print(f"Error reading raw angle: {e}")
        return None

def ReadMagnitude(): # Read magnetism magnitude
    try:
        read_bytes = bus.read_i2c_block_data(DEVICE_AS5600, 0x1B, 2)
        return (read_bytes[0] << 8) | read_bytes[1]
    except Exception as e:
        print(f"Error reading magnitude: {e}")
        return None

def l(theta):
    return 27*np.cos(theta) + np.sqrt(39**2 - (11 + 27*np.sin(theta))**2)


class WidthReader:
    """
    夹爪宽度读取类
    需要先运行gripper_calibration.py进行校准
    初始化时需要输入校准数据路径
    """
    def __init__(self, calibration_file_path="/root/workspace/HandCap/Sensor/angle_calib_params/gripper_calibration.json"):
        from json import load
        with open(calibration_file_path, "r") as f:
            calib_data = load(f)
        print(calib_data["theta_0"])
        self.theta_0 = calib_data["theta_0"][0]
        print(self.theta_0)
        self.b = calib_data["b"][0]
        self.reverse = calib_data["reverse"]
        self.x_0 = calib_data["x_0"]
        self.x_1 = calib_data["x_1"]

    def get_gripper_width(self, use_cm=True):
        """
        返回夹爪宽度，默认单位为cm
        传入use_cm=False则返回mm
        """
        def transformed_read(x):
            """
            将原始传感器读数转换为线性值。
            通过使用取模运算符处理0-4096的环绕情况。
            """
            if self.reverse:
                # sensor value decreases as the gripper opens
                return (self.x_0 - x + 4096) % 4096
            else:
                # sensor value increases as the gripper opens
                return (x - self.x_0 + 4096) % 4096

        raw = ReadRawAngle()
        num = transformed_read(raw)

        theta = num / 4096 * 360 / 180 * math.pi
        width = 2 * l(self.theta_0 - theta) + self.b
        if use_cm:
            width /= 10
        return width

# 测试代码兼使用例
if __name__ == "__main__":
    width_reader = WidthReader(calibration_file_path="/root/workspace/HandCap/Sensor/angle_calib_params/gripper_calibration.json")
    while True:
        width = width_reader.get_gripper_width()
        print("Gripper width: {:.2f} cm".format(width))
        time.sleep(0.1)