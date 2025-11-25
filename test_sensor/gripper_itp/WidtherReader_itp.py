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
    def __init__(self, calibration_file_path="gripper_calibration.json"):
        from json import load
        with open(calibration_file_path, "r") as f:
                calib_d = load(f)

        self.calib_data = calib_d["calibration"]
        self.zero_point = calib_d.get("zero_point", 0)
        self.sample_num = calib_d["metadata"]["sample_num"]

        self.calib_data = [(0, -self.calib_data[1][1])] + self.calib_data
        self.calib_data = self.calib_data + [(2*self.calib_data[-1][0] - self.calib_data[-2][0], 2*self.calib_data[-1][1] - self.calib_data[-2][1])]   

    def map(self, x):
        x = (x - self.zero_point) % 4096
        for i in range(len(self.calib_data) - 1):
            if self.calib_data[i][0] <= x <= self.calib_data[i + 1][0]:
                # 线性插值
                y0, x0 = self.calib_data[i]
                y1, x1 = self.calib_data[i + 1]
                out = (y1 - y0) * (x - x0) / (x1 - x0) + y0
                if out < 0:
                    out = 0
                return out
        print("Warning: value out of calibration range")
        return None

    def get_gripper_width(self, use_mm=False):
        """
        返回夹爪宽度，默认单位为cm
        传入use_mm=False则返回m
        """
        raw = ReadRawAngle()
        num = self.map(raw)

        theta = num / 4096 * 360 / 180 * math.pi
        width = 2 * l(self.theta_0 - theta) + self.b
        if use_mm:
            width *= 1000
        return width

# 测试代码兼使用例
if __name__ == "__main__":
    width_reader = WidthReader(calibration_file_path="gripper_calibration.json")
    while True:
        width = width_reader.get_gripper_width()
        print("Gripper width: {:.2f} m".format(width))
        time.sleep(0.1)