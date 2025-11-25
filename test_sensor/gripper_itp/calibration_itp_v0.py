"""
依照指示对夹爪进行校准，并将校准数据保存为json文件
校准前，需要测量夹爪最大宽度并修改max_width变量
"""

import smbus
import time
import math
from scipy.optimize import fsolve
import numpy as np

# hardware setup
DEVICE_AS5600 = 0x36  #0x36  # Default device I2C address
bus = smbus.SMBus(2)

# optional parameters
max_width = 84  # max width of the gripper(mm)

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
    
max_samples = max_width // 10

def calibrate_gripper():
    calibration_data = {
        "metadata": {
            "unit": "cm",
            "sensor_range": [0, 4095],
            "sample_num": max_samples
        },
        "calibration": [],
        "zero_point": 0,
    }
    
    print("=== Calibration of gripper width and as5600 readings ===")

    user_input = input(f"Adjust gripper width to 0 cm, and press ENTER to start: ")
    width = 0
    readings = []
    for _ in range(5):
        if angle := ReadRawAngle():
            readings.append(angle)
            time.sleep(0.05)
    
    avg_angle = sum(readings) // len(readings)
    zero_point = avg_angle
    calibration_data["calibration"].append( (width / 100.0, 0) )  # meter
    calibration_data["zero_point"] = zero_point

    print(f"Recorded: width {width} cm -> angle value {avg_angle}")

    for width in range(1, max_samples + 1): # max_samples 1cm -> 8cm
        while True:
            user_input = input(f"Adjust gripper width to {width}cm, and press ENTER to record: ")
            
            # 读取5次取平均
            readings = []
            for _ in range(5):
                if angle := ReadRawAngle():
                    readings.append(angle)
                    time.sleep(0.05)
            
            avg_angle = sum(readings) // len(readings)
            avg_angle = (avg_angle - zero_point) % 4096
            calibration_data["calibration"].append( (width / 100.0, avg_angle) )  # meter

            print(f"Recorded: width {width}cm -> angle value {avg_angle}")
            break

    with open("gripper_calibration.json", "w") as f:
        json.dump(calibration_data, f, indent=2)
    
    print("\nSaved to gripper_calibration.json")