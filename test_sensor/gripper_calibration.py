"""
依照指示对夹爪进行校准，并将校准数据保存为json文件
校准前，需要测量夹爪最大宽度并修改max_width变量
"""

import smbus
import time
import math
from scipy.optimize import fsolve
import numpy as np


# hardware parameter
max_width = 84  # max width of the gripper(mm)

# hardware setup
DEVICE_AS5600 = 0x36  #0x36  # Default device I2C address
bus = smbus.SMBus(2)

# optional parameters
change_rate_treshold = 2048  # threshold to detect if the returned value goes from 4095 to 0 or vice versa
closeness_treshold = 50  # threshold to determine the starting point of the gripper(lower->stricter)

# calibration parameters
sgn = 1 # -1 if the returned value goes from 4095 to 0 or vice versa; 1 otherwise
sensor_range = None  # the range of feedback from the sensor

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

# calculate sgn & sensor range
read = ReadRawAngle()

min_read = read
max_read = read

inner_max = 4096
inner_min = 0

read_prev = read

print("Getting sensor range...")
print("Now please gently close the gripper and then open it to the maximum width,")
print("repeating several times,")
print("pausing briefly in both the closed and fully open positions.")

for i in range(600):
    read = ReadRawAngle()
    print(read)
    if read > max_read:
        max_read = read
    if read < min_read:
        min_read = read

    if read > 2048 and read < inner_max:
        inner_max = read
    if read < 2048 and read > inner_min:
        inner_min = read

    if sgn is None and i != 0:
        if abs(read - read_prev) > change_rate_treshold:
            sgn = -1
        read_prev = read

    time.sleep(0.1)
print(sgn)

if sgn == -1:
    upper = inner_max
    lower = inner_min
if sgn == 1:
    upper = max_read
    lower = min_read
if sgn is None:
    print("Fail to determine the direction of the sensor, do the calibration again.")
    exit(1)

print("Sensor range recorded.")
print("Now please gently close the gripper and stay.")

time.sleep(5)

x_0 = None
x_1 = None

for _ in range(2):
    time.sleep(2)
    data = []
    for i in range(60):
        read = ReadRawAngle()
        data.append(read)
        time.sleep(0.1)
    
    print(abs(sum(data)/len(data) - lower), closeness_treshold, lower, upper)
    
    if abs(sum(data)/len(data) - lower) < closeness_treshold:
        x_0 = lower
        x_1 = upper
        break
    elif abs(sum(data)/len(data) - upper) < closeness_treshold:
        x_0 = upper
        x_1 = lower
        break
    else:
        print("Something went wrong, please try again.")
        print("Now gently keep the gripper closed and wait.")
        continue

if x_0 is None or x_1 is None:
    print("Fail to align, do the calibration again.")
    exit(1)

print("closed state recorded.")

# TODO: complete the calibration
if sgn == -1:
    sensor_range = lower - upper + 4096
else:
    sensor_range = upper - lower

# calculate parameters for gripper width estimation 
delta_rad = sensor_range / 4096 * 360 * math.pi / 180
print("delta_rad: ", delta_rad)

def l(theta):
    return 27*np.cos(theta) + np.sqrt(39**2 - (11 + 27*np.sin(theta))**2)

def equation(theta):
    return l(theta - delta_rad) - l(theta) - max_width / 2

theta_0 = fsolve(equation, 0.6)
b = - 2*l(theta_0)

reverse = False

if sgn * (x_0 - x_1) > 0:
    reverse = True

# save calibration result as json
import json

calibration_data = {
    "theta_0": theta_0.tolist(),
    "b": b.tolist(),
    "reverse": reverse,
    "x_0": x_0,
    "x_1": x_1
}

print("Calibration result:" )
print(calibration_data)

with open("../Sensor/angle_calib_params/gripper_calibration.json", "w") as f:
    json.dump(calibration_data, f, indent=4)

print("Calibration data saved to gripper_calibration.json")
