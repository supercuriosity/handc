import smbus
import time
import math
from scipy.optimize import fsolve
import numpy as np


# hardware parameter
max_width = 84  # max width of the gripper(mm)

sensor_range = 596  # the range of feedback from the sensor

DEVICE_AS5600 = 0x36  #0x36  # Default device I2C address
bus = smbus.SMBus(2)
def ReadRawAngle(): # Read angle (0-360 represented as 0-4096)
    read_bytes = bus.read_i2c_block_data(DEVICE_AS5600, 0x0C, 2)
    return (read_bytes[0]<<8) | read_bytes[1];


def ReadMagnitude(): # Read magnetism magnitude
    read_bytes = bus.read_i2c_block_data(DEVICE_AS5600, 0x1B, 2)
    return (read_bytes[0]<<8) | read_bytes[1];

# getting sensor range
# TODO: deal with the reverse case
initial_read = -1
max_read = 4097
print("Getting sensor range...")
print("Recording the closed state...")
print("Please keep the gripper closed and wait...")

for i in range(40):
    read = ReadRawAngle()
    if read > initial_read:
        initial_read = read
    time.sleep(0.1)

print("state recorded, now recording the max width state...")
time.sleep(1)
print("please open the gripper to the max width and wait...")

for i in range(30):
    read = ReadRawAngle()
    if read < max_read:
        max_read = read
    time.sleep(0.1)

print("max width state recorded.")

sensor_range = initial_read - max_read

# calculate parameters for gripper width estimation 
delta_rad = sensor_range / 4096 * 360 * math.pi / 180
print("delta_rad: ", delta_rad)

def l(theta):
    return 27*np.cos(theta) + np.sqrt(39**2 - (11 + 27*np.sin(theta))**2)

def equation(theta):
    global delta_rad, max_width
    return l(theta - delta_rad) - l(theta) - max_width / 2

theta_0 = fsolve(equation, 0.6)
b = - 2*l(theta_0)

print("theta_0: ", theta_0)
print("b: ", b)
print("l_0: ", 2*l(theta_0)+b)
print("l_max: ", (2*l(theta_0-delta_rad)+b))
print("equation(theta_0): ", equation(theta_0))

time.sleep(1)

start = None
min_n = 600
num_space = 50


while True:
    if start is None:
        start = initial_read
    num = start - ReadRawAngle()

    theta = num / 4096 * 360 / 180 * math.pi #/ 4096 * 10 * 2 * math.pi
    width = 2 * l(theta_0 - theta) + b
    num = width

    # 1. output: width
    #print(num)

    # 2. output: visualization
    if num < 0:
        num = 0
    alpha = 0.8  # limiting the max width of the visualization
    num = int(int(num * alpha) * 2)
    max_num = int(int(max_width * alpha) * 2)
    side = (max_num - num) // 2
    print(" "*side+"#"+" "*num+"#"+" "*side)

    time.sleep(0.1)
