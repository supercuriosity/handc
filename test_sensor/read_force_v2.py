import smbus
import time
# import cv2
import numpy as np


# ------------------ 参数定义 (根据原Arduino代码) ------------------
# 1. 传感器的压力(力)量程定义 (单位: 克 g)
PRESS_MIN = 50
PRESS_MAX = 1000

# 2. 传感器输出的有效电压范围定义 (单位: 毫伏 mV)
#    这是进行压力换算的关键区间
VOLTAGE_MIN = 1000
VOLTAGE_MAX = 33000


# 4. 假设您的系统参考电压为5V (即5000mV)，如果您的系统是3.3V，请修改为3300
SYSTEM_VOLTAGE_MV = 5000


def linear_map(value, in_min, in_max, out_min, out_max):
    """
    一个实现类似Arduino map()功能的函数，用于线性映射。
    """
    if in_max == in_min:
        return out_min
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

def convert_to_pressure(adc_value):
    """
    将输入的原始ADC值转换为估算的压力值（单位：克）。
    """
    voltage_mv = linear_map(adc_value, VOLTAGE_MIN, VOLTAGE_MAX, 0, SYSTEM_VOLTAGE_MV)
    
    if voltage_mv < VOLTAGE_MIN:
        pressure_g = 0
    elif voltage_mv > VOLTAGE_MAX:
        pressure_g = PRESS_MAX
    else:
        pressure_g = linear_map(voltage_mv, VOLTAGE_MIN, VOLTAGE_MAX, PRESS_MIN, PRESS_MAX)
        
    return pressure_g, voltage_mv 



# left
DEVICE_force = 0x48  # Default device I2C address
bus = smbus.SMBus(4)

def ReadRawForce():  # Read force (0-100 represented as 0-4096)
    read_bytes = bus.read_i2c_block_data(DEVICE_force, 0x0C, 2)
    return (read_bytes[0] << 8) | read_bytes[1]


def main():
    while True:
        adc_v = ReadRawForce()
        force, Voltage = convert_to_pressure(adc_v)
        print(f"压力值: {force:02f} --- 电压值: {Voltage:02f}")


if __name__ == "__main__":
    main()