import smbus
import time
# import cv2
import numpy as np


# ------------------ 参数定义 (根据提供的Arduino代码进行适配) ------------------

ADC_MIN = 0
ADC_MAX = 32767  

SYSTEM_VOLTAGE_MV = 5000

EFFECTIVE_VOLTAGE_MIN_MV = 500  
EFFECTIVE_VOLTAGE_MAX_MV = 5000 

PRESS_MIN_G = 50
PRESS_MAX_G = 1000


def linear_map(value, in_min, in_max, out_min, out_max):
    """
    一个实现类似Arduino map()功能的函数，用于线性映射。
    进行浮点数运算以保证精度。
    """
    # 避免除以零
    if in_max == in_min:
        return out_min
    return out_min + float(value - in_min) * float(out_max - out_min) / float(in_max - in_min)

def convert_to_pressure(adc_value):
    """
    将输入的原始ADC值转换为估算的压力值（单位：克）。
    其逻辑与Arduino中的 getPressValue 函数完全对应。
    """
    # --- 第1步: 将ADC原始读数转换为电压值(mV) ---
    #    对应Arduino: VOLTAGE_AO = map(value, 0, 1023, 0, 5000);
    #    我们使用硬件的ADC范围 (0-32767)
    voltage_mv = linear_map(adc_value, ADC_MIN, ADC_MAX, 0, SYSTEM_VOLTAGE_MV)
    
    # --- 第2步: 将电压值(mV)转换为压力值(g) ---
    #    对应Arduino中的 if/else if/else 逻辑块
    
    # 首先处理边界情况
    if voltage_mv < EFFECTIVE_VOLTAGE_MIN_MV:
        # 如果电压低于传感器的有效下限，我们认为没有压力
        pressure_g = 0
    elif voltage_mv > EFFECTIVE_VOLTAGE_MAX_MV:
        # 如果电压高于传感器的有效上限，我们认为达到了最大压力
        pressure_g = PRESS_MAX_G
    else:
        # 如果电压在有效范围内, 则进行线性映射
        # 对应Arduino: PRESS_AO = map(VOLTAGE_AO, VOLTAGE_MIN, VOLTAGE_MAX, PRESS_MIN, PRESS_MAX);
        pressure_g = linear_map(voltage_mv, EFFECTIVE_VOLTAGE_MIN_MV, EFFECTIVE_VOLTAGE_MAX_MV, PRESS_MIN_G, PRESS_MAX_G)
        
    return pressure_g, voltage_mv 

# left sensor (right is 4)
I2C_BUS_NUMBER = 3
DEVICE_FORCE_ADDR = 0x48
bus = smbus.SMBus(I2C_BUS_NUMBER)

def ReadRawForce():
    read_bytes = bus.read_i2c_block_data(DEVICE_FORCE_ADDR, 0x0C, 2)
    return (read_bytes[0] << 8) | read_bytes[1]

def main():
    try:
        while True:
            adc_v = ReadRawForce()
            force, voltage = convert_to_pressure(adc_v)
            
            newton = force / 1000 * 9.8
            print(f"压力值: {newton:7.7f} -- {force:7.2f} g | 计算电压: {voltage:7.2f} mV | ADC原始值: {adc_v}")
            time.sleep(0.3) # 对应Arduino的delay(300)
    except KeyboardInterrupt:
        print("\n程序已停止")
        


if __name__ == "__main__":
    main()