import smbus
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64 # 使用标准浮点数消息


ADC_MIN = 0
ADC_MAX = 32767  

SYSTEM_VOLTAGE_MV = 5000

EFFECTIVE_VOLTAGE_MIN_MV = 500  
EFFECTIVE_VOLTAGE_MAX_MV = 5000 

PRESS_MIN_G = 50
PRESS_MAX_G = 1000


def linear_map(value, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return out_min
    return out_min + float(value - in_min) * float(out_max - out_min) / float(in_max - in_min)

def convert_to_pressure(adc_value):
    voltage_mv = linear_map(adc_value, ADC_MIN, ADC_MAX, 0, SYSTEM_VOLTAGE_MV)
    
    if voltage_mv < EFFECTIVE_VOLTAGE_MIN_MV:
        pressure_g = 0
    elif voltage_mv > EFFECTIVE_VOLTAGE_MAX_MV:
        pressure_g = PRESS_MAX_G
    else:
        pressure_g = linear_map(voltage_mv, EFFECTIVE_VOLTAGE_MIN_MV, EFFECTIVE_VOLTAGE_MAX_MV, PRESS_MIN_G, PRESS_MAX_G)
        
    return pressure_g, voltage_mv 


class LeftForcePublisher(Node):
    def __init__(self):
        super().__init__('sensor_node')
        self.publisher_ = self.create_publisher(Float64, '/sensor/force_left', 10)
        timer_period = 0.1 
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.bus = smbus.SMBus(3)
        
    def ReadForce(self):
        DEVICE_force = 0x48  # Default device I2C address
        read_bytes = self.bus.read_i2c_block_data(DEVICE_force, 0x0C, 2)
        adc_v = (read_bytes[0]<<8) | read_bytes[1]
        
        force, voltage = convert_to_pressure(adc_v)
            
        newton = force / 1000 * 9.8
        
        return float(newton)
    
    def timer_callback(self):
        force = self.ReadForce()  # 读取角度
        msg = Float64()
        msg.data = force
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing left force: {force}')

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    angle_publisher = LeftForcePublisher()
    try:
        rclpy.spin(angle_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        angle_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()