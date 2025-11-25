import sys
import time
import enum
import smbus
import numpy as np
import pybullet as pb
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
# sys.path.append('/root/workspace/HandCap/umi')
from umi.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty


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

class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2

class ForceReader:
    def __init__(self, bus_id, device_address):
        
        self.ADC_MIN = 0
        self.ADC_MAX = 32767  
        self.SYSTEM_VOLTAGE_MV = 5000
        self.EFFECTIVE_VOLTAGE_MIN_MV = 500  
        self.EFFECTIVE_VOLTAGE_MAX_MV = 5000 
        self.PRESS_MIN_G = 50
        self.PRESS_MAX_G = 1000
        
        self.bus = smbus.SMBus(bus_id)
        self.address = device_address

    def get_data(self):
        # Read two bytes of data from the sensor
        try:
            read_bytes = self.bus.read_i2c_block_data(self.address, 0x0C, 2)
            raw_value = (read_bytes[0] << 8) | read_bytes[1]  # Combine the two bytes
            t_cal = time.time()
            
            force, voltage = convert_to_pressure(raw_value)
            
            newton = force / 1000 * 9.8
            return newton, t_cal
        except Exception as e:
            print(f"Error reading from force sensor: {e}")
            return None, time.time()
        
    def close(self):
        self.bus.close()



class ForceSensor(mp.Process):
    
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes
    
    def __init__(
            self,
            force_cls,
            shm_manager: 'SharedMemoryManager',
            smbus_id,
            i2c_address,
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            get_max_k=120,
            num_threads=2,
            verbose=False
    ):
        super().__init__()
        
        if put_fps is None:
            put_fps = capture_fps
        
        examples = {
            'force': np.ones(shape=(1,), dtype=np.float32) * 10086,
            'timestamp': 0.0,
            'step_idx': 0
        }
        ring_buffer = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=get_max_k
        )

        # create command queue
        examples = {
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )
        
        # copied variables
        self.force_cls = force_cls
        self.shm_manager = shm_manager
        self.smbus_id = smbus_id
        self.i2c_address = i2c_address
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.num_threads = num_threads
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.command_queue = command_queue
        
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    
    @property
    def queue_count(self) -> int:
        return self.ring_buffer.qsize() if self.ring_buffer else 0

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)

    def get_all(self):
        if self.ring_buffer.empty():
            return None
        data = self.ring_buffer.get_all()
        return data
    
    def run(self):
        threadpool_limits(self.num_threads)
        force_reader = self.force_cls(self.smbus_id, self.i2c_address)
        try:
            fps = self.capture_fps
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            # reuse frame buffer
            iter_idx = 0
            t_start = time.time()
                
            while not self.stop_event.is_set():
                iter_start_time = time.monotonic()
                force, t_cal = force_reader.get_data()
                
                data = dict()
                
                data["force"] = force
                
                put_data = data
                
                if self.put_downsample:
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[t_cal],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            # this is non in first iteration
                            # and then replaced with a concrete number
                            next_global_idx=put_idx,
                            # continue to pump frames even if not started.
                            # start_time is simply used to align timestamps.
                            allow_negative=True
                        )

                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = t_cal
                        self.ring_buffer.put(put_data) # , wait=False)
                
                else:
                    step_idx = int((t_cal - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = t_cal
                    self.ring_buffer.put(put_data) # , wait=False)
                
                if iter_idx == 0:
                    self.ready_event.set()
                
                
                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    # print(f'[UvcCamera {self.dev_video_path}] FPS {frequency}')
                    pass

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                    elif cmd == Command.START_RECORDING.value:
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                    elif cmd == Command.STOP_RECORDING.value:
                        pass

                iter_idx += 1

                iter_duration = time.monotonic() - iter_start_time
                if iter_duration < 1./fps:
                    time.sleep(1./fps-iter_duration)

        finally:
            force_reader.close()