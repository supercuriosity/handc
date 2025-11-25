import os
import sys
import time
import json 
import math
import smbus
import numpy as np
import pybullet as pb
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty

DEVICE_AS5600 = 0x36  # Default device I2C address
AS5600_BUS = smbus.SMBus(2)

def read_rotary_angle():  # Read angle (0-360 represented as 0-4096)
    read_bytes = AS5600_BUS.read_i2c_block_data(DEVICE_AS5600, 0x0C, 2)
    return (read_bytes[0] << 8) | read_bytes[1]


def l(theta):
    return 27*np.cos(theta) + np.sqrt(39**2 - (11 + 27*np.sin(theta))**2)


class AngleSensor(mp.Process):
    def __init__(
            self,
            shm_manager: 'SharedMemoryManager',
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            get_max_k=120,
            num_threads=2,
            verbose=False
    ):
        super().__init__()
        
        calibed_path = "Sensor/angle_calib_params/gripper_calibration.json"
        
        assert os.path.exists(calibed_path), f"Calibration file {calibed_path} does not exist. Please run gripper_calibration.py first."
        
        with open(calibed_path, "r") as f:
            calib_data = json.load(f)
        
        self.theta_0 = calib_data["theta_0"][0]
        self.b = calib_data["b"][0]
        self.reverse = calib_data["reverse"]
        self.x_0 = calib_data["x_0"]
        self.x_1 = calib_data["x_1"]
        
        if put_fps is None:
            put_fps = capture_fps

        # create ring buffer
        examples = {
            'angle': np.ones(shape=(1,), dtype=np.float32),
            'data': np.array([-0.0557848 , -0.103198  , -0.0541233 ,  0.02942063,  0.08058948,
       -0.40684715,  0.90945872]),
            'timestamp': 0.0,
            'step_idx': 0
        }
        
        ring_buffer = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=get_max_k
        )

        # copied variables
        self.shm_manager = shm_manager
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.num_threads = num_threads
        self.verbose = verbose

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        c = pb.connect(pb.DIRECT)
        vis_sp = []
        c_code = c_code = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]
        

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
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
    
    
    def get_gripper_width(self, use_cm=True):
        """
        返回夹爪宽度，默认单位为cm
        传入use_cm=False则返回mm
        """
        # def transformed_read(x):
        #     """
        #     将原始传感器读数转换为线性值。
        #     通过使用取模运算符处理0-4096的环绕情况。
        #     """
        #     if self.reverse:
        #         # sensor value decreases as the gripper opens
        #         return (self.x_0 - x + 4096) % 4096
        #     else:
        #         # sensor value increases as the gripper opens
        #         return (x - self.x_0 + 4096) % 4096

        raw = read_rotary_angle()
        # num = transformed_read(raw)

        # theta = num / 4096 * 360 / 180 * math.pi
        # width = 2 * l(self.theta_0 - theta) + self.b
        # if use_cm:
        #     width /= 10
        
        # # width * 0.01
        
        return raw

    # ========= interval API ===========
    def run(self):
        
        threadpool_limits(self.num_threads)
        try:
            fps = self.capture_fps
            put_idx = None
            put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            
            if iter_idx == 0:
                self.ready_event.set()
                
            while not self.stop_event.is_set():
                iter_start_time = time.monotonic()
                # TODO: revise
                angle = self.get_gripper_width()
                t_cal = time.time()
                step_idx = int((t_cal - put_start_time) * self.put_fps)
                data = {
                    'angle': np.array([angle], dtype=np.float32),
                    'timestamp': t_cal,
                    'step_idx': step_idx
                }

                put_data = data

                self.ring_buffer.put(put_data)

                # signal ready
                
                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                # if self.verbose:
                #     print(f'[AngleSensor] FPS {frequency}')

                iter_idx += 1

                iter_duration = time.monotonic() - iter_start_time
                if iter_duration < 1. / self.capture_fps:
                    time.sleep(1. / self.capture_fps - iter_duration)
                if iter_idx == 0:
                   
                    self.ready_event.set()

        finally:
            pass