import sys
import time     
import numpy as np
import pybullet as pb
import sounddevice as sd
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
# sys.path.append('/root/workspace/HandCap/umi')
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty


class AudioReader:
    def __init__(self, device_id, fps):

        # audio params
        self.fps = fps
        self.SAMPLE_RATE = 44100      #  Sampling rate (Hz)
        self.CHANNELS = 1             # Number of channels (recommended mono for compatibility)
        self.DURATION = 5

        self.frames_per_block = int(self.SAMPLE_RATE / fps)
        self.device_info = sd.query_devices(device_id)
    
    def get_data(self):
        tm = time.time()
        audio_block = sd.rec(
            self.frames_per_block,
            samplerate=self.SAMPLE_RATE,
            channels=1,
            device=None,  # the device index.
            dtype='int16'
        )
        return audio_block, tm
    
    def close(self):
        sd.stop()
        


class AudioSensor(mp.Process):
    def __init__(self,
                 device_id,
                 shm_manager, 
                 capture_fps=30, 
                 audio_sample_rate=44100, 
                 put_fps=None,
                 put_downsample=True,
                 get_max_k=120,
                 num_threads=2,
                 verbose=True,
                 **kwargs):
        # set audio-specific params
        super().__init__()
        
        if put_fps is None:
            put_fps = capture_fps
            
        self.fps = capture_fps
        self.audio_sample_rate = audio_sample_rate
        self.frames_per_block = int(audio_sample_rate / self.fps)
        
        examples = {
            'audio_data': np.ones(shape=(self.frames_per_block, 1), dtype=np.uint16),
            'timestamp': 0.0,
            'step_idx': 0
        }
        
        ring_buffer = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=get_max_k
        )
        self.device_id = device_id
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
        
        
        
    def read_audio_data(self):
        """load audio data from microphone"""
        audio_block = sd.rec(
            self.frames_per_block,
            samplerate=self.audio_sample_rate,
            channels=1,
            device=self.device_id,  # the default input device.
            dtype='int16'
        )
        sd.wait()
        return audio_block
    
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
    
    def run(self):
        threadpool_limits(self.num_threads)
        
        try:
            put_start_time = time.time()
            iter_idx = 0
            t_start = time.time()
            
            if iter_idx == 0:
                self.ready_event.set()
            
            while not self.stop_event.is_set():
                iter_start_time = time.monotonic()
                
                # load audio data
                audio_data = self.read_audio_data()
                
                t_cal = time.time()
                step_idx = int((t_cal - put_start_time) * self.put_fps)
                
                
                data = {
                    'audio_data': audio_data,
                    'timestamp': t_cal,
                    'step_idx': step_idx
                }
                
                put_data = data
                self.ring_buffer.put(put_data)
                
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                
                
                iter_idx += 1
                
                # if self.verbose:
                #     print(f'[AudioSensor] FPS {frequency}')
                    
                # keep the fps to preset value
                iter_duration = time.monotonic() - iter_start_time
                if iter_duration < 1. / self.capture_fps:
                    time.sleep(1. / self.capture_fps - iter_duration)
                
                if iter_idx == 0:
                   
                    self.ready_event.set()
                
        finally:
            sd.stop()