
import os
import cv2
import json
import time
import torch
import threading
import numpy as np
from config import *
from Sensor import (ForceReader, 
                    AudioSensor, 
                    ForceSensor)
from Sensor.angle import AngleSensor
from Sensor.camera_process import UsbCamera, TactileUSBCamera, TactileCamera, WristCamera
from multiprocessing.managers import SharedMemoryManager

class HandCapEnv:
    # audio_device_id
    def __init__(self, tactile_camera_dev_path_dict, wrist_camera_dev_path_dict, force_sensor_dev_path, save_dir, shm_manager, *, 
                        resolution=(640, 480), tactile_resolution=(224, 224), fps=20, buffer_size=1300):
        self.resolution = resolution
        self.tactile_resolution = tactile_resolution
        self.fps = fps
        
        self.keyboard_events = []
        self.keyboard_lock = threading.Lock()
        self.keyboard_thread = None
        self.stop_keyboard_monitoring = False
        
        self.episode_id = 0
        
        print("initializing Angle Sensor...")
        self.angle_sensor = AngleSensor(
            shm_manager=shm_manager,
            capture_fps=fps,
            put_fps=fps,
            get_max_k=buffer_size,
            verbose=True
        )
        
        camera_dict = {}
        print("initializing Tactile Camera...")
        for camera_name, (camera_cls, v4l_path) in tactile_camera_dev_path_dict.items():
            camera_dict[camera_name] = TactileCamera(
                camera_cls=camera_cls,
                dev_video_path=v4l_path,
                shm_manager=shm_manager,
                resolution=(224, 224),
                capture_fps=fps,
                put_fps=fps,
                get_max_k=buffer_size,
                cap_buffer_size=1,
                # put_downsample=False,
                verbose=True
            )
        
        wrist_camera_dict = {}
        print("initializing Wrist Camera...")
        for camera_name, (camera_cls, v4l_path) in wrist_camera_dev_path_dict.items():
            wrist_camera_dict[camera_name] = WristCamera(
                camera_cls=camera_cls,
                dev_video_path=v4l_path,
                shm_manager=shm_manager,
                resolution=resolution,
                capture_fps=fps,
                put_fps=fps,
                get_max_k=buffer_size,
                cap_buffer_size=1,
                # put_downsample=False,
                verbose=True
            )
            
        force_dict = {}
        print("initializing Force Sensor...")
        for force_name, (force_cls, smbus_id, i2c_address) in force_sensor_dev_path.items():
            force_dict[force_name] = ForceSensor(
                force_cls = force_cls,
                smbus_id=smbus_id,
                i2c_address=i2c_address,
                shm_manager=shm_manager,
                capture_fps=fps,
                put_fps=fps,
                get_max_k=buffer_size,
                verbose=True
            )
            
        
        # self.audio_sensor = AudioSensor(device_id = audio_device_id,
        #                                 shm_manager=shm_manager,
        #                                 capture_fps=fps,
        #                                 put_fps=fps,
        #                                 get_max_k=buffer_size,
        #                                 verbose=True)
            
        
        self.camera_dict = camera_dict
        self.wrist_camera_dict = wrist_camera_dict
        self.force_dict = force_dict
        self.last_camera_data = {k: None for k in camera_dict.keys()}
        self.last_force_data = {k: None for k in force_dict.keys()}
        self.last_wrist_camera_data = {k: None for k in wrist_camera_dict.keys()}
        self.save_dir = save_dir
        self.global_step = 0
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        ready_flag = True
        for camera in self.camera_dict.values():
            ready_flag = ready_flag and camera.is_ready
        
        for wrist_camera in self.wrist_camera_dict.values():
            ready_flag = ready_flag and wrist_camera.is_ready

        for force in self.force_dict.values():
            ready_flag = ready_flag and force.is_ready
        
        ready_flag = ready_flag and self.angle_sensor.is_ready
        # ready_flag = ready_flag and self.audio_sensor.is_ready
        return ready_flag
    
    def start(self, wait=True):
        
        self.angle_sensor.start(wait=False)
        for camera in self.camera_dict.values():
            camera.start(wait=False)
        
        for wrist_camera in self.wrist_camera_dict.values():
            wrist_camera.start(wait=False)
        
        for force in self.force_dict.values():
            force.start(wait=False)
        # self.audio_sensor.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for camera in self.camera_dict.values():
            camera.stop(wait=False)
        for wrist_camera in self.wrist_camera_dict.values():
            wrist_camera.stop(wait=False)
        for force in self.force_dict.values():
            force.stop(wait=False)
        # self.audio_sensor.stop(wait=False)
        self.angle_sensor.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.angle_sensor.start_wait()
        for camera in self.camera_dict.values():
            camera.start_wait()
        for wrist_camera in self.wrist_camera_dict.values():
            wrist_camera.start_wait()
        for force in self.force_dict.values():
            force.start_wait()
        # self.audio_sensor.start_wait()
        
    def stop_wait(self):
        for camera in self.camera_dict.values():
            camera.end_wait()
        for wrist_camera in self.wrist_camera_dict.values():
            wrist_camera.end_wait()
        for force in self.force_dict.values():
            force.end_wait()
        # self.audio_sensor.end_wait()
        self.angle_sensor.end_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_and_save_data(self):
        assert self.is_ready
        start_time = time.monotonic()
        for camera_name, camera in self.camera_dict.items():
            # print("Fetching data", camera_name)
            self.last_camera_data[camera_name] = camera.get_all()
        
        for camera_name, wrist_camera in self.wrist_camera_dict.items():
            # print("Fetching wrist camera data", camera_name)
            self.last_wrist_camera_data[camera_name] = wrist_camera.get_all()
        
        for force_name, force in self.force_dict.items():
            # print("Fetching force data", force_name)
            self.last_force_data[force_name] = force.get_all()
        
        # print("Fetching angle data...")
        self.last_angle_data = self.angle_sensor.get_all()
        # print("Fetching audio data...")
        # self.last_audio_data = self.audio_sensor.get_all()
        
        # print("Takes", time.monotonic() - start_time)
        threads = []
        
        print("Saving data...")
        # tactile camera
        for camera_name, camera_data in self.last_camera_data.items():
            thread = threading.Thread(target=self._save_tactile_video_and_json, args=("tactile/" + camera_name + "tactile", camera_data))
            threads.append(thread)
            thread.start()
        # wrist camera
        for camera_name, camera_data in self.last_wrist_camera_data.items():
            thread = threading.Thread(target=self._save_video_and_json, args=("wrist/" + camera_name + "camera", camera_data))
            threads.append(thread)
            thread.start()
        # force
        for force_name, force_data in self.last_force_data.items():
            force_thread = threading.Thread(target=self._save_force_data, args=("force/" + force_name + "force", force_data,))
            threads.append(force_thread)
            force_thread.start()
        # audio 
        # if self.last_audio_data:
        #     audio_thread = threading.Thread(target=self._save_audio_data, args=(self.last_audio_data,))
        #     threads.append(audio_thread)
        #     audio_thread.start()
        # angle
        if self.last_angle_data:
            angle_thread = threading.Thread(target=self._save_angle_data, args=(self.last_angle_data,))
            threads.append(angle_thread)
            angle_thread.start()
        
        for thread in threads:
            thread.join()
        
        self.global_step += 1
        self.last_camera_data = {k: None for k in self.camera_dict.keys()}
        self.last_wrist_camera_data = {k: None for k in self.wrist_camera_dict.keys()}
        self.last_force_data = {k: None for k in self.force_dict.keys()}
        self.last_angle_data = None
        # self.last_audio_data = None
    
    def _save_video_and_json(self, camera_name, camera_data):
        start_time = time.monotonic()
        save_dir = os.path.join(self.save_dir, camera_name.split('/')[0])
        os.makedirs(save_dir, exist_ok=True)
        video_writer = cv2.VideoWriter(
            os.path.join(save_dir, f"{camera_name.split('/')[1]}_{self.global_step}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, self.resolution)

        video_shape = camera_data['color'].shape
        
        assert len(video_shape) == 4, str(video_shape)  # T,w,h,c
        
        for frame in camera_data['color']:
            video_writer.write(frame)
        video_writer.release()
        save_dir = os.path.join(self.save_dir, camera_name.split('/')[0])
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{camera_name.split('/')[1]}_{self.global_step}.json"), 'w') as json_file:
            timestamps = camera_data["timestamp"].tolist()
            json.dump(timestamps, json_file)

        print("Saved camera data")
    
    def _save_tactile_video_and_json(self, camera_name, camera_data):
        start_time = time.monotonic()
        save_dir = os.path.join(self.save_dir, camera_name.split('/')[0])
        os.makedirs(save_dir, exist_ok=True)
        video_writer = cv2.VideoWriter(
            os.path.join(save_dir, f"{camera_name.split('/')[1]}_{self.global_step}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, (224,224))

        video_shape = camera_data['color'].shape
        
        assert len(video_shape) == 4, str(video_shape)  # T,w,h,c
        
        for frame in camera_data['color']:
            video_writer.write(frame)
        video_writer.release()
        save_dir = os.path.join(self.save_dir, camera_name.split('/')[0])
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{camera_name.split('/')[1]}_{self.global_step}.json"), 'w') as json_file:
            timestamps = camera_data["timestamp"].tolist()
            json.dump(timestamps, json_file)

        print("Saved camera data")

    def _save_realsense_video_and_json(self, camera_name, camera_data):
        start_time = time.monotonic()
        video_writer_colored = cv2.VideoWriter(
            os.path.join(self.save_dir, f"{camera_name}_colored_{self.global_step}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, self.resolution)
        video_writer_depth = cv2.VideoWriter(
            os.path.join(self.save_dir, f"{camera_name}_depth_{self.global_step}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, self.resolution)

        video_shape = camera_data['color'].shape
        
        assert len(video_shape) == 4, str(video_shape)  # T,w,h,c
        
        for frame in camera_data['color']:
            video_writer_colored.write(frame)
        video_writer_colored.release()
        for frame in camera_data['depth']:
            video_writer_depth.write(frame)
        video_writer_depth.release()

        with open(os.path.join(self.save_dir, f"{camera_name}_{self.global_step}.json"), 'w') as json_file:
            timestamps = camera_data["timestamp"].tolist()
            json.dump(timestamps, json_file)
        print("Saved realsese data")
    
    def _save_angle_data(self, angle_data):
        start_time = time.monotonic()
        save_dir = os.path.join(self.save_dir, "angle")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"angle_{self.global_step}.json"), 'w') as json_file:
            angles = angle_data['angle'].tolist()
            data = angle_data['data'].tolist()
            timestamps = angle_data["timestamp"].tolist()
            json.dump({'angles': angles, 'data':data, 'timestamps': timestamps}, json_file)
        print("\033[92m Successfully saved angle&pose data \033[0m")
        
    def _save_force_data(self, force_name, force_data):
        start_time = time.monotonic()
        save_dir = os.path.join(self.save_dir, force_name.split('/')[0])
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{force_name.split('/')[1]}_{self.global_step}.json"), 'w') as json_file:
            forces = force_data['force'].tolist()
            timestamps = force_data["timestamp"].tolist()
            json.dump({'forces': forces, 'timestamps': timestamps}, json_file)
        print("\033[92m Successfully saved force data \033[0m")
    
    def _save_audio_data(self, audio_data):
        start_time = time.monotonic()
        save_dir = os.path.join(self.save_dir, "audio")
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"audio_{self.global_step}.npz")
        audio = audio_data['audio_data']
        np.savez_compressed(output_path, audio=audio, timestamp=audio_data['timestamp'])


def main():
    
    # define tactile device
    tactile_camera_dev_path_dict = {}
    for side, device in TACTILE_CAMERA.items():
        tactile_camera_dev_path_dict[side] = [TactileUSBCamera, device]
    
    # define force device
    force_dev_path_dict = {}
    for side, device in FORCE_SENSOR.items():
        force_dev_path_dict[side] = [ForceReader, device['smbus_id'], device['i2c_address']]
    
    # define wrist camera device
    wrist_camera_dev_path_dict = {}
    for side, device in WRIST_CAMERA.items():
        wrist_camera_dev_path_dict[side] = [UsbCamera, device]

    # audio_device_id = AUDIO_DEVICE_ID
    
    save_dir = "data/handcap_" + str(time.time())
    assert not os.path.exists(save_dir), "path exists"
    os.makedirs(save_dir)
    
    save_per_second = 5.0
    
    with SharedMemoryManager() as shm_manager:
        # audio_device_id
        with HandCapEnv(tactile_camera_dev_path_dict, wrist_camera_dev_path_dict, force_dev_path_dict, save_dir, shm_manager) as env:
            while True:
                print('[DEBUG] Saving')
                start_time = time.monotonic()
                env.get_and_save_data()
                duration = time.monotonic()-start_time
                if duration > save_per_second:
                    print(f"WARNING: save duration exeeds the save gap: {duration} > {save_per_second}")
                else:
                    print(f"[DEBUG] {duration} {save_per_second}")
                    time.sleep(save_per_second - duration)
                

            print("End.")

if __name__ == "__main__":
    main()