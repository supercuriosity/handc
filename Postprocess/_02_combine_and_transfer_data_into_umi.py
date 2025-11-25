"""This module combines data from multiple times collected into a single dataset.
"""

import os
import cv2
import sys
import json
import numpy as np 
from pathlib import Path
from argparse import ArgumentParser
from scipy.interpolate import interp1d
import scipy.spatial.transform as st
from huggingface_hub.constants import HF_HOME
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
# from lerobot.datasets.factory import resolve_delta_timestamps
from dp_utils.replay_buffer import ReplayBuffer
import zarr
from dp_utils.imagecodecs_numcodecs import register_codecs, JpegXl, Zlib
register_codecs()
from collections import defaultdict
# CODEBASE_VERSION = "v2.1"
# default_cache_path = Path(HF_HOME) / "lerobot"
# HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

import concurrent.futures
from tqdm import tqdm
########################################### interpolator ###########################################


class WrappedInterp1d:
    def __init__(self, X, Y, extend_boundary=None):
        """interpolated matching from X to Y
        not necessary to sort the data"""

        sort_id = np.argsort(X)
        X = np.array(X)[sort_id]
        Y = [Y[i] for i in sort_id]
        
        self.traj_interp = interp1d(X, Y, axis=0, bounds_error=True)

        extend_boundary = 0 if extend_boundary is None else extend_boundary
        self.left_min, self.left_max = X[0]-extend_boundary, X[0]
        self.right_min, self.right_max = X[-1], X[-1]+extend_boundary
        self.left_val = Y[0]
        self.right_val = Y[-1]
        

    def __call__(self, x):
        if self.left_min < x < self.left_max:
            print(f"WARNING: extrapolating left: {self.left_min} < {x} < {self.left_max}")
            return self.left_val
        elif self.right_min < x < self.right_max:
            print(f"WARNING: extrapolating right: {self.right_min} < {x} < {self.right_max}")
            return self.right_val
        else:
            return self.traj_interp(x)


def rotation_matrix_to_quaternion(R):
        """rotation matrix to quaternion"""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            
        return [w, x, y, z]
    
def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

x_rotation_clockwise_90 = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0]
])


z_rotation_clockwise_90 = np.array([
    [ 0,  1,  0],
    [-1,  0,  0],
    [ 0,  0,  1]
])


def compute_pose_transform(raw_pose):
    # Apply transformations to the raw_pose data

    xyz = raw_pose[:3]
    rot = raw_pose[3:]
    
    combined_world_rotation = np.dot(x_rotation_clockwise_90, z_rotation_clockwise_90)
    combined_world_rotation_inverse = combined_world_rotation.T
    position_transformed = np.dot(combined_world_rotation_inverse, xyz)
    
    transformed_rotation_matrix = quaternion_to_rotation_matrix(rot)
    
    y_rotation_clockwise_90 = np.array([
                        [ 0,  0,  1],
                        [ 0,  1,  0],
                        [-1,  0,  0]
                    ])
    
    z_rotation_180 = np.array([
        [-1,  0, 0],
        [ 0, -1, 0],
        [ 0,  0, 1]
    ])
    
    transformed_rotation_matrix = np.dot(transformed_rotation_matrix, y_rotation_clockwise_90)
    transformed_rotation_matrix = np.dot(transformed_rotation_matrix, z_rotation_180)
    orientation_transformed = rotation_matrix_to_quaternion(transformed_rotation_matrix)

    raw_xyz = np.array(position_transformed)
    raw_rotation = np.array(orientation_transformed)
    return np.concatenate([raw_xyz, raw_rotation])
    
    




features =  {
        "start_pose": {
            "dtype": "float32",
            "shape": (7,), # 3 xyz, 4D pose
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]  
        },
        "end_pose": {
            "dtype": "float32",
            "shape": (7,), # 3 xyz, 4D pose
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"] 
        },
        "action": {
            "dtype": "float32",
            "shape": (10,), # 3 xyz, 4D pose + 1D gripper
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper", "none1", "none2"] 
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper", "none1", "none2"]
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": None
        },
        "observation.tactiles.left": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": None
        },
        "observation.tactiles.right": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": None
        },
        "observation.forces.left": {
            "dtype": "float32",
            "shape": (1,),
            "names": None
        },
        "observation.forces.right": {
            "dtype": "float32",
            "shape": (1,),
            "names": None
        },
        "timestamp": {
            "dtype": "float32",
            "shape": (1,),
            "names": None
        },
        "frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        },
        "episode_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        },
        "index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        },
        "task_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        }
}

def find_closest_timestamp(target_time, timestamps, values, current_idx):
    """
    Finds the value corresponding to the closest timestamp to the target_time.
    Efficiently searches starting from current_idx.
    """
    start_idx = current_idx - 10000
    best_idx = -1
    
    # Search forward from the current index
    for i in range(start_idx, len(timestamps)):
        ts = timestamps[i]
        if i > 0 and ts > target_time:
            # We've passed the target time. Decide between current and previous.
            prev_ts = timestamps[i-1]
            if abs(target_time - prev_ts) <= abs(target_time - ts):
                best_idx = i - 1
            else:
                best_idx = i
            break
    else:
        # If loop finishes, the last element is the closest
        best_idx = len(timestamps) - 1

    if best_idx == -1:
        raise ValueError(f"Could not find a suitable timestamp for target {target_time}")

    # Check if the time difference is too large
    if abs(target_time - timestamps[best_idx]) > 0.5:
        print(f"Warning: Time difference > 0.5s for target {target_time}. Closest timestamp: {timestamps[best_idx]}")
        if current_idx > 100000:
            start_idx = current_idx - 100000
        else:
            start_idx = 0
            
        best_idx = -1
        
        # Search forward from the current index
        for i in range(start_idx, len(timestamps)):
            ts = timestamps[i]
            if i > 0 and ts > target_time:
                # We've passed the target time. Decide between current and previous.
                prev_ts = timestamps[i-1]
                if abs(target_time - prev_ts) <= abs(target_time - ts):
                    best_idx = i - 1
                else:
                    best_idx = i
                break
        else:
            # If loop finishes, the last element is the closest
            best_idx = len(timestamps) - 1

        if best_idx == -1:
            raise ValueError(f"Could not find a suitable timestamp for target {target_time}")
        if abs(target_time - timestamps[best_idx]) > 0.5:
            raise ValueError(f"Time difference > 0.5s for target {target_time}. Something is wrong!")
            # return None, best_idx

    return values[best_idx], best_idx


def record_loop(args):
    
    global angle_idx, ltactile_idx, rtactile_idx, wrist_idx, lforce_idx, rforce_idx
    ltactile_idx, rtactile_idx, wrist_idx = 0, 0, 0
    
    num_workers = 4
    file_list = os.listdir(args.data_root)
    tracker_file_list = []
    time_files = args.time_file.split(" ")
    for tf in time_files:
        tracker_file_list += [f for f in file_list if f.startswith(tf) and not f.endswith("_pi")]
        
    tracker_file_list = set(sorted(tracker_file_list))
    handcap_file_list = [f+"_pi" for f in tracker_file_list]

    def preload_data(dataroot, file_list, prefix, value_key, time_key, is_npz=False):
        all_values = []
        all_times = []
        for file_name in sorted(file_list, key=lambda f: int(f.split('_')[-1].split('.')[0])):
            file_path = os.path.join(dataroot, file_name)
            if is_npz:
                data = np.load(file_path)
                # For audio, we store the file path, not the content
                all_values.extend(data[value_key])
                all_times.extend(data[time_key])
            else:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                all_values.extend(data[value_key])
                all_times.extend(data[time_key])
                
        return np.array(all_times), all_values
    
    def preload_angle_data(dataroot, file_list, prefix, value_key, time_key, is_npz=False): # as5600_preprocess, as5600_to_width,
        all_values = []
        all_times = []
        for file_name in sorted(file_list, key=lambda f: int(f.split('_')[-1].split('.')[0])):
            file_path = os.path.join(dataroot, file_name)
            if is_npz:
                data = np.load(file_path)
                # For audio, we store the file path, not the content
                all_values.extend(data[value_key])
                all_times.extend(data[time_key])
            else:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                all_values.extend(data[value_key])
                all_times.extend(data[time_key])
        all_times = np.array(all_times)
        
        return all_times, all_values

    def preload_video_data(dataroot, file_list, prefix):
        all_times = []
        all_video_paths = []
        all_frame_indices = []
        for file_name in sorted(file_list, key=lambda f: int(f.split('_')[-1].split('.')[0])):
            json_path = os.path.join(dataroot, file_name)
            video_path = os.path.join(dataroot, file_name.replace('.json', '.mp4'))
            with open(json_path, 'r') as f:
                timestamps = json.load(f)
            all_times.extend(timestamps)
            all_video_paths.extend([video_path] * len(timestamps))
            all_frame_indices.extend(range(len(timestamps)))
        
        # Combine video path and frame index into a single value
        combined_values = list(zip(all_video_paths, all_frame_indices))
        return np.array(all_times), combined_values
    
    # load angle processing functions
    
    with open(args.angle_calib_file, "r") as fp:
        width_calib = json.load(fp)["calibration"]
        widths, values = zip(*width_calib)
        
        jump_point = None
        for i in range(len(values)-1):
            if np.abs(values[i] - values[i+1]) > 2048:
                # here's a jump in the circular data
                if jump_point is not None:
                    raise ValueError(f"Multiple jumps found in calibration data: {values}")
                jump_point = i

        if jump_point is None:
            as5600_preprocess = lambda x: x
        else:
            group_1, group_2 = values[:jump_point+1], values[jump_point+1:]
            max_1, min_1 = np.max(group_1), np.min(group_1)
            max_2, min_2 = np.max(group_2), np.min(group_2)

            if min_1 > max_2:
                threshold = (min_1 + max_2) / 2
            elif min_2 > max_1:
                threshold = (min_2 + max_1) / 2
            else:
                raise ValueError(f"????why the two groups overlap???? {values}")
            
            def as5600_preprocess(x):
                if x < threshold:
                    return x + 4096
                else:
                    return x

        values = list(map(as5600_preprocess, values))
        if values[0] > values[-1]:
            values = [values[0]+200] + values + [values[-1]-200]
        else:
            values = [values[0]-200] + values + [values[-1]+200]
        widths = list(widths)
        widths = [widths[0]] + widths + [widths[-1]]
        
        as5600_to_width = WrappedInterp1d(values, widths, extend_boundary=10)
        
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    all_videos = set()
    vid_args = list()
    
    buffer_start = 0
    
    for batch_idx, (handcap_data_root, tracker_root) in enumerate(zip(handcap_file_list, tracker_file_list)):
        angle_dataroot = os.path.join(args.data_root, handcap_data_root, "angle")
        force_dataroot = os.path.join(args.data_root, handcap_data_root, "force")
        videos_dict = defaultdict(list)
        
        angle_file_list = os.listdir(angle_dataroot)
        angle_file_list = [f for f in angle_file_list if f.endswith(".json")]
        
        angle_times, angle_values = preload_angle_data(angle_dataroot, angle_file_list, "angle_", "angles", "timestamps") # , as5600_preprocess, as5600_to_width
        
        force_file_list = os.listdir(force_dataroot)
        left_force_file_list = [f for f in force_file_list if f.startswith("leftforce_") and f.endswith(".json")]
        right_force_file_list = [f for f in force_file_list if f.startswith("rightforce_") and f.endswith(".json")]
        
        left_force_times, left_force_values = preload_data(force_dataroot, left_force_file_list, "leftforce_", "forces", "timestamps")
        right_force_times, right_force_values = preload_data(force_dataroot, right_force_file_list, "rightforce_", "forces", "timestamps")
        
        print("Data pre-loading complete.")

        angle_idx, lforce_idx, rforce_idx = 0, 0, 0
        
        if batch_idx == 0:
            start_idx = args.start_eposide_idx
            end_idx = len(os.listdir(os.path.join(args.data_root, tracker_root)))
        else:
            start_idx = end_idx
            end_idx += len(os.listdir(os.path.join(args.data_root, tracker_root)))
            
        for idx in range(start_idx, end_idx):
            print(f"Processing episode {idx}...")
            
            each_episode_folder = os.path.join(args.data_root, tracker_root, f"pose_{idx}")
            
            pose_file_list = os.listdir(each_episode_folder)
            pose_file_list = [f for f in pose_file_list if f.endswith(".npz")]

            frame_episode_idx = 0
            n_frames = 0
            
            episode_timestamps = []
            for pose_file in pose_file_list:
                pose_file_path = os.path.join(each_episode_folder, pose_file)

                # load pose from tracker
                pose_data = np.load(pose_file_path)
                # process pose_data as needed
                start_pose = pose_data["pose"][0]
                start_pose = compute_pose_transform(start_pose)
                start_pose_xyz = start_pose[:3]
                start_pose_rot = st.Rotation.from_quat(start_pose[3:]).as_rotvec()
                start_pose = np.concatenate([start_pose_xyz, start_pose_rot])
                end_pose = pose_data["pose"][-1]
                end_pose = compute_pose_transform(end_pose)
                end_pose_xyz = end_pose[:3]
                end_pose_rot = st.Rotation.from_quat(end_pose[3:]).as_rotvec()
                end_pose = np.concatenate([end_pose_xyz, end_pose_rot])
                
                episode_eef_pose = []
                episode_eef_rot_axis_angle = []
                episode_gripper_width = []
                episode_left_force = []
                episode_right_force = []
                
                
                for time_idx, (each_pose, each_time_step) in enumerate(zip(pose_data["pose"], pose_data["time"])):
                    n_frames += 1
                    current_pose = compute_pose_transform(each_pose)
                    agree_value, angle_idx = find_closest_timestamp(each_time_step, angle_times, angle_values, angle_idx)
                    
                    
                    _angles = [agree_value]
                    width_list = [[as5600_to_width(as5600_preprocess(x[0]))] for x in _angles][0]
                    episode_eef_pose.append(current_pose[:3])
                    episode_eef_rot_axis_angle.append(st.Rotation.from_quat(current_pose[3:]).as_rotvec())
                    episode_gripper_width.append(width_list[0])
                    episode_timestamps.append(each_time_step)
                    
                    left_force_value, lforce_idx = find_closest_timestamp(each_time_step, left_force_times, left_force_values, lforce_idx)
                    right_force_value, rforce_idx = find_closest_timestamp(each_time_step, right_force_times, right_force_values, rforce_idx)
                    episode_left_force.append(left_force_value)
                    episode_right_force.append(right_force_value)
                
                robot_name = f'robot0'
                episode_data = dict()
                episode_data[robot_name + '_eef_pos'] = np.array(episode_eef_pose).astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle'] = np.array(episode_eef_rot_axis_angle).astype(np.float32)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(np.array(episode_gripper_width), axis=-1).astype(np.float32)
                episode_length = len(episode_eef_pose)
                episode_data[robot_name + '_demo_start_pose'] = np.repeat(start_pose[None, :], episode_length, axis=0).astype(np.float32)
                episode_data[robot_name + '_demo_end_pose'] = np.repeat(end_pose[None, :], episode_length, axis=0).astype(np.float32)
                # episode_data[robot_name + '_timestamps'] = np.array(episode_timestamps).astype(np.float32)
                out_replay_buffer.add_episode(data=episode_data, compressors=None)
                
                
                
            videos_dict[f"episode_{idx}"].append({
                'buffer_start': buffer_start,
                "time_stamps": episode_timestamps,
            })
            buffer_start += n_frames
        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    print(f"{len(all_videos)} videos used in total!")
    # write
    img_compressor = JpegXl(level=99, numthreads=1)
    name = f'camera0_rgb'
    _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + (224, 224) + (3,),
            chunks=(1,) + (224, 224) + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )
    
    out_tactile_res = (224, 224)
    # name = f'tactile0_left'
    # _ = out_replay_buffer.data.require_dataset(
    #         name=name,
    #         shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_tactile_res + (3,),
    #         chunks=(1,) + out_tactile_res + (1,),
    #         compressor=img_compressor,
    #         dtype=np.uint8
    #         )
    # name = f'tactile0_right'
    # _ = out_replay_buffer.data.require_dataset(
    #         name=name,
    #         shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_tactile_res + (3,),
    #         chunks=(1,) + out_tactile_res + (1,),
    #         compressor=img_compressor,
    #         dtype=np.uint8
    #         )
    name = "tactile_combined"
    _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_tactile_res + (3,),
            chunks=(1,) + out_tactile_res + (1,),
            compressor=img_compressor,
            dtype=np.uint8
            )
    
    def video_to_zarr(replay_buffer, mp4_path, tasks):
        # pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        # tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        ltactile_idx, rtactile_idx, wrist_idx = 0, 0, 0
        video_caps = {} 
        tasks = sorted(tasks, key=lambda x: x['time_stamps'])
        def get_frame(video_path, frame_idx, is_tactile=False):
            if video_path not in video_caps:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Error opening video file: {video_path}")
                video_caps[video_path] = cap
            
            cap = video_caps[video_path]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # If read fails, try reopening the file once.
                print(f"Warning: Failed to read frame {frame_idx} from {video_path}. Retrying...")
                cap.release()
                del video_caps[video_path]
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Error reopening video file: {video_path}")
                video_caps[video_path] = cap
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Failed to read frame {frame_idx} from {video_path} on retry.")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        
        wrist_dataroot = os.path.join(args.data_root, handcap_file_list[0], "wrist")
        tactile_dataroot = os.path.join(args.data_root, handcap_file_list[0], "tactile")
        name = 'camera0_rgb'
        img_array = replay_buffer.data[name]
        # left_tactile_name = 'tactile0_left'
        # right_tactile_name = 'tactile0_right'
        # left_tactile_array = replay_buffer.data[left_tactile_name]
        # right_tactile_array = replay_buffer.data[right_tactile_name]
        combined_tactile_name = 'tactile_combined'
        combined_tactile_array = replay_buffer.data[combined_tactile_name]
        
        wrist_file_list = os.listdir(wrist_dataroot)
        wrist_file_list = [f for f in wrist_file_list if f.endswith(".json")]
        wrist_times, wrist_values = preload_video_data(wrist_dataroot, wrist_file_list, "wrist_")
        tactile_file_list = os.listdir(tactile_dataroot)
        left_tactile_file_list = [f for f in tactile_file_list if f.startswith("lefttactile_") and f.endswith(".json")]
        right_tactile_file_list = [f for f in tactile_file_list if f.startswith("righttactile_") and f.endswith(".json")]
        left_tactile_times, left_tactile_values = preload_video_data(tactile_dataroot, left_tactile_file_list, "lefttactile_")
        right_tactile_times, right_tactile_values = preload_video_data(tactile_dataroot, right_tactile_file_list, "righttactile_")
        buffer_idx = 0
        # print(type(tasks), tasks[0])
        # print(type(tasks["time_stamps"]), len(tasks["time_stamps"]), tasks["time_stamps"][:10])
        for each_time_step in tasks[0]["time_stamps"]:
            each_time_step = float(each_time_step)
            (wrist_video_path, wrist_frame_idx), wrist_idx = find_closest_timestamp(each_time_step, wrist_times, wrist_values, wrist_idx)
            (left_tactile_video_path, left_tactile_frame_idx), ltactile_idx = find_closest_timestamp(each_time_step, left_tactile_times, left_tactile_values, ltactile_idx)
            (right_tactile_video_path, right_tactile_frame_idx), rtactile_idx = find_closest_timestamp(each_time_step, right_tactile_times, right_tactile_values, rtactile_idx)
            left_tactile_value = get_frame(left_tactile_video_path, left_tactile_frame_idx, is_tactile=True)
            right_tactile_value = get_frame(right_tactile_video_path, right_tactile_frame_idx, is_tactile=True)
            wrist_value = get_frame(wrist_video_path, wrist_frame_idx)
            wrist_value = cv2.resize(wrist_value, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            combined_tactile = np.concatenate([left_tactile_value, right_tactile_value], axis=1)
            combined_tactile = cv2.resize(combined_tactile, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            img_array[buffer_idx] = wrist_value
            combined_tactile_array[buffer_idx] = combined_tactile
            # right_tactile_array[buffer_idx] = right_tactile_value
            buffer_idx += 1
    
    
    with tqdm(total=len(vid_args), ncols=60) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, mp4_path, tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print("Dumped rgb camera")
    print([x.result() for x in completed])
    print(f"Saving ReplayBuffer to {args.output_root}/combined_data.zarr...")
    with zarr.ZipStore(f"{args.output_root}/combined_data.zarr", mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")

def main(args):
    
    if os.path.exists(args.output_root):
        print(f"[WARNING!!!] Output folder {args.output_root} already exists. Are you sure to resume? (y/n)")
        user_input = input("y/n:")
        if user_input.strip().lower() == 'n':
            print("Exiting...")
            sys.exit(0)
    
    # if user_input.strip().lower() == 'n':
        
    #     dataset = LeRobotDataset.create(
    #             repo_id = "lihongcs/pick_and_place_handcap",
    #             fps=20,
    #             root=args.output_root,
    #             use_videos=True,
    #             features=features
    #         )
    # else:
    #     ds_meta = LeRobotDatasetMetadata(
    #         repo_id = "lihongcs/pick_and_place_handcap",
    #         root=args.output_root
    #     )
    #     # delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    #     dataset = LeRobotDataset(
    #         repo_id = "lihongcs/pick_and_place_handcap",
    #         root=args.output_root,
    #         # delta_timestamps=delta_timestamps,
    #     )
    
    # with VideoEncodingManager(dataset):
    #     record_loop(dataset,
    #                 args)
    record_loop(args)
        
    
        
        
                   

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--time_file", type=str, default="20251103") # yearmonthday
    parser.add_argument("--data_root", type=str, default="handcap_data")
    parser.add_argument("--output_root", type=str, default="combined_data")
    parser.add_argument("--angle_calib_file", type=str, default="../Sensor/angle_calib_params/gripper_calibration_interpolation1111.json")
    parser.add_argument("--start_eposide_idx", type=int, default=0)

    args = parser.parse_args()
    main(args)




