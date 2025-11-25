
import os
import cv2
import sys
import json
import numpy as np 
from pathlib import Path
from argparse import ArgumentParser
from huggingface_hub.constants import HF_HOME
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

CODEBASE_VERSION = "v2.1"
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

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
    
    
    
def get_frame_at_index(video_path, frame_index):
    """"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error to open video file: {video_path}")
    
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    
    cap.release()
    
    if ret:
        return frame
    else:
        return None



features =  {
        "action": {
            "dtype": "float64",
            "shape": (8,) # 3 xyz, 4D pose + 1D gripper
        },
        "observation.state": {
            "dtype": "float64",
            "shape": (8,)
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": None
        },
        "observation.tactiles.left": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": None
        },
        "observation.tactiles.right": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": None
        },
        "observation.forces.left": {
            "dtype": "float64",
            "shape": (1,)
        },
        "observation.forces.right": {
            "dtype": "float64",
            "shape": (1,)
        },
        "observation.audio": {
            "dtype": "float32",
            "shape": (1837, 1)
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": None
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        }
}

def find_closest_timestamp(target_time, timestamps, values, current_idx):
    """
    Finds the value corresponding to the closest timestamp to the target_time.
    Efficiently searches starting from current_idx.
    """
    start_idx = current_idx
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
    if abs(target_time - timestamps[best_idx]) > 1.0:
        raise ValueError(f"Time difference > 1s for target {target_time}. Something is wrong!")

    return values[best_idx], best_idx


@safe_stop_image_writer
def record_loop(dataset,
                args):
    handcap_data_root = args.handcap_data_root
    tracker_root = args.tracker_root
    
    angle_dataroot = os.path.join(handcap_data_root, "angle")
    audio_dataroot = os.path.join(handcap_data_root, "audio")
    force_dataroot = os.path.join(handcap_data_root, "force")
    tactile_dataroot = os.path.join(handcap_data_root, "tactile")
    wrist_dataroot = os.path.join(handcap_data_root, "wrist")
    
    
    angle_file_list = os.listdir(angle_dataroot)
    angle_file_list = [f for f in angle_file_list if f.endswith(".json")]
    tactile_file_list = os.listdir(tactile_dataroot)
    left_tactile_file_list = [f for f in tactile_file_list if f.startswith("lefttactile_") and f.endswith(".json")]
    right_tactile_file_list = [f for f in tactile_file_list if f.startswith("righttactile_") and f.endswith(".json")]

    force_file_list = os.listdir(force_dataroot)
    left_force_file_list = [f for f in force_file_list if f.startswith("leftforce_") and f.endswith(".json")]
    right_force_file_list = [f for f in force_file_list if f.startswith("rightforce_") and f.endswith(".json")]

    audio_file_list = os.listdir(audio_dataroot)
    audio_file_list = [f for f in audio_file_list if f.endswith(".npz")]
    
    wrist_file_list = os.listdir(wrist_dataroot)
    wrist_file_list = [f for f in wrist_file_list if f.endswith(".json")]
    
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
    
    angle_times, angle_values = preload_data(angle_dataroot, angle_file_list, "angle_", "angles", "timestamps")
    
    left_tactile_times, left_tactile_values = preload_video_data(tactile_dataroot, left_tactile_file_list, "lefttactile_")
    right_tactile_times, right_tactile_values = preload_video_data(tactile_dataroot, right_tactile_file_list, "righttactile_")
    
    wrist_times, wrist_values = preload_video_data(wrist_dataroot, wrist_file_list, "wrist_")
    
    left_force_times, left_force_values = preload_data(force_dataroot, left_force_file_list, "leftforce_", "forces", "timestamps")
    right_force_times, right_force_values = preload_data(force_dataroot, right_force_file_list, "rightforce_", "forces", "timestamps")
    
    audio_times, audio_values = preload_data(audio_dataroot, audio_file_list, "audio_", 'audio', 'timestamp', is_npz=True)
    print("Data pre-loading complete.")
    
    angle_idx, ltactile_idx, rtactile_idx, wrist_idx, lforce_idx, rforce_idx, audio_idx = 0, 0, 0, 0, 0, 0, 0
    
    # --- Video capture objects ---
    video_caps = {} 
    
    
    for idx in range(args.num_episodes):
        each_episode_folder = os.path.join(tracker_root, f"pose_{idx}")
        
        pose_file_list = os.listdir(each_episode_folder)
        pose_file_list = [f for f in pose_file_list if f.endswith(".npz")]

        frame_episode_idx = 0
        
        for pose_file in pose_file_list:
            pose_file_path = os.path.join(each_episode_folder, pose_file)

            # load pose from tracker
            pose_data = np.load(pose_file_path)
            # process pose_data as needed

            for time_idx, (each_pose, each_time_step) in enumerate(zip(pose_data["pose"], pose_data["time"])):
                
                current_pose = compute_pose_transform(each_pose)
                
                agree_value, angle_idx = find_closest_timestamp(each_time_step, angle_times, angle_values, angle_idx)
                
                (left_tactile_video_path, left_tactile_frame_idx), ltactile_idx = find_closest_timestamp(each_time_step, left_tactile_times, left_tactile_values, ltactile_idx)
                (right_tactile_video_path, right_tactile_frame_idx), rtactile_idx = find_closest_timestamp(each_time_step, right_tactile_times, right_tactile_values, rtactile_idx)
                
                (wrist_video_path, wrist_frame_idx), wrist_idx = find_closest_timestamp(each_time_step, wrist_times, wrist_values, wrist_idx)

                left_force_value, lforce_idx = find_closest_timestamp(each_time_step, left_force_times, left_force_values, lforce_idx)
                right_force_value, rforce_idx = find_closest_timestamp(each_time_step, right_force_times, right_force_values, rforce_idx)
                audio_value, audio_idx = find_closest_timestamp(each_time_step, audio_times, audio_values, audio_idx)

                def get_frame(video_path, frame_idx):
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
                    return frame

                left_tactile_value = get_frame(left_tactile_video_path, left_tactile_frame_idx)
                right_tactile_value = get_frame(right_tactile_video_path, right_tactile_frame_idx)
                wrist_value = get_frame(wrist_video_path, wrist_frame_idx)
                
                if time_idx == 0:
                    current_frame = {"observation.state": np.concatenate([current_pose, agree_value]),
                                     "observation.images.wrist": wrist_value,
                                     "observation.tactiles.left": left_tactile_value,
                                     "observation.tactiles.right": right_tactile_value,
                                     "observation.forces.left": np.array(left_force_value),
                                     "observation.forces.right": np.array(right_force_value),
                                     "observation.audio": audio_value}
                else:
                    next_frame = {"observation.state": np.concatenate([current_pose, agree_value]),
                                  "observation.images.wrist": wrist_value,
                                  "observation.tactiles.left": left_tactile_value,
                                  "observation.tactiles.right": right_tactile_value,
                                  "observation.forces.left": np.array(left_force_value),
                                  "observation.forces.right": np.array(right_force_value),
                                  "observation.audio": audio_value}
                    
                    current_frame["action"] = next_frame["observation.state"]
                    
                    dataset.add_frame(current_frame, task="wipe_board")
                    # if frame_episode_idx % 100 == 0:
                    print(f"Added frame {frame_episode_idx + 1} to episode {idx}")
                    
                    frame_episode_idx += 1
                    current_frame = next_frame
            
        dataset.save_episode()          
    for cap in video_caps.values():
        cap.release()

def main(args):
    
    if os.path.exists(args.output_root):
        print(f"[WARNING!!!] Output folder {args.output_root} already exists. Whether to overwrite it? (y/n)")
        user_input = input("y/n:")
        user_input = user_input.strip().lower()
        if user_input != 'y':
            print("Exiting without overwriting.")
            sys.exit(0)
        else:
            print(f"Overwriting the existing folder {args.output_root}...")
            os.system(f"rm -rf {args.output_root}")
    dataset = LeRobotDataset.create(
            repo_id = "lihongcs/wipe_board",
            fps=30,
            root=args.output_root,
            use_videos=True,
            features = features
        )
    with VideoEncodingManager(dataset):
        record_loop(dataset,
                    args)
        
        
                   

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--handcap_data_root", type=str, default="handcap_data")
    parser.add_argument("--tracker_root", type=str, default="tracker_data")
    parser.add_argument("--output_root", type=str, default="combined_data")
    parser.add_argument("--num_episodes", type=int, default=100)

    args = parser.parse_args()
    main(args)







