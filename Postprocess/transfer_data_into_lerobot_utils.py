
import os
import zarr
import torch
import datasets
import PIL.Image
import numpy as np
from pathlib import Path
from huggingface_hub.constants import HF_HOME
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.video_utils import get_safe_default_codec
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import (DEFAULT_IMAGE_PATH,
                                    get_hf_features_from_features,
                                    hf_transform_to_torch,
                                    validate_frame)

CODEBASE_VERSION = "v2.1"
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

from collections.abc import Callable


features =  {
        "action": {
            "dtype": "float32",
            "shape": (8,) # 3 xyz, 4D pose + 1D gripper
        },
        "observation.state": {
            "dtype": "float32",
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
                224,
                224,
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
                224,
                224,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": None
        },
        "observation.force.left": {
            "dtype": "float32",
            "shape": (1,)
        },
        "observation.force.right": {
            "dtype": "float32",
            "shape": (1,)
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






@safe_stop_image_writer
def record_loop(dataset,
                zarr_store,
                episode_index):
    
    
    # TODO: get the data from two source, observation and 6D pose
    # observation: external camera, wrist camera, tactile images, force
    # action: 6D pose + gripper
    # state: 6D pose + gripper 
    
    if episode_index == 0:
        start_frame = 0
        end_frame = zarr_store["data"]["episode_ends"][episode_index]
    else:
        start_frame = zarr_store["data"]["episode_ends"][episode_index - 1]
        end_frame = zarr_store["data"]["episode_ends"][episode_index]
        
    external_img = zarr_store["data"]["external_img"][start_frame:end_frame]
    wrist_img = zarr_store["data"]["left_wrist_img"][start_frame:end_frame]
    action = zarr_store["data"]["action"][start_frame:end_frame]
    state = zarr_store["data"]["state"][start_frame:end_frame]
    
    num_timetamp = external_img.shape[0]
    
    for i in range(num_timetamp):
        frame = {
            "observation.state": state[i].squeeze() if state[i].ndim > 1 else state[i],
            "observation.images.side": external_img[i],
            "observation.images.wrist": wrist_img[i],
            "action": action[i].squeeze() if action[i].ndim > 1 else action[i],
        }
        
        # display_cameras = True
        # if display_cameras:
        #     image_keys = [key for key in observation if "image" in key]
        #     positions = {
        #         "observation.images.head": (700, 0),         # top right
        #         "observation.images.side": (0, 520),         # bottom left
        #         "observation.images.wrist": (700, 520)        # bottom right
        #     }
        #     for key in image_keys:
        #         cv2.imshow(key, cv2.cvtColor(
        #             observation[key].numpy(), cv2.COLOR_RGB2BGR))
        #         # move the window to the top left corner
        #         cv2.moveWindow(key, positions[key][0], positions[key][1])
        # Add frame to the dataset
        dataset.add_frame(frame, task="wipe_board")
        
        if i % 100 == 0:
            print(f"Added frame {i + 1}/{num_timetamp} to episode {episode_index}")
    

# load .zarr files
zarr_path = "/home/lixingyu/workspace_lxy2/workspace/robotpolicy/Data/wipe_board_high_freq_downsample1_zarr/wipe_board.zarr"

zarr_store = zarr.open(zarr_path, mode="r")

num_episodes = len(zarr_store["data"]["episode_ends"])

with VideoEncodingManager(dataset):
    
    for recorded_episodes in range(num_episodes):
        print(f"Recording episode {recorded_episodes + 1}/{num_episodes}")
        record_loop(
                zarr_store=zarr_store,
                dataset=dataset,
                episode_index=recorded_episodes
            )

        # exit()
        dataset.save_episode()
print(zarr_store.tree())