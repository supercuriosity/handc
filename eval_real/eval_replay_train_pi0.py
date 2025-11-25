"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os
import copy
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
# from omegaconf import OmegaConf
import json
# from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
# from umi.common.cv_util import (
#     parse_fisheye_intrinsics,
#     FisheyeRectConverter
# )
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
# from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.pose_util import *

# OmegaConf.register_new_resolver("eval", eval, replace=True)


from diffusion_policy.common.replay_buffer import ReplayBuffer

import zarr
import numpy as np

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()
import sys
import dataclasses
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import jax
sys.path.append('/home/xuyue/tactile_umi_grav/openpi/src')
from openpi.models import model as _model
from openpi.policies import umi_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

class ReplayFlexivUmiEnv:
    def read_replay_buffer(self, zarr_path, episode_id):
        with zarr.ZipStore(zarr_path, mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, 
                store=zarr.MemoryStore()
            )
        
        episode_start = replay_buffer.meta.episode_ends[episode_id-1] if episode_id > 0 else 0
        episode_end = replay_buffer.meta.episode_ends[episode_id]

        # ├── data
        # │   ├── camera0_rgb (9489, 224, 224, 3) uint8
        # │   ├── robot0_demo_end_pose (9489, 6) float64
        # │   ├── robot0_demo_start_pose (9489, 6) float64
        # │   ├── robot0_eef_pos (9489, 3) float32
        # │   ├── robot0_eef_rot_axis_angle (9489, 3) float32
        # │   └── robot0_gripper_width (9489, 1) float32
        # └── meta
        #     └── episode_ends (29,) int64

        return {
            "camera0_rgb":          replay_buffer['camera0_rgb'][episode_start:episode_end],   # N, 224,224,3
            "robot0_eef_pos":       replay_buffer['robot0_eef_pos'][episode_start:episode_end],
            "robot0_eef_rot_axis_angle": replay_buffer['robot0_eef_rot_axis_angle'][episode_start:episode_end],
            "robot0_gripper_width": replay_buffer['robot0_gripper_width'][episode_start:episode_end],
        }


    def __init__(self, zarr_path, episode_id=0, obs_horizon=2, robot_downsample=1, camera_downsample=1):
        self.data = self.read_replay_buffer(zarr_path, episode_id)
        self.index = 0

        self.obs_horizon = obs_horizon
        self.robot_downsample = robot_downsample
        self.camera_downsample = camera_downsample



    def get_obs(self, action_steps):

        if self.index > len(self.data['camera0_rgb']) - self.obs_horizon:
            raise StopIteration

        obs_data = {
            'camera0_rgb': None,
            'robot0_eef_pos': None,
            'robot0_eef_rot_axis_angle': None,
            'robot0_gripper_width': None,
        }
        next_data = {k: None for k in obs_data.keys()}

        for key in obs_data:
            obs_data[key] = self.data[key][self.index: self.index+self.obs_horizon]
            next_data[key] = self.data[key][
                self.index+self.obs_horizon : 
                self.index+self.obs_horizon+action_steps*self.robot_downsample]
        self.index += self.obs_horizon

        if self.camera_downsample > 1:
            obs_data['camera0_rgb'] = obs_data['camera0_rgb'][::self.camera_downsample]

        if self.robot_downsample > 1:
            for key in ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width']:
                obs_data[key] = obs_data[key][::self.robot_downsample]
                next_data[key] = next_data[key][::self.robot_downsample]

        return obs_data, next_data



    def exec_actions(self, actions, timestamps):
        pass


@click.command()
@click.option('--config_path', default=None, help='')
@click.option('--policy', '-ip', required=True, help='Name of the policy')
@click.option('--obs_horizon', default=2, type=int, help="")
@click.option('--ckpt_path', '-i', required=True, help='Path to checkpoint')
@click.option('--frequency', '-f', default=5, type=float, help="Control frequency in Hz.")
@click.option('--robot_frequency', default=20, type=float, help="Control frequency in Hz.")
@click.option('--steps_per_inference', default=5, type=int, help="")
@click.option('--dataset_path', '-o')
def main(ckpt_path, config_path, frequency, robot_frequency, obs_horizon, steps_per_inference, policy, dataset_path):
    device = torch.device('cuda')
    dt = 1./frequency
    robot_dt = 1./robot_frequency

    # ========== load policy ==============
    # if not ckpt_path.endswith('.ckpt'):
    #     ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    # payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    # cfg = payload['cfg']
    config_path = ckpt_path if config_path is None else config_path
    if not config_path.endswith('.ckpt'):
        config_path = os.path.join(config_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(config_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

    if policy == 'pi0':
        use_pi0 = True
        config = _config.get_config("pi0_fast_umi")
        checkpoint_dir = ckpt_path
        # Create a trained policy.
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)

        obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
        action_pose_repr = cfg.task.pose_repr.action_pose_repr

    else:
        raise ValueError(f"Unknown policy: {policy}")

    # ========== environment initialize ===========

    # env = ReplayFlexivUmiEnv(cfg.task.dataset.dataset_path, obs_horizon=obs_horizon)
    env = ReplayFlexivUmiEnv(dataset_path, obs_horizon=obs_horizon)

    obs, _ = env.get_obs(steps_per_inference)
    episode_start_pose = [
        np.concatenate([
            obs[f'robot0_eef_pos'],
            obs[f'robot0_eef_rot_axis_angle']
        ], axis=-1)[-1]
    ]

    t_start = time.monotonic()


    # ========== policy control loop ==============
    with KeystrokeCounter() as key_counter:
        while True:
            s = time.time()

            obs, next_obs = env.get_obs(steps_per_inference)

            last_obs_timestamps = time.time()

            print("Obs latency", time.time()-s)

            # for key, value in obs.items():
            #     print(key, value.shape)

            with torch.no_grad():
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=None,
                    episode_start_pose=episode_start_pose)
                next_obs_dict_np = get_real_umi_obs_dict(
                    env_obs=next_obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=None,
                    episode_start_pose=episode_start_pose,
                    use_first_as_relative_pose_base=True)

                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: np.expand_dims(x,0))
                    
                raw_action = policy.infer(obs_dict)
                raw_action = raw_action['actions']
                print(raw_action.shape)

                raw_action[:, 9] = raw_action[:, 9] / 10

                def format_arr(x):
                    return ", ".join(["%.4f"%x for x in x])
                print("Obs", format_arr(obs[f'robot0_eef_pos'][-1]), format_arr(obs[f'robot0_eef_rot_axis_angle'][-1]))
                for idx in range(steps_per_inference):
                    pose10d = raw_action[idx][:9]
                    pose = mat_to_pose(pose10d_to_mat(pose10d))
                    print(f"RawAct-{idx}", format_arr(pose), "||", raw_action[idx][-1])

                for idx in range(steps_per_inference):
                    pose10d = np.concatenate([
                        next_obs_dict_np['robot0_eef_pos'][idx],
                        next_obs_dict_np['robot0_eef_rot_axis_angle'][idx]
                    ])
                    pose = mat_to_pose(pose10d_to_mat(pose10d))
                    
                    print(f"GTAct--{idx}", 
                        format_arr(pose),
                        "||", next_obs_dict_np["robot0_gripper_width"][idx])

                action = get_real_umi_action(raw_action, obs, action_pose_repr)
                for idx in range(steps_per_inference):
                    print(f"ExecAct-{idx}", format_arr(action[idx][:6]), "||", action[idx][-1])


            print('Inference latency:', time.time() - s)
            
            
            this_target_poses = action
            this_target_poses = this_target_poses[:steps_per_inference, :]
            action_timestamps = (1+np.arange(len(this_target_poses), dtype=np.float64)
                ) * robot_dt + time.time()

            env.exec_actions(
                actions=this_target_poses,
                timestamps=action_timestamps
            )
            print(f"Submitted {len(this_target_poses)} steps of actions.")
            

            print('Action latency:', time.time() - s)

            # visualize
            vis_img = obs['camera0_rgb'][-1]
            print(obs['camera0_rgb'].shape)
            text = 'Time: {:.1f}'.format(
                time.monotonic() - t_start
            )
            print(vis_img.shape, vis_img.dtype)
            # cv2.putText(
            #     vis_img,
            #     text,
            #     (10,20),
            #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=0.5,
            #     thickness=1,
            #     color=(255,255,255)
            # )
            cv2.imshow('default', vis_img[...,::-1])

            _ = cv2.pollKey()
            press_events = key_counter.get_press_events()
            for key_stroke in press_events:
                if key_stroke == KeyCode(char='s'):
                    print('Stopped.')
                    break

            input("waiting for key...")
            
            print('Visualize latency:', time.time() - s)

            precise_wait(max(0, s+dt-time.time()))


# %%
if __name__ == '__main__':
    main()