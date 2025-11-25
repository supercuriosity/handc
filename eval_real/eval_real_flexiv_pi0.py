import os
import time
import click
import cv2
import dill
import hydra
import numpy as np
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_umi_obs_dict,
                                                get_real_umi_action)
# from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.pose_util import *

# OmegaConf.register_new_resolver("eval", eval, replace=True)

import sys
from umi.real_world.flexiv import *
sys.path.append('/home/rhos/tactile_umi_grav/flexiv_api/lib_py')
import flexivrdk
# from umi.common.pose_util import *
from umi.real_world.flexiv_simple_env import SimpleFlexivEnv, FooFlexivEnv

import pysnooper

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

@click.command()
@click.option('--policy', '-ip', required=True, help='Name of the policy')
@click.option('--ckpt_path', required=True, help='Path to checkpoint')
# @click.option('--frequency', '-f', default=5, type=float, help="Control frequency in Hz.")
@click.option('--robot_frequency', default=20, type=float, help="Control frequency in Hz.")
@click.option('--obs_horizon', default=1, type=int, help="")
@click.option('--steps_per_inference', default=5, type=int, help="")
# @pysnooper.snoop()
def main(policy, ckpt_path, robot_frequency, steps_per_inference, obs_horizon):
    device = torch.device('cuda')
    robot_dt = 1./robot_frequency
    use_pi0 = False
    # ========== load policy ==============
    obs_pose_rep = 'relative'
    action_pose_repr = 'relative'

    print('obs_pose_rep', obs_pose_rep)
    print('action_pose_repr', action_pose_repr)
    
    if policy == 'pi0':
        use_pi0 = True
        config = _config.get_config("pi0_fast_umi")
        checkpoint_dir = ckpt_path
        # Create a trained policy.
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    
    else:
        raise ValueError(f"Unknown policy: {policy}")
        

    # ========== environment initialize ===========
    # init_qpos = [-1.3136, 0.0514, 1.3010, 2.2039, 0.2174, 1.2170, -0.18627563]  
    # init_qpos = [-0.36204, -0.511710, 0.741515, 2.273541, 0.776503, 2.2775321, -0.75361437] # box
    # init_qpos=[0.23440, -0.39599, -0.036091, 2.2007, 0.7132, 1.7370, -0.55981]  # pick place, old
    # init_qpos=[-0.3926, -0.4882, 0.2110, 1.4517, 0.2123, 0.3067, -1.7398]

    
    init_qpos=eval(os.environ.get("FLEXIV_INIT_POSE", "Set the pose in environ"))

    env = SimpleFlexivEnv(init_qpos, obs_horizon=obs_horizon, use_gripper_width_mapping=False)



    # with KeystrokeCounter() as key_counter:
    while True:
        env.reset()
        policy.reset()


        obs = env.get_obs()
        episode_start_pose = [
            np.concatenate([
                obs[f'robot0_eef_pos'],
                obs[f'robot0_eef_rot_axis_angle']
            ], axis=-1)[-1]
        ]
        
        obs = env.get_obs()

        t_start = time.monotonic()


        # ========== policy control loop ==============
        while True:
            s = time.time()

            obs = env.get_obs()

            last_obs_timestamps = time.time()

            print("Obs latency", time.time()-s)

            # for key, value in obs.items():
            #     print(key, value.shape)

            with torch.no_grad():
                
                if use_pi0:
                    obs_dict = dict_apply(obs, 
                    lambda x: np.expand_dims(x,0))  
                    result = policy.infer(obs_dict)
                    raw_action = result['actions']
                    raw_action[:, 9] = raw_action[:, 9] / 10
                else:
                    obs_dict = dict_apply(obs, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))  
                    result = policy.predict_action(obs_dict)
                    raw_action = result['action_pred'][0].detach().to('cpu').numpy()

                def format_arr(x):
                    return ", ".join(["%.4f"%x for x in x])
                print("Obs", format_arr(obs[f'robot0_eef_pos'][-1]), format_arr(obs[f'robot0_eef_rot_axis_angle'][-1]))
                for idx in range(steps_per_inference):
                    print(f"RawAct-{idx}", format_arr(raw_action[idx][:6]), "||",raw_action[idx][-1])

                action = get_real_umi_action(raw_action, obs, action_pose_repr)
                for idx in range(steps_per_inference):
                    print(f"ExeAct-{idx}", format_arr(action[idx][:6]), "||",action[idx][-1])


            print('Inference latency:', time.time() - s)
            
            
            this_target_poses = action

            # deal with timing
            # the same step actions are always the target for
            # action_timestamps = (np.arange(len(action), dtype=np.float64)
            #     ) * robot_dt + last_obs_timestamps
            # action_exec_latency = 0.01
            # curr_time = time.time()
            # is_new = action_timestamps > (curr_time + action_exec_latency)
            # if np.sum(is_new) == 0:
            #     # exceeded time budget, still do something
            #     this_target_poses = this_target_poses[[-1]]
            #     # schedule on next available step
            #     next_step_idx = int(np.ceil((curr_time - eval_t_start) / robot_dt))
            #     action_timestamp = eval_t_start + (next_step_idx) * robot_dt
            #     print('Over budget', action_timestamp - curr_time)
            #     action_timestamps = np.array([action_timestamp])
            # else:
            #     this_target_poses = this_target_poses[is_new]
            #     action_timestamps = action_timestamps[is_new]
            
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
            text = 'Time: {:.1f}'.format(
                time.monotonic() - t_start
            )
            cv2.putText(
                vis_img,
                text,
                (10,20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1,
                color=(255,255,255)
            )
            cv2.imshow('default', vis_img[...,::-1])

            # _ = cv2.pollKey()
            # press_events = key_counter.get_press_events()
            # for key_stroke in press_events:
            #     if key_stroke == KeyCode(char='s'):
            #         print('Stopped.')
            #         break

            key = cv2.waitKey(1)
            if key == ord('q') or key == "s":
                print('Stopped.')
                break

            # input("waiting for key...")
            
            print('Visualize latency:', time.time() - s)

            # precise_wait(max(0, s+dt-time.time()))

        print("Episode done.")




# %%
if __name__ == '__main__':
    main()
