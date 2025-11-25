"""
Main script for UMI SLAM pipeline.
python run_slam_pipeline.py <session_dir>
"""

import sys
import os
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import json
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
# from scipy.optimize import minimize
from umi.common.timecode_util import mp4_get_start_datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def get_tactile_data(npz_file, latency):    
    traj_times = []
    traj_energy = []
    data = np.load(npz_file,allow_pickle=True)  
    images = data['images']
    traj_times += data['time'].tolist()  
    initial_energy = np.mean(images[0])
    for i in range(len(images)):
        energy = np.mean(images[i]) - initial_energy
        traj_energy.append(energy)
    
    traj_energy = np.array(traj_energy)
    kernel_size = 25
    traj_energy = median_filter(traj_energy, size=kernel_size)
    sort_id = np.argsort(traj_times)
    traj_times = np.array(traj_times)[sort_id]
    traj_energy = np.array(traj_energy)[sort_id]
    
    print("trajectory time range", traj_times[0], traj_times[-1])

    return traj_energy, traj_times


def detect_valley(points, kernel_size=25, prominences=0.001):
    points = np.array(points)
    filtered_data = median_filter(points, size=kernel_size)

    inverted_data = -filtered_data
    valley, _ = find_peaks(inverted_data, prominence=prominences)
    valleys = []
    prev = -100
    for v in valley:
        if v - prev >= 10:
            prev = v
            valleys.append(v)
    print(valleys)
    return filtered_data, valleys


def plot_trajectories(aruco_trajectory, tactile_trajectory, save_path):
    
    plt.figure(figsize=(18,4))
    plt.subplot(1,2,1)
    aruco_t = []

    plt.scatter([d[0] for d in aruco_trajectory], [d[1] for d in aruco_trajectory], label='aruco')
    plt.legend()
    aruco_trajectory_t = [d[0] for d in aruco_trajectory]
    aruco_trajectory_e = [d[1] for d in aruco_trajectory]
    filtered_data, valleys = detect_valley(aruco_trajectory_e)
    for valley in valleys:
        aruco_t.append(aruco_trajectory_t[valley])
        plt.scatter(aruco_trajectory_t[valley], filtered_data[valley], color='red', label='Valleys')
    plt.title('Detection of Valleys in Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()

    
    plt.subplot(1,2,2)
    plt.scatter([d[0] for d in tactile_trajectory], [d[1] for d in tactile_trajectory],label='tactile')
    arcap_trajectory_t = [d[0] for d in tactile_trajectory]
    arcap_trajectory_e = [d[1] for d in tactile_trajectory]
    
    filtered_data, valleys = detect_valley(arcap_trajectory_e,prominences=0.05)
    arcap_t = []
    for valley in valleys:
        arcap_t.append(arcap_trajectory_t[valley])
        plt.scatter(arcap_trajectory_t[valley], filtered_data[valley], color='red', label='Valleys')
        
    plt.title('Detection of Valleys in Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.legend()
    plt.savefig(str(save_path))
    plt.close()

    return arcap_t, aruco_t

    
# %%
@click.command()
@click.argument('session_dir')
@click.option('-c', '--calibration_dir', required=True, help='')
def main(session_dir, calibration_dir):
    calibration_dir = pathlib.Path(calibration_dir)
    # script_dir = pathlib.Path(__file__).parent.joinpath('scripts_slam_pipeline') obsolete

    session_dir = pathlib.Path(__file__).parent.joinpath(session_dir).absolute()
    tactile_data_dir = session_dir.joinpath('data')
    latency_calib_dir = session_dir.joinpath('latency_calibration')

    assert tactile_data_dir.is_dir()
    assert latency_calib_dir.is_dir()

    tactile_path = list(tactile_data_dir.glob('**/*.npz'))
    calibration_mp4_paths = list(latency_calib_dir.glob('**/*.MP4')) + list(latency_calib_dir.glob('**/*.mp4'))
    if len(calibration_mp4_paths) == 0:
        raise FileNotFoundError(f"No mp4 files found in {latency_calib_dir}")
    elif len(calibration_mp4_paths) > 1:
        print(f"Multiple mp4 files found in {latency_calib_dir}. Using the first one.")  # TODO use multiple in the future
    
    calibration_mp4 = calibration_mp4_paths[0]

    print('load tactile')
    traj_energy, traj_time= get_tactile_data(str(tactile_path[0]), latency=0.0)
    print("detect_aruco")
    script_path = pathlib.Path(__file__).parent.parent.joinpath('scripts', 'detect_aruco_newversion.py')
    assert script_path.is_file()
    
    camera_intrinsics = calibration_dir.joinpath('gopro_intrinsics_2_7k.json')
    aruco_config = calibration_dir.joinpath('aruco_config.yaml')
    assert camera_intrinsics.is_file()
    assert aruco_config.is_file()
    aruco_out_dir = latency_calib_dir.joinpath('tag_detection.pkl')
    if not aruco_out_dir.is_file(): 
        cmd = [
            'python', script_path,
            '--input', str(calibration_mp4),
            '--output', str(aruco_out_dir),
            '--intrinsics_json', camera_intrinsics,
            '--aruco_yaml', str(aruco_config),
            '--num_workers', '1'
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0
    else:
        print(f"tag_detection.pkl already exists, skipping {calibration_mp4}")


    print("align visual and trajectory")

    # get aruco trajectory
    video_start_time = mp4_get_start_datetime(str(calibration_mp4)).timestamp()
    print(video_start_time)
    aruco_pickle_path = latency_calib_dir.joinpath('tag_detection.pkl')
    with open(str(aruco_pickle_path), "rb") as fp:
        aruco_pkl = pickle.load(fp)

        aruco_trajectory = []
        for frame in aruco_pkl:

            if 6 in frame['tag_dict'] and 7 in frame['tag_dict']:
                x1 = frame['tag_dict'][6]['tvec'][0]
                x2 = frame['tag_dict'][7]['tvec'][0]
                aruco_trajectory.append((frame['time']+video_start_time, np.abs(x1-x2)))

    # get tactile trajectory
    tactile_trajectory = []

    for i in range(len(traj_energy)):
        tactile_trajectory.append((traj_time[i], traj_energy[i]))


    arcap_t, aruco_t = plot_trajectories(aruco_trajectory, tactile_trajectory, 
                latency_calib_dir.joinpath(f'latency_trajectory_offset.png'))

    assert len(arcap_t) == len(aruco_t)
    
    latency_of_tactile = []
    for i in range(len(arcap_t)):
        latency_of_tactile.append(arcap_t[i] - aruco_t[i])
    
    latency_of_tactile = np.array(latency_of_tactile)
    latency_of_tactile_result = {}
    if np.std(latency_of_tactile) < 0.1:
        # everything is good               
        latency_of_tactile_result = {
            "mean": np.mean(latency_of_tactile),
            "std": np.std(latency_of_tactile),
        }

    with open(str(latency_calib_dir.joinpath('latency_of_arcap.json')), "w") as fp:
        json.dump(latency_of_tactile_result, fp)
    
## %%
if __name__ == "__main__":
    main()

# %%
