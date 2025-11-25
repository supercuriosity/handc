
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import subprocess
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
sys.path.append('/root/workspace/HandCap/umi')
from umi.common.timecode_util import mp4_get_start_datetime

from time_calib.utils.constants import ARUCO_ID
from time_calib.utils.misc import (
    get_single_path, custom_minimize, 
    plot_trajectories, plot_long_horizon_trajectory
)
from time_calib.utils.data_loading import load_proprio_interp

