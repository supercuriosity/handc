#!/bin/bash
source /opt/anaconda3/bin/activate
conda activate handcap

python _01_combine_and_transfer_data_into_lerobot_resume.py \
    --time_file "20251104_133415" \
    --data_root  "/Users/macbookpro/Desktop/workspace/handcap/data/pick_and_place/pick_and_place_raw" \
    --output_root  "/Users/macbookpro/Desktop/workspace/handcap/data/pick_and_place/pick_and_place_handcap" \
    --angle_calib_file "/Users/macbookpro/Desktop/workspace/handcap/Sensor/angle_calib_params/gripper_calibration_interpolation1104.json" \
    --start_eposide_idx 87