#!/bin/bash
source /opt/anaconda3/bin/activate
conda activate handcap

python _0_combine_and_transfer_data_into_lerobot.py \
    --time_file "20251105_154027" \
    --data_root  "/Users/macbookpro/Desktop/workspace/handcap/data/stack_cups/stack_cups_raw" \
    --output_root  "/Users/macbookpro/Desktop/workspace/handcap/data/stack_cups/stack_cups_handcap" \
    --angle_calib_file "/Users/macbookpro/Desktop/workspace/handcap/Sensor/angle_calib_params/gripper_calibration_interpolation1105.json"