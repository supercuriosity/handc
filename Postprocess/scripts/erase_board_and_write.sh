#!/bin/bash
source /opt/anaconda3/bin/activate
conda activate handcap

python _01_combine_and_transfer_data_into_lerobot_resume.py \
    --time_file "20251111_143624" \
    --data_root  "/Users/macbookpro/Desktop/workspace/handcap/data/erase_board_and_write/raw_data" \
    --output_root  "/Users/macbookpro/Desktop/workspace/handcap/data/erase_board_and_write/erase_board_and_write_handcap" \
    --angle_calib_file "/Users/macbookpro/Desktop/workspace/handcap/Sensor/angle_calib_params/gripper_calibration_interpolation1111.json" \
    --start_eposide_idx 80