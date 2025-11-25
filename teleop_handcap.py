import socket
import time
from argparse import ArgumentParser
import pybullet as pb
import numpy as np
from .config import *
# from .quest_robot_module_clean import QuestRightArmLeapModule
from scipy.spatial.transform import Rotation as R
import sys


# sys.path.append("../")
from umi.common.pose_util import *
from umi.real_world.flexiv import *
from umi.real_world.flexiv_controller import FlexivInterface

sys.path.append("flexiv_api/lib_py")

import flexivrdk



class PlaceholderInterface:
    def __init__(self, *args, **kwargs):
        pass

    def send_flange_pose(self, flange_pose):
        print("Pretend to be sending flange pose", flange_pose)

    def get_flange_pose(self):
        return np.array([0., 0., 0., 0., 0., 0.])

    def gripper_set_binary(self, is_close: bool):
        print("Setting gripper to", "close" if is_close else "open")

    def move_to_home(self, joint_init=None):
        print("Pretend to be moving to home")





tx_arhand = np.identity(4)  # pose of arhand at arbase coordinates
tx_arhand[:3, 3] = [0.03227652, -0.0598839, -0.0737011]
tx_arhand[:3, :3] = R.from_euler(
    "xyz", [16.2832578, -1.30209369, 23.76593208], degrees=True
).as_matrix()
tx_arhand_inv = np.linalg.inv(tx_arhand)

tx_arbase_at_flexivbase = np.identity(4)  # pose of flexiv-base at arbase coordinates
tx_arbase_at_flexivbase[:3, :3] = R.from_euler("z", [-90], degrees=True).as_matrix()

tx_flexivobj_at_arobj = np.identity(4)  # pose of flexiv-base at arbase coordinates
tx_flexivobj_at_arobj[:3, :3] = R.from_euler("z", [90], degrees=True).as_matrix()

tx_flexivcamera_at_flexivobj = np.identity(4)
tx_flexivcamera_at_flexivobj[:3, :3] = R.from_euler("y", [90], degrees=True).as_matrix()


def arcap_to_flexiv_pose(pose):
    def print_matrix(m, prompt):
        def pprint_vec3(v):
            return "[%.2f, %.2f, %.2f]" % (v[0], v[1], v[2])

        p = m[:3, 3]
        r = R.from_matrix(m[:3, :3]).as_euler("xyz", degrees=True)
        print("%010s" % prompt, pprint_vec3(p), pprint_vec3(r))

    pos, quat = pose
    mat = np.identity(4)
    mat[:3, 3] = pos
    mat[:3, :3] = R.from_quat(quat).as_matrix()

    print_matrix(mat, "Received")

    # mat = tx_flexivbase_inv @ (tx_arhand_inv @ mat) @ tx_flexivbase @ tx_rotate_camera # get pose in flexivbase coordinates

    # now best
    tx_arobj_at_arbase =  mat @ tx_arhand_inv
    print_matrix(tx_arobj_at_arbase, "Arcap")
    mat = tx_arbase_at_flexivbase @ tx_arobj_at_arbase @ tx_flexivobj_at_arobj @ tx_flexivcamera_at_flexivobj

    # tx = np.identity(4)
    # tx[:3, :3] = R.from_quat([0.78656833, -0.1232358 ,  0.16076551,  0.58333322]).as_matrix()
    # tx = np.linalg.inv(tx)
    # mat =  mat @ tx

    # print_matrix(tx_arobj_at_arbase, "Arcap")  # Very correct, X: right, Y, front, Z: up
    # print_matrix(mat, "Sent")

    pos = mat[:3, 3]
    rot = R.from_matrix(mat[:3, :3])

    return pos_rot_to_pose(np.array(pos), rot)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frequency", type=int, default=1)
    # parser.add_argument(
    #     "--urdf_path",
    #     type=str,
    #     default="../gloveDemo/leap_assets/leap_hand/robot.urdf",
    # )
    # handedness: "right"
    parser.add_argument("--serial_port", type=str, default="COM3")
    parser.add_argument("--serial_baud", type=int, default=115200)
    args = parser.parse_args()

    c = pb.connect(pb.DIRECT)
    vis_sp = []
    c_code = c_code = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]

    quest = QuestRightArmLeapModule(
        VR_HOST, LOCAL_HOST, POSE_CMD_PORT, IK_RESULT_PORT, vis_sp=None
    )

    robot = FlexivInterface("192.168.2.100", "192.168.2.104")

    start_time = time.time()
    fps_counter = 0
    packet_counter = 0
    print("Initialization completed")

    current_ts = time.time()
    dt = 1.0 / args.frequency

    init_transl_offset = None

    while True:
        now = time.time()
        # TODO: May cause communication issues, need to tune on AR side.
        if now - current_ts < dt:
            continue
        else:
            current_ts = now

        try:
            wrist, head_pose = quest.receive()

            if wrist is None:
                continue


            print(wrist)
            flexiv_pose = arcap_to_flexiv_pose(wrist)
            # print(flexiv_pose)
            if init_transl_offset is None:
                x = input("Enter something")
                real_curr_pose = robot.get_flange_pose()
                init_transl_offset = real_curr_pose[:3] - flexiv_pose[:3]
                print("Init transl", init_transl_offset)
            target_pose = flexiv_pose.copy()
            target_pose[:3] += init_transl_offset

            pos, rot = pose_to_pos_rot(robot.get_flange_pose())
            print(
                "Curr pose:", pos, rot.as_euler("xyz", degrees=True), rot.as_quat()
            )
            pos, rot = pose_to_pos_rot(target_pose)
            print(
                "Sent pose:", pos, rot.as_euler("xyz", degrees=True), rot.as_quat()
            )

            x = input("Enter something")
            if x == "y":
                robot.send_flange_pose(target_pose)
                time.sleep(3)
            else:
                print("Skip")
                


        except socket.error as e:
            print(e)
            pass

        except KeyboardInterrupt:
            quest.close()
            break

        else:
            packet_time = time.time()
            fps_counter += 1
            packet_counter += 1

            if (packet_time - start_time) > 1.0:
                print(f"received {fps_counter} packets in a second", end="\r")
                start_time += 1.0
                fps_counter = 0
