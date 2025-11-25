import os
import xr
import sys
import time
import ctypes
import msvcrt
import select
import datetime
import threading
import numpy as np
import open3d as o3d
import pybullet as pb
from config import *
from ctypes import cast, byref
from argparse import ArgumentParser
import Utils.recordings_utils as utils
from ctypes import cast, byref, POINTER
from scipy.spatial.transform import Rotation as R
from test_sensor.open3d_vis_obj import VIVEOpen3DVisualizer

collecting = False
running = True
episode_index = 1

z_rotation_world_counterclockwise_90 = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
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


class CoordinateSystemVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        
        self.world_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        
        
        self.combined_world_rotation = np.dot(x_rotation_clockwise_90, z_rotation_clockwise_90)
        
        self.world_coordinate_frame.rotate(self.combined_world_rotation, center=(0, 0, 0))

        self.vis.add_geometry(self.world_coordinate_frame)
        
        self.object_coordinate_frames = {}
        
        self.trajectories = {}  # 存储每个物体的轨迹点
        self.trajectory_lines = {}  # 存储轨迹线几何体
        self.max_trajectory_points = 1000  # 最大轨迹点数
        
        # 轨迹颜色配置
        self.trajectory_colors = {
            "handheld_object": [1, 0, 0],      # 红色
            "left_foot": [0, 1, 0],            # 绿色
            "right_foot": [0, 0, 1],           # 蓝色
            "left_shoulder": [1, 1, 0],        # 黄色
            "right_shoulder": [1, 0, 1],       # 紫色
            "left_elbow": [0, 1, 1],           # 青色
            "right_elbow": [1, 0.5, 0],        # 橙色
            "left_knee": [0.5, 1, 0.5],        # 浅绿
            "right_knee": [0.5, 0.5, 1],       # 浅蓝
            "waist": [1, 0.5, 0.5],            # 粉色
            "chest": [0.5, 0, 0.5],            # 深紫
            "camera": [0.7, 0.7, 0.7],         # 灰色
            "keyboard": [0.2, 0.2, 0.2],       # 深灰
        }
    
    def add_object_coordinate_system(self, object_id, position, orientation, size=0.3):
        """添加物体坐标系并更新轨迹
        
        Args:
            object_id: 物体标识符
            position: [x, y, z] 位置
            orientation: [w, x, y, z] 四元数方向
            size: 坐标系大小
        """
        # 更新轨迹
        self._update_trajectory(object_id, position)
        
        # 创建物体坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=[0, 0, 0]
        )
        
        # 转换四元数为旋转矩阵
        rotation_matrix = self.quaternion_to_rotation_matrix(orientation)
        
        # 应用变换
        coord_frame.rotate(rotation_matrix, center=(0, 0, 0))
        coord_frame.translate(position)
        
        # 如果已存在，先移除
        if object_id in self.object_coordinate_frames:
            self.vis.remove_geometry(self.object_coordinate_frames[object_id])
        
        # 添加新的坐标系
        self.object_coordinate_frames[object_id] = coord_frame
        self.vis.add_geometry(coord_frame)
    def _update_trajectory(self, object_id, position):
        """更新物体轨迹"""
        # 初始化轨迹存储
        if object_id not in self.trajectories:
            self.trajectories[object_id] = []
        
        # 添加新位置点
        self.trajectories[object_id].append(position.copy())
        
        # 限制轨迹点数量
        if len(self.trajectories[object_id]) > self.max_trajectory_points:
            self.trajectories[object_id].pop(0)
        
        # 更新轨迹线
        self._update_trajectory_line(object_id)
    
    def _update_trajectory_line(self, object_id):
        """更新轨迹线几何体"""
        if len(self.trajectories[object_id]) < 2:
            return
        
        # 移除旧的轨迹线
        if object_id in self.trajectory_lines:
            self.vis.remove_geometry(self.trajectory_lines[object_id])
        
        # 创建新的轨迹线
        points = self.trajectories[object_id]
        lines = [[i, i + 1] for i in range(len(points) - 1)]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # 设置颜色
        color = self.trajectory_colors.get(object_id, [0.5, 0.5, 0.5])
        colors = [color for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # 添加到可视化器
        self.trajectory_lines[object_id] = line_set
        self.vis.add_geometry(line_set)
    
    def clear_trajectory(self, object_id=None):
        """清除轨迹
        
        Args:
            object_id: 指定物体ID，如果为None则清除所有轨迹
        """
        if object_id is None:
            # 清除所有轨迹
            for obj_id in list(self.trajectory_lines.keys()):
                self.vis.remove_geometry(self.trajectory_lines[obj_id])
                del self.trajectory_lines[obj_id]
                self.trajectories[obj_id] = []
        else:
            # 清除指定物体轨迹
            if object_id in self.trajectory_lines:
                self.vis.remove_geometry(self.trajectory_lines[object_id])
                del self.trajectory_lines[object_id]
            if object_id in self.trajectories:
                self.trajectories[object_id] = []
    
    def set_max_trajectory_points(self, max_points):
        """设置最大轨迹点数"""
        self.max_trajectory_points = max_points
    
    def get_trajectory_info(self):
        """获取轨迹信息"""
        info = {}
        for object_id, trajectory in self.trajectories.items():
            info[object_id] = {
                'point_count': len(trajectory),
                'total_distance': self._calculate_trajectory_distance(trajectory)
            }
        return info

    def _calculate_trajectory_distance(self, trajectory):
        """计算轨迹总距离"""
        if len(trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(trajectory)):
            point1 = np.array(trajectory[i-1])
            point2 = np.array(trajectory[i])
            total_distance += np.linalg.norm(point2 - point1)
        
        return total_distance
    
    def rotation_matrix_to_quaternion(self, R):
        """旋转矩阵转四元数"""
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
    
    def quaternion_to_rotation_matrix(self, q):
        """四元数转旋转矩阵"""
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
    
    def update_display(self):
        """更新显示"""
        # 更新世界坐标系
        self.vis.update_geometry(self.world_coordinate_frame)
        
        # 更新所有物体坐标系
        for coord_frame in self.object_coordinate_frames.values():
            self.vis.update_geometry(coord_frame)
        
        # 更新所有轨迹线
        for trajectory_line in self.trajectory_lines.values():
            self.vis.update_geometry(trajectory_line)
        
        self.vis.poll_events()
        self.vis.update_renderer()


def keyboard_listener():
    global collecting, running, episode_index
    
    while running:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b' ':
                collecting = not collecting
                if collecting:
                    print("Beginning...")
                else:
                    print("Paused...")
            elif key in [b'q', b'Q']:
                collecting = False
                print("collected finished.")
            elif key in [b'e', b'E']:
                collecting = False
                running = False
                print("Exiting.")
            elif key == b'\xe0':
                key2 = msvcrt.getch()
                if key2 == b'K':
                    print(f"delete {episode_index} episode data...")
                    os.system(f"rmdir /s /q data\\pose_{episode_index} 2>nul")
                elif key2 == b'M': 
                    # 修复：确保 episode_index 为整数类型进行运算
                    episode_index = int(episode_index) + 1
                    print(f"switch to {episode_index} episode...")
        time.sleep(0.01)


class ContextObject(object):
    def __init__(
            self,
            instance_create_info: xr.InstanceCreateInfo = xr.InstanceCreateInfo(),
            session_create_info: xr.SessionCreateInfo = xr.SessionCreateInfo(),
            reference_space_create_info: xr.ReferenceSpaceCreateInfo = xr.ReferenceSpaceCreateInfo(),
            view_configuration_type: xr.ViewConfigurationType = xr.ViewConfigurationType.PRIMARY_STEREO,
            environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
            form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY,
    ):
        self._instance_create_info = instance_create_info
        self.instance = None
        self._session_create_info = session_create_info
        self.session = None
        self.session_state = xr.SessionState.IDLE
        self._reference_space_create_info = reference_space_create_info
        self.view_configuration_type = view_configuration_type
        self.environment_blend_mode = environment_blend_mode
        self.form_factor = form_factor
        self.graphics = None
        self.graphics_binding_pointer = None
        self.action_sets = []
        self.render_layers = []
        self.swapchains = []
        self.swapchain_image_ptr_buffers = []
        self.swapchain_image_buffers = []  # Keep alive
        self.exit_render_loop = False
        self.request_restart = False  # TODO: do like hello_xr
        self.session_is_running = False

    def __enter__(self):
        self.instance = xr.create_instance(
            create_info=self._instance_create_info,
        )
        self.system_id = xr.get_system(
            instance=self.instance,
            get_info=xr.SystemGetInfo(
                form_factor=self.form_factor,
            ),
        )

        if self._session_create_info.next is not None:
            self.graphics_binding_pointer = self._session_create_info.next

        self._session_create_info.system_id = self.system_id
        self.session = xr.create_session(
            instance=self.instance,
            create_info=self._session_create_info,
        )
        self.space = xr.create_reference_space(
            session=self.session,
            create_info=self._reference_space_create_info
        )
        self.default_action_set = xr.create_action_set(
            instance=self.instance,
            create_info=xr.ActionSetCreateInfo(
                action_set_name="default_action_set",
                localized_action_set_name="Default Action Set",
                priority=0,
            ),
        )
        self.action_sets.append(self.default_action_set)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.default_action_set is not None:
            xr.destroy_action_set(self.default_action_set)
            self.default_action_set = None
        if self.space is not None:
            xr.destroy_space(self.space)
            self.space = None
        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None
        if self.graphics is not None:
            self.graphics.destroy()
            self.graphics = None
        if self.instance is not None:
            xr.destroy_instance(self.instance)
            self.instance = None

    def frame_loop(self):
        xr.attach_session_action_sets(
            session=self.session,
            attach_info=xr.SessionActionSetsAttachInfo(
                count_action_sets=len(self.action_sets),
                action_sets=(xr.ActionSet * len(self.action_sets))(
                    *self.action_sets
                )
            ),
        )
        while True:
            self.exit_render_loop = False
            self.poll_xr_events()
            if self.exit_render_loop:
                break
            if self.session_is_running:
                if self.session_state in (
                        xr.SessionState.READY,
                        xr.SessionState.SYNCHRONIZED,
                        xr.SessionState.VISIBLE,
                        xr.SessionState.FOCUSED,
                ):
                    frame_state = xr.wait_frame(self.session)
                    xr.begin_frame(self.session)
                    self.render_layers = []

                    yield frame_state

                    xr.end_frame(
                        self.session,
                        frame_end_info=xr.FrameEndInfo(
                            display_time=frame_state.predicted_display_time,
                            environment_blend_mode=self.environment_blend_mode,
                            layers=self.render_layers,
                        )
                    )
            else:
                # Throttle loop since xrWaitFrame won't be called.
                time.sleep(0.250)

    def poll_xr_events(self):
        self.exit_render_loop = False
        self.request_restart = False
        while True:
            try:
                event_buffer = xr.poll_event(self.instance)
                try:
                    event_type = xr.StructureType(event_buffer.type)
                except ValueError as e:
                    print(f"Unknown event type: {event_buffer.type}, skipping...")
                    continue
                # event_type = xr.StructureType(event_buffer.type)
                if event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                    # still handle rest of the events instead of immediately quitting
                    self.exit_render_loop = True
                    self.request_restart = True
                elif event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED \
                        and self.session is not None:
                    event = cast(
                        byref(event_buffer),
                        POINTER(xr.EventDataSessionStateChanged)).contents
                    self.session_state = xr.SessionState(event.state)
                    if self.session_state == xr.SessionState.READY:
                        xr.begin_session(
                            session=self.session,
                            begin_info=xr.SessionBeginInfo(
                                self.view_configuration_type,
                            ),
                        )
                        self.session_is_running = True
                    elif self.session_state == xr.SessionState.STOPPING:
                        self.session_is_running = False
                        xr.end_session(self.session)
                    elif self.session_state == xr.SessionState.EXITING:
                        self.exit_render_loop = True
                        self.request_restart = False
                    elif self.session_state == xr.SessionState.LOSS_PENDING:
                        self.exit_render_loop = True
                        self.request_restart = True
                elif event_type == xr.StructureType.EVENT_DATA_VIVE_TRACKER_CONNECTED_HTCX:
                    vive_tracker_connected = cast(byref(event_buffer), POINTER(xr.EventDataViveTrackerConnectedHTCX)).contents
                    paths = vive_tracker_connected.paths.contents
                    persistent_path_str = xr.path_to_string(self.instance, paths.persistent_path)
                    # print(f"Vive Tracker connected: {persistent_path_str}")
                    if paths.role_path != xr.NULL_PATH:
                        role_path_str = xr.path_to_string(self.instance, paths.role_path)
                        # print(f" New role is: {role_path_str}")
                    else:
                        # print(f" No role path.")
                        pass
                elif event_type == xr.StructureType.EVENT_DATA_INTERACTION_PROFILE_CHANGED:
                    # print("data interaction profile changed")
                    # TODO:
                    pass
            except xr.EventUnavailable:
                break

    def view_loop(self, frame_state):
        if frame_state.should_render:
            layer = xr.CompositionLayerProjection(space=self.space)
            view_state, views = xr.locate_views(
                session=self.session,
                view_locate_info=xr.ViewLocateInfo(
                    view_configuration_type=self.view_configuration_type,
                    display_time=frame_state.predicted_display_time,
                    space=self.space,
                )
            )
            num_views = len(views)
            projection_layer_views = tuple(xr.CompositionLayerProjectionView() for _ in range(num_views))

            vsf = view_state.view_state_flags
            if (vsf & xr.VIEW_STATE_POSITION_VALID_BIT == 0
                    or vsf & xr.VIEW_STATE_ORIENTATION_VALID_BIT == 0):
                return  # There are no valid tracking poses for the views.
            for view_index, view in enumerate(views):
                view_swapchain = self.swapchains[view_index]
                swapchain_image_index = xr.acquire_swapchain_image(
                    swapchain=view_swapchain.handle,
                    acquire_info=xr.SwapchainImageAcquireInfo(),
                )
                xr.wait_swapchain_image(
                    swapchain=view_swapchain.handle,
                    wait_info=xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
                )
                layer_view = projection_layer_views[view_index]
                assert layer_view.type == xr.StructureType.COMPOSITION_LAYER_PROJECTION_VIEW
                layer_view.pose = view.pose
                layer_view.fov = view.fov
                layer_view.sub_image.swapchain = view_swapchain.handle
                layer_view.sub_image.image_rect.offset[:] = [0, 0]
                layer_view.sub_image.image_rect.extent[:] = [
                    view_swapchain.width, view_swapchain.height, ]
                swapchain_image_ptr = self.swapchain_image_ptr_buffers[view_index][swapchain_image_index]
                swapchain_image = cast(swapchain_image_ptr, POINTER(xr.SwapchainImageOpenGLKHR)).contents
                assert layer_view.sub_image.image_array_index == 0  # texture arrays not supported.
                color_texture = swapchain_image.image
                self.graphics.begin_frame(layer_view, color_texture)

                yield view

                self.graphics.end_frame()
                xr.release_swapchain_image(
                    swapchain=view_swapchain.handle,
                    release_info=xr.SwapchainImageReleaseInfo()
                )
            layer.views = projection_layer_views
            self.render_layers.append(byref(layer))

class DataChunker:
    def __init__(self, chunksize=1200):
        self.chunksize = chunksize
        self.records = []
        self.timestamps = []

        self.last_save_dir = None

    def put(self, x, save_dir):  # quest.data_dir
        if save_dir is not None:

            if self.last_save_dir is None or self.last_save_dir == save_dir:
                # normal: push the data and check if need to save
                self.last_save_dir = save_dir
                self.records.append(x)
                self.timestamps.append(time.time())
            
                if len(self.records) > self.chunksize:
                    self.save_and_reset()

            else:
                # suddenly change a dir: save previous chunk first
                self.save_and_reset()

                self.records.append(x)
                self.timestamps.append(time.time())
                self.last_save_dir = save_dir

        else:
            
            if self.last_save_dir is not None:
                # stop saving: save the previous chunk
                self.save_and_reset()

            else:
                # just in case
                self.records = []
                self.timestamps = []

            self.last_save_dir = None
            


    def save_and_reset(self):
        assert self.last_save_dir is not None
        os.makedirs(self.last_save_dir, exist_ok=True)
        path = f"{self.last_save_dir}/chunk_{self.timestamps[0]}_{self.timestamps[-1]}.npz"
        np.savez(
            path,
            pose=self.records,   # xyz-xyzw
            time=self.timestamps,
        )
        
        self.records = []
        self.timestamps = []
        self.last_save_dir = None

        # print(f"Saved chunk to {path}")


def main(frequency):
    
    global episode_index
    wf_receive_ts = time.time()
    coord_visualizer = CoordinateSystemVisualizer()
    first = True
    first_2 = True
    first_3 = True
    episode_index = input("Enter episode index: ")
    print("******** Control Instructions *********************")
    print("******** Press 'space' to start collecting ********")
    print("******** Press 'q' to quit ************************")
    print("******** Press '<-' delete collected data *********")
    print("******** Press '->' to next episode ***************")
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    with ContextObject(
        instance_create_info=xr.InstanceCreateInfo(
            enabled_extension_names=[
                xr.MND_HEADLESS_EXTENSION_NAME,
                xr.extension.HTCX_vive_tracker_interaction.NAME,
            ],
        ),
    ) as context:
        instance = context.instance
        session = context.session

        enumerateViveTrackerPathsHTCX = cast(
            xr.get_instance_proc_addr(instance, "xrEnumerateViveTrackerPathsHTCX"),
            xr.PFN_xrEnumerateViveTrackerPathsHTCX
        )

        role_strings = [
            "handheld_object", "left_foot", "right_foot", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_knee", "right_knee", "waist", "chest",
            "camera", "keyboard",
        ]
        role_path_strings = [f"/user/vive_tracker_htcx/role/{role}" for role in role_strings]
        role_paths = (xr.Path * len(role_path_strings))(*[xr.string_to_path(instance, role_string) for role_string in role_path_strings])
        pose_action = xr.create_action(
            action_set=context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="tracker_pose",
                localized_action_name="Tracker Pose",
                count_subaction_paths=len(role_paths),
                subaction_paths=role_paths,
            ),
        )
        suggested_binding_paths = (xr.ActionSuggestedBinding * len(role_path_strings))(
            *[xr.ActionSuggestedBinding(pose_action, xr.string_to_path(instance, f"{role_path_string}/input/grip/pose")) for role_path_string in role_path_strings]
        )
        xr.suggest_interaction_profile_bindings(instance=instance, suggested_bindings=xr.InteractionProfileSuggestedBinding(
            interaction_profile=xr.string_to_path(instance, "/interaction_profiles/htc/vive_tracker_htcx"),
            count_suggested_bindings=len(suggested_binding_paths), suggested_bindings=suggested_binding_paths,
        ))
        tracker_action_spaces = (xr.Space * len(role_paths))(
            *[xr.create_action_space(session=session, create_info=xr.ActionSpaceCreateInfo(action=pose_action, subaction_path=role_path)) for role_path in role_paths]
        )

        n_paths = ctypes.c_uint32(0)
        result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
        if xr.check_result(result).is_exception():
            raise result
        vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
        result = enumerateViveTrackerPathsHTCX(instance, n_paths, byref(n_paths), vive_tracker_paths)
        if xr.check_result(result).is_exception():
            raise result
        print(xr.Result(result), n_paths.value)

        current_ts = time.time()
        data_chunker = DataChunker(chunksize=2000)
        last_print = time.time()

        formatted_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        for frame_state in context.frame_loop():
            
            if not running:
                break
                
            
            if context.session_state == xr.SessionState.FOCUSED:
                active_action_set = xr.ActiveActionSet(action_set=context.default_action_set, subaction_path=xr.NULL_PATH)
                xr.sync_actions(session=session, sync_info=xr.ActionsSyncInfo(count_active_action_sets=1, active_action_sets=ctypes.pointer(active_action_set)))

                n_paths = ctypes.c_uint32(0)
                result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
                if xr.check_result(result).is_exception():
                    raise result
                vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
                result = enumerateViveTrackerPathsHTCX(instance, n_paths, byref(n_paths), vive_tracker_paths)
                if xr.check_result(result).is_exception():
                    raise result
                found_tracker_count = 0

                
                now = time.time()
                if now - current_ts < 1 / frequency:
                    continue
                else:
                    current_ts = now
                
                for index, space in enumerate(tracker_action_spaces): 
                    
                    # space_location = xr.locate_space(space=space, base_space=context.space, time=frame_state.predicted_display_time)
                    space_location = xr.locate_space(space=space, base_space=context.space, time=time.perf_counter_ns())
                    if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                        if role_strings[index] == 'right_elbow':
                            # if first:
                            #     visualizer.set_pose_first([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 0)
                            #     first = False
                            # else:
                            #     visualizer.set_pose([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 0)

                            # print("Right Elbow:", [space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z])
                            
                            position = [
                                            space_location.pose.position.x, 
                                            space_location.pose.position.y, 
                                            space_location.pose.position.z
                                        ]
                            orientation = [
                                    space_location.pose.orientation.w, 
                                    space_location.pose.orientation.x, 
                                    space_location.pose.orientation.y, 
                                    space_location.pose.orientation.z
                                ]
                            
                            position_transformed = position
                            
                            combined_world_rotation_inverse = coord_visualizer.combined_world_rotation.T
                            position_transformed = np.dot(combined_world_rotation_inverse, position_transformed)
                            
                            object_rotation_matrix = coord_visualizer.quaternion_to_rotation_matrix(orientation)
                            
                            transformed_rotation_matrix = object_rotation_matrix
                            
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
                            orientation_transformed = coord_visualizer.rotation_matrix_to_quaternion(transformed_rotation_matrix)
                
                            raw_xyz = np.array(position_transformed)
                            raw_rotation = np.array(orientation_transformed)
                            
                            coord_visualizer.add_object_coordinate_system(
                                            object_id=role_strings[index],
                                            position=position_transformed,
                                            orientation=orientation_transformed,
                                            size=0.3
                                        )
                            coord_visualizer.update_display()
                
                        # elif role_strings[index] == 'left_elbow':
                        #     if first_2:
                        #         visualizer.set_pose_first([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 1)
                        #         first_2 = False
                        #     else:
                        #         visualizer.set_pose([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 1)
                            
                        #     # print("Left Elbow:", [space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z])
                        # elif role_strings[index] == 'chest':
                        #     if first_3:
                        #         visualizer.set_pose_first([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 2)
                        #         first_3 = False
                        #     else:
                        #         visualizer.set_pose([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 2)
                        #     print("Chest:", [space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z])
                          
                        found_tracker_count += 1
                
                if collecting:   
                    data_dir = f"data/{formatted_time}/pose_{episode_index}"
                    data_chunker.put(np.concatenate([np.array(position), np.array(orientation)]).astype(np.float32), data_dir)
                else:
                    data_chunker.put(None, None)  # 不收集时传递None
                    
                if time.time() - last_print > 1.0:
                    last_print = time.time()
                    rot = R.from_quat(raw_rotation)
                    status = "collecting" if collecting else "paused"
                    status_color = "\033[92m" if collecting else "\033[91m"
                    reset_color = "\033[0m"
                    print(f"{status_color}Episode {episode_index} --- Time: [{last_print}]---[{status}] Data: {raw_xyz} {rot.as_euler('xyz', degrees=True)}{reset_color}")
                else:
                    continue
                    # # if collecting:   
                    # formatted_time = datetime.datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
                    
                    # data_dir = f"data/pose_{episode_index}/{formatted_time}"
                    # data_chunker.put(np.concatenate([raw_xyz, raw_rotation]), data_dir)
                    # if time.time() - last_print > 1.0:
                    #     last_print = time.time()
                    #     rot = R.from_quat(raw_rotation)
                    #     print("Data:", raw_xyz, rot.as_euler("xyz", degrees=True))
                    # else:
                    #     continue
                    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frequency", type=int, default=20)
    args = parser.parse_args()
    main(args.frequency)
