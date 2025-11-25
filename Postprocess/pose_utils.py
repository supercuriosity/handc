
import numpy as np
import scipy.spatial.transform as st



def quat_to_rot(quat_pos):
    shape = quat_pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=quat_pos.dtype)
    if quat_pos.shape[-1] == 7:
        rot = st.Rotation.from_quat(quat_pos[...,3:])
    elif quat_pos.shape[-1] == 8:
        rot = st.Rotation.from_quat(quat_pos[...,3:-1])
    else:
        raise ValueError
    pos = pose[...,:3]
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose


def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot



def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))



def convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep='abs', backward=False):
    if not backward:
        # training transform
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'rel':
            # legacy buggy implementation
            # for compatibility
            raise RuntimeError("rel pose_rep is not supported in training")
            pos = pose_mat[...,:3,3] - base_pose_mat[:3,3]
            rot = pose_mat[...,:3,:3] @ np.linalg.inv(base_pose_mat[:3,:3])
            out = np.copy(pose_mat)
            out[...,:3,:3] = rot
            out[...,:3,3] = pos
            return out
        elif pose_rep == 'relative':
            out = np.linalg.inv(base_pose_mat) @ pose_mat
            return out
        elif pose_rep == 'delta':
            all_pos = np.concatenate([base_pose_mat[None,:3,3], pose_mat[...,:3,3]], axis=0)
            out_pos = np.diff(all_pos, axis=0)
            
            all_rot_mat = np.concatenate([base_pose_mat[None,:3,:3], pose_mat[...,:3,:3]], axis=0)
            prev_rot = np.linalg.inv(all_rot_mat[:-1])
            curr_rot = all_rot_mat[1:]
            out_rot = np.matmul(curr_rot, prev_rot)
            
            out = np.copy(pose_mat)
            out[...,:3,:3] = out_rot
            out[...,:3,3] = out_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

    else:
        # eval transform
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'rel':
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[...,:3,3] + base_pose_mat[:3,3]
            rot = pose_mat[...,:3,:3] @ base_pose_mat[:3,:3]
            out = np.copy(pose_mat)
            out[...,:3,:3] = rot
            out[...,:3,3] = pos
            return out
        elif pose_rep == 'relative':
            out = base_pose_mat @ pose_mat
            return out
        elif pose_rep == 'delta':
            output_pos = np.cumsum(pose_mat[...,:3,3], axis=0) + base_pose_mat[:3,3]
            
            output_rot_mat = np.zeros_like(pose_mat[...,:3,:3])
            curr_rot = base_pose_mat[:3,:3]
            for i in range(len(pose_mat)):
                curr_rot = pose_mat[i,:3,:3] @ curr_rot
                output_rot_mat[i] = curr_rot
            
            out = np.copy(pose_mat)
            out[...,:3,:3] = output_rot_mat
            out[...,:3,3] = output_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")


def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose10d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pos_rot_to_quatpose(pos, rot):
    return np.concatenate([pos, rot.as_quat()], axis=-1)


def pos_rot_to_se3(translation, rotation):
    try:
        len(rotation)
    except TypeError:
        return _single_pos_rot_to_se3(translation, rotation)

    assert len(translation) == len(rotation)
    
    res = []
    for i in range(len(translation)):
        se3 = _single_pos_rot_to_se3(translation[i], rotation[i])
        res.append(se3)
        
    return np.stack(res, axis=0)


def _single_pos_rot_to_se3(translation: np.ndarray, rotation: st.Rotation) -> np.ndarray:
    """
    对数映射 (Logarithm Map): 从 SE(3) 李群到 se(3) 李代数。
    将一个旋转对象和一个平移向量转换为一个 6D 的运动旋量向量。

    参数:
    - rotation (scipy.Rotation): 旋转部分。
    - translation (np.ndarray): 3D 平移向量。

    返回:
    - np.ndarray: 形状为 (6,) 的 se(3) 向量 [v, w]。
    """
    if not isinstance(rotation, st.Rotation):
        raise TypeError("rotation must be a scipy.Rotation object")
    if translation.shape != (3,):
        print(translation, rotation)
        raise ValueError("Translation vector must have shape (3,)")

    # 计算旋转分量 w = log(R)
    w = rotation.as_rotvec()
    theta = np.linalg.norm(w)

    # 处理边缘情况：当旋转非常小时
    if np.isclose(theta, 0.0):
        v = translation
        # w 已经是近似零向量了
        return np.concatenate([v, w])

    # 计算平移分量 v = V_inv @ t
    # V_inv = I - 1/2 * [w] + (1/theta^2 - (1+cos(theta))/(2*theta*sin(theta))) * [w]^2
    w_skew = _skew_symmetric(w)
    w_skew_sq = w_skew @ w_skew
    
    A = -0.5
    # 使用 cot(theta/2) = sin(theta) / (1 - cos(theta)) 的关系可以避免分母为零
    # 使用另一个稳定的公式
    half_theta = theta / 2.0
    B_factor = (1 - half_theta * (np.cos(half_theta) / np.sin(half_theta)))
    B = B_factor / (theta**2)

    V_inv = np.eye(3) + A * w_skew + B * w_skew_sq
    
    v = V_inv @ translation
    
    return np.concatenate([v, w])

def mat_to_quatpose(mat):
    return pos_rot_to_quatpose(*mat_to_pos_rot(mat))

def mat_to_se3(mat):
    return pos_rot_to_se3(*mat_to_pos_rot(mat))


def pos_rot_to_rotvecpose(pos, rot):
    return np.concatenate([pos, rot.as_rotvec()], axis=-1)

def mat_to_rotvecpose(mat):
    return pos_rot_to_rotvecpose(*mat_to_pos_rot(mat))


def mat_to_certain_pose_type(mat, pose_type):
    if pose_type == "10d":
        return mat_to_pose10d(mat)
    elif pose_type == "quat":
        return mat_to_quatpose(mat)
    elif pose_type == "se3":
        return mat_to_se3(mat)
    elif pose_type == "rotvec":
        return mat_to_rotvecpose(mat)
    else:
        raise NotImplementedError(pose_type)