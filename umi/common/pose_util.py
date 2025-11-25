from typing import Tuple
import numpy as np
import scipy.spatial.transform as st

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

def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose

def transform_point(tx, point):
    return point @ tx[:3,:3].T + tx[:3,3]

def project_point(k, point):
    x = point @ k.T
    uv = x[...,:2] / x[...,[2]]
    return uv

def apply_delta_pose(pose, delta_pose):
    new_pose = np.zeros_like(pose)

    # simple add for position
    new_pose[:3] = pose[:3] + delta_pose[:3]

    # matrix multiplication for rotation
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose

def normalize(vec, tol=1e-7):
    return vec / np.maximum(np.linalg.norm(vec), tol)

def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

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

def pose10d_to_mat(d10):
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out



def pos_rot_to_quatpose(pos, rot):
    return np.concatenate([pos, rot.as_quat()], axis=-1)

def quatpose_to_pos_rot(quatpose):
    pos, quat = quatpose[..., :3], quatpose[..., 3:]
    return pos, st.Rotation.from_quat(quat)


def mat_to_quatpose(mat):
    return pos_rot_to_quatpose(*mat_to_pos_rot(mat))

def quatpose_to_mat(quatpose):
    return pos_rot_to_mat(*quatpose_to_pos_rot(quatpose))





def pos_rot_to_rotvecpose(pos, rot):
    return np.concatenate([pos, rot.as_rotvec()], axis=-1)

def rotvecpose_to_pos_rot(rotvecpose):
    pos, rotvec = rotvecpose[..., :3], rotvecpose[..., 3:]
    return pos, st.Rotation.from_rotvec(rotvec)



def mat_to_rotvecpose(mat):
    return pos_rot_to_rotvecpose(*mat_to_pos_rot(mat))

def rotvecpose_to_mat(rotvecpose):
    return pos_rot_to_mat(*rotvecpose_to_pos_rot(rotvecpose))
    


    
def _skew_symmetric(w: np.ndarray) -> np.ndarray:
    """
    根据一个3D向量w，生成其对应的3x3反对称矩阵 [w]_x (skew-symmetric matrix)
    """
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])



def se3_to_pos_rot(se3s):
    if isinstance(se3s[0], float):
        return _single_se3_to_pos_rot(se3s)

    trans, rots = [], []
    for se3 in se3s:
        t, r = _single_se3_to_pos_rot(se3)
        trans.append(t)
        rots.append(r.as_matrix())
        
    return np.array(trans), st.Rotation.from_matrix(np.stack(rots, axis=0))


def _single_se3_to_pos_rot(twist: np.ndarray) -> Tuple[np.ndarray, st.Rotation]:
    """
    指数映射 (Exponential Map): 从 se(3) 李代数到 SE(3) 李群。
    将一个 6D 的运动旋量向量转换为一个旋转对象和一个平移向量。

    参数:
    - twist (np.ndarray): 形状为 (6,) 的 se(3) 向量 [v, w]。

    返回:
    - 一个元组 (rotation, translation)
        - translation (np.ndarray): 3D 平移向量。
        - rotation (scipy.Rotation): 旋转部分。
    """
    if twist.shape != (6,):
        print("twist", twist)
        raise ValueError("Twist vector must have shape (6,)")

    v = twist[:3]  # 平移分量
    w = twist[3:]  # 旋转分量 (so(3) 向量)

    theta = np.linalg.norm(w)
    
    # 处理边缘情况：当旋转非常小时 (近似纯平移)
    if np.isclose(theta, 0.0):
        rotation = st.Rotation.identity()
        translation = v
        return translation, rotation

    # 计算旋转部分 R = exp(w)
    rotation = st.Rotation.from_rotvec(w)
    
    # 计算平移部分 t = V @ v
    # V = I + (1 - cos(theta))/theta^2 * [w] + (theta - sin(theta))/theta^3 * [w]^2
    w_skew = _skew_symmetric(w)
    w_skew_sq = w_skew @ w_skew
    
    A = (1 - np.cos(theta)) / (theta ** 2)
    B = (theta - np.sin(theta)) / (theta ** 3)
    
    V = np.eye(3) + A * w_skew + B * w_skew_sq
    
    translation = V @ v
    
    return translation, rotation

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


def mat_to_se3(mat):
    return pos_rot_to_se3(*mat_to_pos_rot(mat))

def se3_to_mat(se3):
    return pos_rot_to_mat(*se3_to_pos_rot(se3))
    
    
    
    
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
    
    
    
def certain_pose_type_to_mat(certain_pose, pose_type):
    if pose_type == "10d":
        return pose10d_to_mat(certain_pose)
    elif pose_type == "quat":
        return quatpose_to_mat(certain_pose)
    elif pose_type == "se3":
        return se3_to_mat(certain_pose)
    elif pose_type == "rotvec":
        return rotvecpose_to_mat(certain_pose)
    else:
        raise NotImplementedError(pose_type)
    
if __name__ == '__main__':
    print("--- 验证转换的正确性 ---")

    # 1. 创建一个随机的 se(3) 运动旋量向量 (twist)
    #    为了测试的泛用性，我们创建一个旋转较大的向量
    random_twist = np.random.randn(6) * 1.5
    print(f"原始 se(3) 向量 (twist):\n{random_twist}\n")

    # 2. 【指数映射】se(3) -> SE(3)
    #    将 se(3) 向量转换为 (Rotation 对象, 平移向量)
    t_vec, R_obj = se3_to_pos_rot(random_twist)

    print("--- 指数映射结果 (exp) ---")
    print(f"转换后的旋转 (欧拉角 deg):\n{R_obj.as_euler('xyz', degrees=True)}")
    print(f"转换后的平移向量:\n{t_vec}\n")

    # 3. 【对数映射】SE(3) -> se(3)
    #    将转换回来的 (Rotation, translation) 再次转换为 se(3) 向量
    reconstructed_twist = pos_rot_to_se3(t_vec, R_obj)
    print("--- 对数映射结果 (log) ---")
    print(f"重建后的 se(3) 向量:\n{reconstructed_twist}\n")

    # 4. 验证
    #    检查重建后的向量是否与原始向量非常接近
    is_close = np.allclose(random_twist, reconstructed_twist)
    print(f"验证结果：重建向量是否与原始向量一致？ -> {is_close}")
    if not is_close:
        print("差值:", random_twist - reconstructed_twist)
        
    print("\n--- 边缘情况测试 (纯平移) ---")
    # 创建一个纯平移的 twist (旋转部分为零)
    pure_translation_twist = np.array([0.5, -1.0, 2.0, 0.0, 0.0, 0.0])
    t_pure, R_pure = se3_to_pos_rot(pure_translation_twist)
    reconstructed_pure = pos_rot_to_se3(t_pure, R_pure)
    
    print(f"原始纯平移 se(3) 向量:\n{pure_translation_twist}")
    print(f"重建后的 se(3) 向量:\n{reconstructed_pure}")
    print(f"验证结果 -> {np.allclose(pure_translation_twist, reconstructed_pure)}")