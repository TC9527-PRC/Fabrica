#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry.py - 机器人硬件几何模型处理模块

该模块负责处理机器人硬件的几何模型，包括机械臂和夹持器的模型加载、转换和操作。
主要功能包括：
1. 网格模型的加载、合并和变换
2. 部件模型的处理
3. 夹持器模型的加载、配置和变换
4. 机械臂模型的加载、配置和变换

支持的硬件类型：
- 机械臂：xarm7、panda、ur5e
- 夹持器：panda、robotiq-85、robotiq-140
"""

import os
import sys

# 添加项目根目录到Python路径
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from assets.load import load_assembly
from assets.transform import get_scale_matrix, get_translate_matrix, get_revolute_matrix, get_transform_matrix_quat, get_pos_quat_from_pose


'''Utils - 工具函数模块'''


def apply_transform(matrix_base, matrix_transform):
    """
    对基础矩阵应用变换矩阵（原地操作）
    
    参数：
        matrix_base (numpy.ndarray): 基础变换矩阵 (4x4)
        matrix_transform (numpy.ndarray): 要应用的变换矩阵 (4x4)
    
    返回：
        numpy.ndarray: 变换后的矩阵
    """
    return np.matmul(matrix_transform, matrix_base, out=matrix_base)


def save_meshes(meshes, folder, include_color=False):
    """
    保存网格模型到指定文件夹
    
    参数：
        meshes (dict): 网格模型字典，键为模型名称，值为trimesh对象
        folder (str): 保存文件夹路径
        include_color (bool): 是否包含颜色信息
    """
    for name, mesh in meshes.items():
        mesh.export(os.path.join(folder, f'{name}.obj'), header=None, include_color=include_color)


def add_buffer_to_mesh(mesh, buffer):
    """
    为网格模型添加缓冲区（沿法线方向扩展）
    
    参数：
        mesh (trimesh.Trimesh): 原始网格模型
        buffer (float): 缓冲区大小
    
    返回：
        trimesh.Trimesh: 添加缓冲区后的网格模型
    """
    vertex_normals = mesh.vertex_normals
    new_vertices = mesh.vertices + buffer * vertex_normals
    return trimesh.Trimesh(new_vertices, mesh.faces, vertex_normals=vertex_normals)


def get_buffered_meshes(meshes, buffer):
    """
    为网格模型字典或列表添加缓冲区
    
    参数：
        meshes (trimesh.Trimesh, dict, list): 单个网格、网格字典或网格列表
        buffer (float): 缓冲区大小
    
    返回：
        与输入类型相同的带缓冲区的网格对象
    """
    if isinstance(meshes, trimesh.Trimesh):
        return add_buffer_to_mesh(meshes, buffer=buffer)
    elif isinstance(meshes, dict):
        meshes = {k: v.copy() for k, v in meshes.items()}
        for name, mesh in meshes.items():
            meshes[name] = add_buffer_to_mesh(mesh, buffer=buffer)
    elif isinstance(meshes, list):
        meshes = [mesh.copy() for mesh in meshes]
        for i, mesh in enumerate(meshes):
            meshes[i] = add_buffer_to_mesh(mesh, buffer=buffer)
    else:
        raise NotImplementedError
    return meshes


def get_buffered_gripper_meshes(gripper_type, gripper_meshes):
    """
    为夹持器网格添加缓冲区，手指和手掌/指节使用不同的缓冲区大小
    
    参数：
        gripper_type (str): 夹持器类型
        gripper_meshes (dict): 夹持器网格字典
    
    返回：
        dict: 带缓冲区的夹持器网格字典
    """
    finger_buffer = 0.25  # 手指缓冲区较小
    hand_knuckle_buffer = 1.0  # 手掌和指节缓冲区较大
    gripper_finger_meshes = {}
    gripper_hand_knuckle_meshes = {}
    
    # 分离手指和手掌/指节网格
    for name, mesh in gripper_meshes.items():
        if name in get_gripper_finger_names(gripper_type):
            gripper_finger_meshes[name] = mesh.copy()
        else:
            gripper_hand_knuckle_meshes[name] = mesh.copy()
    
    # 为不同部分添加不同大小的缓冲区
    gripper_finger_meshes = get_buffered_meshes(gripper_finger_meshes, buffer=finger_buffer)
    gripper_hand_knuckle_meshes = get_buffered_meshes(gripper_hand_knuckle_meshes, buffer=hand_knuckle_buffer)
    
    # 合并结果
    return {**gripper_finger_meshes, **gripper_hand_knuckle_meshes}


def get_buffered_arm_meshes(arm_meshes):
    """
    为机械臂网格添加缓冲区
    
    参数：
        arm_meshes (dict): 机械臂网格字典
    
    返回：
        dict: 带缓冲区的机械臂网格字典
    """
    return get_buffered_meshes(arm_meshes, buffer=1.0)


def get_combined_mesh(scene_or_mesh):
    """
    合并网格场景或返回单个网格
    
    参数：
        scene_or_mesh (trimesh.Scene or trimesh.Trimesh): 网格场景或单个网格
    
    返回：
        trimesh.Trimesh: 合并后的网格
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [scene_or_mesh.geometry[name] for name in scene_or_mesh.geometry]
        combined_mesh = trimesh.util.concatenate(meshes)
        return combined_mesh
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    else:
        raise ValueError("Input is neither a trimesh.Trimesh nor a trimesh.Scene.")
    

def get_combined_meshes(meshes):
    """
    合并网格字典或列表中的网格
    
    参数：
        meshes (dict or list): 网格字典或列表
    
    返回：
        与输入类型相同的合并后的网格对象
    """
    if isinstance(meshes, dict):
        return {key: get_combined_mesh(mesh) for key, mesh in meshes.items()}
    elif isinstance(meshes, list):
        return [get_combined_mesh(mesh) for mesh in meshes]
    else:
        raise ValueError("Input is neither a dict nor a list.")


'''Parts - 部件处理模块'''


def load_part_meshes(assembly_dir, transform='none', rename=True, combined=True, **kwargs):
    """
    加载装配部件的网格模型
    
    参数：
        assembly_dir (str): 装配目录路径
        transform (str): 变换类型，默认为'none'
        rename (bool): 是否重命名部件为'part0', 'part1'等
        combined (bool): 是否合并网格
        **kwargs: 传递给load_assembly的额外参数
    
    返回：
        dict: 部件网格字典
    """
    assembly = load_assembly(assembly_dir, transform=transform, **kwargs)
    if rename:
        part_meshes = {f'part{part_id}': assembly[part_id]['mesh'] for part_id in assembly}
    else:
        part_meshes = {part_id: assembly[part_id]['mesh'] for part_id in assembly}
    if combined:
        part_meshes = get_combined_meshes(part_meshes)
    return part_meshes


def get_part_mesh_transform(pos, quat, pose=None):
    """
    获取部件网格的变换矩阵
    
    参数：
        pos (list): 位置坐标 [x, y, z]
        quat (list): 四元数 [x, y, z, w]
        pose (dict, optional): 姿态字典，用于覆盖pos和quat
    
    返回：
        numpy.ndarray: 变换矩阵 (4x4)
    """
    if pose is not None:
        pos, quat = get_pos_quat_from_pose(pos, quat, pose)
    return get_transform_matrix_quat(pos, quat)


def transform_part_mesh(mesh, pos, quat, pose=None):
    """
    变换部件网格到指定位置和姿态
    
    参数：
        mesh (trimesh.Trimesh): 部件网格
        pos (list): 位置坐标 [x, y, z]
        quat (list): 四元数 [x, y, z, w]
        pose (dict, optional): 姿态字典，用于覆盖pos和quat
    
    返回：
        trimesh.Trimesh: 变换后的部件网格
    """
    mesh = mesh.copy()
    mesh.apply_transform(get_part_mesh_transform(pos, quat, pose))
    return mesh


def get_part_meshes_transforms(meshes, pos_dict, quat_dict, pose=None):
    """
    获取多个部件网格的变换矩阵
    
    参数：
        meshes (dict): 部件网格字典
        pos_dict (dict): 位置字典，键为部件名称，值为位置坐标
        quat_dict (dict): 四元数字典，键为部件名称，值为四元数
        pose (dict, optional): 姿态字典，用于覆盖pos和quat
    
    返回：
        dict: 变换矩阵字典
    """
    return {k: get_part_mesh_transform(pos_dict[k], quat_dict[k], pose) for k in meshes.keys()}


def transform_part_meshes(meshes, pos_dict, quat_dict, pose=None):
    """
    变换多个部件网格到指定位置和姿态
    
    参数：
        meshes (dict): 部件网格字典
        pos_dict (dict): 位置字典，键为部件名称，值为位置坐标
        quat_dict (dict): 四元数字典，键为部件名称，值为四元数
        pose (dict, optional): 姿态字典，用于覆盖pos和quat
    
    返回：
        dict: 变换后的部件网格字典
    """
    transforms = get_part_meshes_transforms(meshes, pos_dict, quat_dict, pose)
    meshes = {k: v.copy() for k, v in meshes.items()}
    for name, mesh in meshes.items():
        mesh.apply_transform(transforms[name])
    return meshes


'''Grippers - 夹持器处理模块'''


def get_panda_grasp_base_offset():
    """
    获取Panda夹持器的抓取基准偏移量
    
    返回：
        float: 抓取基准偏移量
    """
    return 10.34 + 0.9  # NOTE: extra 0.9 for finger tip


def get_robotiq_85_grasp_base_offset(open_ratio):
    """
    获取Robotiq 85夹持器的抓取基准偏移量
    
    参数：
        open_ratio (float): 夹持器张开比例 (0-1)
    
    返回：
        float: 抓取基准偏移量
    """
    return 3.5 + np.sin(0.9208 + (1 - open_ratio) * 0.8757) * 5.715 + 6.93075 + 0.9  # NOTE: extra 0.9 for finger tip


def get_robotiq_140_grasp_base_offset(open_ratio):
    """
    获取Robotiq 140夹持器的抓取基准偏移量
    
    参数：
        open_ratio (float): 夹持器张开比例 (0-1)
    
    返回：
        float: 抓取基准偏移量
    """
    return 1.5 + 3.8 + np.sin(0.8680 + (1 - open_ratio) * 0.8757) * 10 + 5.4905 + 1.0  # NOTE: extra 1.0 for finger tip


def get_gripper_grasp_base_offset(gripper_type, open_ratio, delta=0.0):
    """
    获取指定类型夹持器的抓取基准偏移量
    
    参数：
        gripper_type (str): 夹持器类型
        open_ratio (float): 夹持器张开比例 (0-1)
        delta (float): 额外偏移量
    
    返回：
        float: 抓取基准偏移量
    """
    if gripper_type == 'panda':
        return get_panda_grasp_base_offset() + delta
    elif gripper_type == 'robotiq-85':
        return get_robotiq_85_grasp_base_offset(open_ratio) + delta
    elif gripper_type == 'robotiq-140':
        return get_robotiq_140_grasp_base_offset(open_ratio) + delta
    else:
        raise NotImplementedError


def get_panda_basis_directions():
    """
    获取Panda夹持器的基准方向
    
    返回：
        tuple: (抓取方向, 手指方向)
    """
    return [0, 0, -1], [0, 1, 0]


def get_robotiq_85_basis_directions():
    """
    获取Robotiq 85夹持器的基准方向
    
    返回：
        tuple: (抓取方向, 手指方向)
    """
    return [0, 0, -1], [-1, 0, 0]


def get_robotiq_140_basis_directions():
    """
    获取Robotiq 140夹持器的基准方向
    
    返回：
        tuple: (抓取方向, 手指方向)
    """
    return [0, 0, -1], [0, 1, 0]


def get_gripper_basis_directions(gripper_type):
    """
    获取指定类型夹持器的基准方向
    
    参数：
        gripper_type (str): 夹持器类型
    
    返回：
        tuple: (抓取方向, 手指方向)
    """
    if gripper_type == 'panda':
        return get_panda_basis_directions()
    elif gripper_type == 'robotiq-85':
        return get_robotiq_85_basis_directions()
    elif gripper_type == 'robotiq-140':
        return get_robotiq_140_basis_directions()
    else:
        raise NotImplementedError


def get_panda_open_ratio(antipodal_points):
    """
    根据对顶点计算Panda夹持器的张开比例
    
    参数：
        antipodal_points (list): 两个对顶点的坐标 [[x1, y1, z1], [x2, y2, z2]]
    
    返回：
        float or None: 张开比例 (0-1)，如果超出范围返回None
    """
    antipodal_points = np.array(antipodal_points, dtype=float)
    antipodal_width = np.linalg.norm(antipodal_points[1] - antipodal_points[0])
    open_ratio = antipodal_width / 8
    if open_ratio > 1:
        return None
    else:
        return open_ratio


def get_robotiq_85_open_ratio(antipodal_points):
    """
    根据对顶点计算Robotiq 85夹持器的张开比例
    
    参数：
        antipodal_points (list): 两个对顶点的坐标 [[x1, y1, z1], [x2, y2, z2]]
    
    返回：
        float or None: 张开比例 (0-1)，如果超出范围返回None
    """
    antipodal_points = np.array(antipodal_points, dtype=float)
    antipodal_width = np.linalg.norm(antipodal_points[1] - antipodal_points[0])
    if antipodal_width > 3.92853109 * 2:
        return None
    else:
        return 1.0 - (np.arccos((antipodal_width / 2 + 0.8 - 1.27) / 5.715) - 0.9208) / 0.8757


def get_robotiq_140_open_ratio(antipodal_points):
    """
    根据对顶点计算Robotiq 140夹持器的张开比例
    
    参数：
        antipodal_points (list): 两个对顶点的坐标 [[x1, y1, z1], [x2, y2, z2]]
    
    返回：
        float or None: 张开比例 (0-1)，如果超出范围返回None
    """
    antipodal_points = np.array(antipodal_points, dtype=float)
    antipodal_width = np.linalg.norm(antipodal_points[1] - antipodal_points[0])
    if antipodal_width > 7.22376574 * 2:
        return None
    else:
        return 1.0 - (np.arccos((antipodal_width / 2 + 0.325 + 2.3 - 1.7901 - 1.27) / 10) - 0.8680) / 0.8757


def get_gripper_open_ratio(gripper_type, antipodal_points):
    """
    根据对顶点计算指定类型夹持器的张开比例
    
    参数：
        gripper_type (str): 夹持器类型
        antipodal_points (list): 两个对顶点的坐标 [[x1, y1, z1], [x2, y2, z2]]
    
    返回：
        float or None: 张开比例 (0-1)，如果超出范围返回None
    """
    if gripper_type == 'panda':
        return get_panda_open_ratio(antipodal_points)
    elif gripper_type == 'robotiq-85':
        return get_robotiq_85_open_ratio(antipodal_points)
    elif gripper_type == 'robotiq-140':
        return get_robotiq_140_open_ratio(antipodal_points)
    else:
        raise NotImplementedError
    

def get_panda_finger_states(open_ratio):
    """
    获取Panda夹持器的手指状态
    
    参数：
        open_ratio (float): 张开比例 (0-1)
    
    返回：
        dict: 手指状态字典，键为手指名称，值为关节角度
    """
    finger_open_extent = 4 * open_ratio
    return {
        'panda_leftfinger': [finger_open_extent],
        'panda_rightfinger': [finger_open_extent],
    }


def get_robotiq_85_finger_states(open_ratio):
    """
    获取Robotiq 85夹持器的手指状态
    
    参数：
        open_ratio (float): 张开比例 (0-1)
    
    返回：
        dict: 手指状态字典，键为手指部件名称，值为关节角度
    """
    finger_states = {}
    for side_i in ['left', 'right']:
        for side_j in ['outer', 'inner']:
            for link in ['knuckle', 'finger']:
                name = f'{side_i}_{side_j}_{link}'
                if name in ['left_outer_finger', 'right_outer_finger']: continue
                sign = -1 if name in ['left_inner_finger', 'right_inner_knuckle'] else 1
                finger_states[f'robotiq_{name}'] = [sign * (1 - open_ratio) * 0.8757]
    return finger_states


def get_robotiq_140_finger_states(open_ratio):
    """
    获取Robotiq 140夹持器的手指状态
    
    参数：
        open_ratio (float): 张开比例 (0-1)
    
    返回：
        dict: 手指状态字典，键为手指部件名称，值为关节角度
    """
    finger_states = {}
    for side in ['left', 'right']:
        for link in ['outer_knuckle', 'inner_finger', 'inner_knuckle']:
            name = f'{side}_{link}'
            sign = 1 if link == 'inner_finger' or name == 'left_outer_knuckle' else -1
            finger_states[f'robotiq_{name}'] = [sign * (1 - open_ratio) * 0.8757]
    return finger_states


def get_gripper_finger_states(gripper_type, open_ratio, suffix=None):
    """
    获取指定类型夹持器的手指状态
    
    参数：
        gripper_type (str): 夹持器类型
        open_ratio (float): 张开比例 (0-1)
        suffix (str, optional): 名称后缀
    
    返回：
        dict: 手指状态字典
    """
    if gripper_type == 'panda':
        finger_states = get_panda_finger_states(open_ratio)
    elif gripper_type == 'robotiq-85':
        finger_states = get_robotiq_85_finger_states(open_ratio)
    elif gripper_type == 'robotiq-140':
        finger_states = get_robotiq_140_finger_states(open_ratio)
    else:
        raise NotImplementedError
    if suffix is not None:
        finger_states = {f'{key}_{suffix}': value for key, value in finger_states.items()}
    return finger_states


def get_gripper_base_name(gripper_type, suffix=None):
    """
    获取夹持器基座名称
    
    参数：
        gripper_type (str): 夹持器类型
        suffix (str, optional): 名称后缀
    
    返回：
        str: 夹持器基座名称
    """
    if gripper_type == 'panda':
        name = 'panda_hand'
    elif gripper_type in ['robotiq-85', 'robotiq-140']:
        name = 'robotiq_base'
    else:
        raise NotImplementedError
    if suffix is not None:
        name = f'{name}_{suffix}'
    return name


def get_gripper_hand_names(gripper_type, suffix=None):
    """
    获取夹持器手部部件名称列表
    
    参数：
        gripper_type (str): 夹持器类型
        suffix (str, optional): 名称后缀
    
    返回：
        list: 手部部件名称列表
    """
    if gripper_type == 'panda':
        names = ['panda_hand']
    elif gripper_type == 'robotiq-85':
        names = ['robotiq_base']
        for side_i in ['left', 'right']:
            for side_j in ['outer', 'inner']:
                for link in ['knuckle', 'finger']:
                    name = f'{side_i}_{side_j}_{link}'
                    if name not in ['left_inner_finger', 'right_inner_finger']:
                        names.append(f'robotiq_{name}')
    elif gripper_type == 'robotiq-140':
        names = ['robotiq_base']
        for side_i in ['left', 'right']:
            for side_j in ['outer', 'inner']:
                for link in ['knuckle', 'finger']:
                    name = f'{side_i}_{side_j}_{link}'
                    names.append(f'robotiq_{name}')
    else:
        raise NotImplementedError
    if suffix is not None:
        names = [f'{name}_{suffix}' for name in names]
    return names


def get_gripper_knuckle_names(gripper_type, suffix=None):
    """
    获取夹持器指节名称列表（用于避免在指节内部抓取部件）
    
    参数：
        gripper_type (str): 夹持器类型
        suffix (str, optional): 名称后缀
    
    返回：
        list: 指节名称列表
    """
    if gripper_type == 'panda':
        names = []
    elif gripper_type == 'robotiq-85' or gripper_type == 'robotiq-140':
        names = []
        for side_i in ['left', 'right']:
            for side_j in ['outer', 'inner']:
                name = f'{side_i}_{side_j}_knuckle'
                names.append(f'robotiq_{name}')
    else:
        raise NotImplementedError
    if suffix is not None:
        names = [f'{name}_{suffix}' for name in names]
    return names


def get_gripper_finger_names(gripper_type, suffix=None):
    """
    获取夹持器手指名称列表
    
    参数：
        gripper_type (str): 夹持器类型
        suffix (str, optional): 名称后缀
    
    返回：
        list: 手指名称列表
    """
    if gripper_type == 'panda':
        names = ['panda_leftfinger', 'panda_rightfinger']
    elif gripper_type == 'robotiq-85':
        names = ['robotiq_left_inner_finger', 'robotiq_right_inner_finger']
    elif gripper_type == 'robotiq-140':
        names = ['robotiq_left_pad', 'robotiq_right_pad']
    else:
        raise NotImplementedError
    if suffix is not None:
        names = [f'{name}_{suffix}' for name in names]
    return names


def load_panda_meshes(asset_folder, visual=False):
    """
    加载Panda夹持器的网格模型
    
    参数：
        asset_folder (str): 资产文件夹路径
        visual (bool): 是否加载视觉模型（否则加载碰撞模型）
    
    返回：
        dict: 夹持器网格字典
    """
    meshes = {}
    dir_name = 'visual' if visual else 'collision'
    meshes['panda_hand'] = trimesh.load(os.path.join(asset_folder, 'panda', dir_name, 'hand.obj'))
    meshes['panda_leftfinger'] = trimesh.load(os.path.join(asset_folder, 'panda', dir_name, 'finger.obj'))
    meshes['panda_rightfinger'] = trimesh.load(os.path.join(asset_folder, 'panda', dir_name, 'finger.obj'))
    return meshes


def load_robotiq_85_meshes(asset_folder, visual=False):
    """
    加载Robotiq 85夹持器的网格模型
    
    参数：
        asset_folder (str): 资产文件夹路径
        visual (bool): 是否加载视觉模型（否则加载碰撞模型）
    
    返回：
        dict: 夹持器网格字典
    """
    meshes = {}
    dir_name = 'visual' if visual else 'collision'
    postfix = 'fine' if visual else 'coarse'
    meshes['robotiq_base'] = trimesh.load(os.path.join(asset_folder, 'robotiq_85', dir_name, f'robotiq_base_{postfix}.obj'))
    for side_i in ['left', 'right']:
        for side_j in ['outer', 'inner']:
            for link in ['knuckle', 'finger']:
                meshes[f'robotiq_{side_i}_{side_j}_{link}'] = trimesh.load(os.path.join(asset_folder, 'robotiq_85', dir_name, f'{side_j}_{link}_{postfix}.obj'))
    return meshes


def load_robotiq_140_meshes(asset_folder, visual=False):
    """
    加载Robotiq 140夹持器的网格模型
    
    参数：
        asset_folder (str): 资产文件夹路径
        visual (bool): 是否加载视觉模型（否则加载碰撞模型）
    
    返回：
        dict: 夹持器网格字典
    """
    meshes = {}
    dir_name = 'visual' if visual else 'collision'
    postfix = 'fine' if visual else 'coarse'
    meshes['robotiq_base'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'robotiq_base_{postfix}.obj'))
    for side in ['left', 'right']:
        for link in ['outer_knuckle', 'outer_finger', 'inner_finger']:
            meshes[f'robotiq_{side}_{link}'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'{link}_{postfix}.obj'))
        meshes[f'robotiq_{side}_pad'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'pad_{postfix}.obj'))
        meshes[f'robotiq_{side}_inner_knuckle'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'inner_knuckle_{postfix}.obj'))
    return meshes


def get_ft_sensor_spec():
    """
    获取力扭矩传感器的规格参数
    
    返回：
        dict: 传感器规格，包含半径和高度
    """
    return {'radius': 3.75, 'height': 3.0}  # NOTE: hardcoded values for robotiq sensor


def load_ft_sensor_mesh():
    """
    加载力扭矩传感器的网格模型
    
    返回：
        trimesh.Trimesh: 传感器网格模型
    """
    spec = get_ft_sensor_spec()
    return trimesh.creation.cylinder(radius=spec['radius'], height=spec['height'])


def load_gripper_meshes(gripper_type, asset_folder, has_ft_sensor=False, visual=False, combined=True):
    """
    加载指定类型夹持器的网格模型
    
    参数：
        gripper_type (str): 夹持器类型
        asset_folder (str): 资产文件夹路径
        has_ft_sensor (bool): 是否包含力扭矩传感器
        visual (bool): 是否加载视觉模型（否则加载碰撞模型）
        combined (bool): 是否合并网格
    
    返回：
        dict: 夹持器网格字典
    """
    if gripper_type == 'panda':
        gripper_meshes = load_panda_meshes(asset_folder, visual=visual)
    elif gripper_type == 'robotiq-85':
        gripper_meshes =  load_robotiq_85_meshes(asset_folder, visual=visual)
    elif gripper_type == 'robotiq-140':
        gripper_meshes =  load_robotiq_140_meshes(asset_folder, visual=visual)
    else:
        raise NotImplementedError
    if has_ft_sensor:
        gripper_meshes['ft_sensor'] = load_ft_sensor_mesh()
    if combined:
        gripper_meshes = get_combined_meshes(gripper_meshes)
    return gripper_meshes


def get_panda_meshes_transforms(meshes, open_ratio):
    """
    获取Panda夹持器各部件的变换矩阵
    
    参数：
        meshes (dict): 夹持器网格字典
        open_ratio (float): 张开比例 (0-1)
    
    返回：
        dict: 变换矩阵字典
    """
    transforms = {k: np.eye(4) for k in meshes.keys()}

    transforms['panda_leftfinger'] = get_translate_matrix([0, 4 * open_ratio, 5.84])
    transforms['panda_rightfinger'] = get_translate_matrix([0, -4 * open_ratio, 5.84]) @ get_scale_matrix([1, -1, 1])
    
    return transforms


def get_robotiq_85_meshes_transforms(meshes, open_ratio):
    """
    获取Robotiq 85夹持器各部件的变换矩阵
    
    参数：
        meshes (dict): 夹持器网格字典
        open_ratio (float): 张开比例 (0-1)
    
    返回：
        dict: 变换矩阵字典
    """
    transforms = {k: np.eye(4) for k in meshes.keys()}

    close_extent = 0.8757 * (1 - open_ratio)

    transforms['robotiq_left_outer_knuckle'] = get_translate_matrix([3.06011444260539, 0.0, 6.27920162695395]) @ get_revolute_matrix('Y', -close_extent)
    transforms['robotiq_left_outer_finger'] = transforms['robotiq_left_outer_knuckle'] @ get_translate_matrix([3.16910442266543, 0.0, -0.193396375724605])
    transforms['robotiq_left_inner_knuckle'] = get_translate_matrix([1.27000000001501, 0.0, 6.93074999999639]) @ get_revolute_matrix('Y', -close_extent)
    transforms['robotiq_left_inner_finger'] = transforms['robotiq_left_inner_knuckle'] @ get_translate_matrix([3.4585310861294003, 0.0, 4.5497019381797505]) @ get_revolute_matrix('Y', close_extent)

    transforms['robotiq_right_outer_knuckle'] = get_transform_matrix_quat([-3.06011444260539, 0.0, 6.27920162695395], [0, 0, 0, 1]) @ get_revolute_matrix('Y', -close_extent)
    transforms['robotiq_right_outer_finger'] = transforms['robotiq_right_outer_knuckle'] @ get_translate_matrix([3.16910442266543, 0.0, -0.193396375724605])
    transforms['robotiq_right_inner_knuckle'] = get_transform_matrix_quat([-1.27000000001501, 0.0, 6.93074999999639], [0, 0, 0, 1]) @ get_revolute_matrix('Y', -close_extent)
    transforms['robotiq_right_inner_finger'] = transforms['robotiq_right_inner_knuckle'] @ get_translate_matrix([3.4585310861294003, 0.0, 4.5497019381797505]) @ get_revolute_matrix('Y', close_extent)

    return transforms


def get_robotiq_140_meshes_transforms(meshes, open_ratio):
    """
    获取Robotiq 140夹持器各部件的变换矩阵
    
    参数：
        meshes (dict): 夹持器网格字典
        open_ratio (float): 张开比例 (0-1)
    
    返回：
        dict: 变换矩阵字典
    """
    transforms = {k: np.eye(4) for k in meshes.keys()}

    close_extent = 0.8757 * (1 - open_ratio)

    transforms['robotiq_left_outer_knuckle'] = get_transform_matrix_quat([0.0, -3.0601, 5.4905], [0.41040502, 0.91190335, 0.0, 0.0]) @ get_revolute_matrix('X', -close_extent)
    transforms['robotiq_left_outer_finger'] = transforms['robotiq_left_outer_knuckle'] @ get_translate_matrix([0.0, 1.821998610742, 2.60018192872234])
    transforms['robotiq_left_inner_finger'] = transforms['robotiq_left_outer_finger'] @ get_transform_matrix_quat([0.0, 8.17554015893473, -2.82203446692936], [0.93501321, -0.35461287, 0.0, 0.0]) @ get_revolute_matrix('X', close_extent)
    transforms['robotiq_left_pad'] = transforms['robotiq_left_inner_finger'] @ get_transform_matrix_quat([0.0, 3.8, -2.3], [0, 0, 0.70710678, 0.70710678])
    transforms['robotiq_left_inner_knuckle'] = get_transform_matrix_quat([0.0, -1.27, 6.142], [0.41040502, 0.91190335, 0.0, 0.0]) @ get_revolute_matrix('X', -close_extent)

    transforms['robotiq_right_outer_knuckle'] = get_transform_matrix_quat([0.0, 3.0601, 5.4905], [0.0, 0.0, 0.91190335, 0.41040502]) @ get_revolute_matrix('X', -close_extent)
    transforms['robotiq_right_outer_finger'] = transforms['robotiq_right_outer_knuckle'] @ get_translate_matrix([0.0, 1.821998610742, 2.60018192872234])
    transforms['robotiq_right_inner_finger'] = transforms['robotiq_right_outer_finger'] @ get_transform_matrix_quat([0.0, 8.17554015893473, -2.82203446692936], [0.93501321, -0.35461287, 0.0, 0.0]) @ get_revolute_matrix('X', close_extent)
    transforms['robotiq_right_pad'] = transforms['robotiq_right_inner_finger'] @ get_transform_matrix_quat([0.0, 3.8, -2.3], [0, 0, 0.70710678, 0.70710678])
    transforms['robotiq_right_inner_knuckle'] = get_transform_matrix_quat([0.0, 1.27, 6.142], [0.0, 0.0, -0.91190335, -0.41040502]) @ get_revolute_matrix('X', -close_extent)

    return transforms


def get_ft_sensor_mesh_transform(gripper_type):
    """
    获取力扭矩传感器的变换矩阵
    
    参数：
        gripper_type (str): 夹持器类型
    
    返回：
        numpy.ndarray: 变换矩阵 (4x4)
    """
    spec = get_ft_sensor_spec()

    # 计算变换矩阵以将圆柱体对齐到夹持器的基准方向
    base_basis_direction, _ = get_gripper_basis_directions(gripper_type)
    base_basis_direction /= np.linalg.norm(base_basis_direction)
    default_axis = np.array([0, 0, 1])
    if np.allclose(base_basis_direction, default_axis):
        rotation_matrix = np.eye(3)
    else:
        rot_vec = np.cross(default_axis, base_basis_direction)
        angle = np.arccos(np.dot(default_axis, base_basis_direction))
        rotation_matrix = R.from_rotvec(rot_vec * angle).as_matrix()
    
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = (spec['height'] / 2) * base_basis_direction
    return transform


def get_gripper_meshes_transforms(gripper_type, meshes, pos, quat, pose, open_ratio):
    """
    获取夹持器各部件的变换矩阵
    
    参数：
        gripper_type (str): 夹持器类型
        meshes (dict): 夹持器网格字典
        pos (list): 夹持器位置 [x, y, z]
        quat (list): 夹持器姿态四元数 [x, y, z, w]
        pose (dict, optional): 姿态字典，用于覆盖pos和quat
        open_ratio (float): 张开比例 (0-1)
    
    返回：
        dict: 变换矩阵字典
    """
    if gripper_type == 'panda':
        transforms = get_panda_meshes_transforms(meshes, open_ratio)
    elif gripper_type == 'robotiq-85':
        transforms = get_robotiq_85_meshes_transforms(meshes, open_ratio)
    elif gripper_type == 'robotiq-140':
        transforms = get_robotiq_140_meshes_transforms(meshes, open_ratio)
    else:
        raise NotImplementedError
    
    if 'ft_sensor' in meshes:
        transforms['ft_sensor'] = get_ft_sensor_mesh_transform(gripper_type)
    
    pos, quat = get_pos_quat_from_pose(pos, quat, pose)
    for name, transform in transforms.items():
        apply_transform(transform, get_transform_matrix_quat(pos, quat))
    return transforms


def transform_gripper_meshes(gripper_type, meshes, pos, quat, pose, open_ratio):
    """
    变换夹持器各部件到指定位置和姿态
    
    参数：
        gripper_type (str): 夹持器类型
        meshes (dict): 夹持器网格字典
        pos (list): 夹持器位置 [x, y, z]
        quat (list): 夹持器姿态四元数 [x, y, z, w]
        pose (dict, optional): 姿态字典，用于覆盖pos和quat
        open_ratio (float): 张开比例 (0-1)
    
    返回：
        dict: 变换后的夹持器网格字典
    """
    transforms = get_gripper_meshes_transforms(gripper_type, meshes, pos, quat, pose, open_ratio)
    meshes = {k: v.copy() for k, v in meshes.items()}
    for name, mesh in meshes.items():
        mesh.apply_transform(transforms[name])
    return meshes


'''Arms - 机械臂处理模块'''


def load_arm_meshes(arm_type, asset_folder, visual=False, convex=True, combined=True):
    """
    加载机械臂的网格模型（链接名称与网格文件之间的映射）
    
    参数：
        arm_type (str): 机械臂类型
        asset_folder (str): 资产文件夹路径
        visual (bool): 是否加载视觉模型（否则加载碰撞模型）
        convex (bool): 是否使用凸包（仅对碰撞模型有效）
        combined (bool): 是否合并网格
    
    返回：
        dict: 机械臂网格字典
    """
    meshes = {}
    if arm_type == 'xarm7':
        if visual:
            meshes['linkbase'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'visual', 'linkbase_smooth.obj'))
            for i in range(1, 8):
                meshes[f'link{i}'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'visual', f'link{i}_smooth.obj'))
        else:
            meshes['linkbase'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'collision', 'linkbase_vhacd.obj'))
            for i in range(1, 8):
                meshes[f'link{i}'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'collision', f'link{i}_vhacd.obj'))
    elif arm_type == 'panda':
        if visual:
            for i in range(0, 8):
                meshes[f'panda_link{i}'] = trimesh.load(os.path.join(asset_folder, 'panda', 'visual', f'link{i}.obj'))
        else:
            for i in range(0, 8):
                meshes[f'panda_link{i}'] = trimesh.load(os.path.join(asset_folder, 'panda', 'collision', f'link{i}.obj'))
    elif arm_type == 'ur5e':
        linknames = ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
        if visual:
            for name in linknames:
                filename = name.replace('_link', '').replace('_', '')
                meshes[name] = trimesh.load(os.path.join(asset_folder, 'ur5e', 'visual', f'{filename}.obj'))
        else:
            for name in linknames:
                filename = name.replace('_link', '').replace('_', '')
                meshes[name] = trimesh.load(os.path.join(asset_folder, 'ur5e', 'collision', f'{filename}.obj'))
    else:
        raise NotImplementedError
    if not visual and convex:
        meshes = {k: v.convex_hull for k, v in meshes.items()}
    if combined:
        meshes = get_combined_meshes(meshes)
    return meshes


def get_arm_meshes_transforms(meshes, chain, q):
    """
    获取机械臂各部件的变换矩阵
    
    参数：
        meshes (dict): 机械臂网格字典
        chain (pybullet_robotics.Robot): 机械臂链对象
        q (list): 关节角度
    
    返回：
        dict: 变换矩阵字典
    """
    transforms = {k: np.eye(4) for k in meshes.keys()}
    matrices = chain.forward_kinematics(q, full_kinematics=True)
    for name, matrix in zip(transforms.keys(), matrices):
        transforms[name] = matrix
    return transforms


def transform_arm_meshes(meshes, chain, q):
    """
    变换机械臂各部件到指定关节角度
    
    参数：
        meshes (dict): 机械臂网格字典
        chain (pybullet_robotics.Robot): 机械臂链对象
        q (list): 关节角度
    
    返回：
        dict: 变换后的机械臂网格字典
    """
    transforms = get_arm_meshes_transforms(meshes, chain, q)
    meshes = {k: v.copy() for k, v in meshes.items()}
    for name, mesh in meshes.items():
        mesh.apply_transform(transforms[name])
    return meshes