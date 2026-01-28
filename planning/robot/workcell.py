#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workcell.py - 机器人工作单元配置模块

该模块定义了不同类型机械臂的工作单元配置参数，包括：
1. 机械臂位置配置
2. 机械臂姿态配置
3. 工作空间边界
4. 装配中心位置
5. 夹具位置限制

支持的机械臂类型：
- xarm7
- panda
- ur5e
"""

import numpy as np


def get_board_dx():
    """
    获取工作板的基本单位长度
    
    返回：
        float: 基本单位长度
    """
    return 2.5


def get_move_arm_pos(arm_type):
    """
    获取移动臂的位置坐标
    
    参数：
        arm_type (str): 机械臂类型
    
    返回：
        numpy.ndarray: 移动臂位置坐标 [x, y, z]
    """
    dx = get_board_dx()
    if arm_type == 'xarm7':
        return np.array([14.5 * dx, 10 * dx, 0])
    elif arm_type == 'panda':
        return np.array([18 * dx, 8 * dx, 0])
    elif arm_type == 'ur5e':
        return np.array([18 * dx, 10 * dx, 0])
    else:
        raise NotImplementedError


def get_hold_arm_pos(arm_type):
    """
    获取夹持臂的位置坐标
    
    参数：
        arm_type (str): 机械臂类型
    
    返回：
        numpy.ndarray: 夹持臂位置坐标 [x, y, z]
    """
    dx = get_board_dx()
    if arm_type == 'xarm7':
        return np.array([-14.5 * dx, 10 * dx, 0])
    elif arm_type == 'panda':
        return np.array([-18 * dx, 8 * dx, 0])
    elif arm_type == 'ur5e':
        return np.array([-18 * dx, 10 * dx, 0])
    else:
        raise NotImplementedError


def get_dual_arm_pos(arm_type):
    """
    获取双臂系统的位置坐标
    
    参数：
        arm_type (str): 机械臂类型
    
    返回：
        tuple: (移动臂位置, 夹持臂位置)
    """
    return get_move_arm_pos(arm_type), get_hold_arm_pos(arm_type)


def get_single_arm_pos(arm_type):
    """
    获取单臂系统的位置坐标
    
    参数：
        arm_type (str): 机械臂类型
    
    返回：
        numpy.ndarray: 机械臂位置坐标 [x, y, z]
    """
    return get_move_arm_pos(arm_type)


def get_move_arm_euler():
    """
    获取移动臂的欧拉角姿态
    
    返回：
        numpy.ndarray: 欧拉角 [roll, pitch, yaw]
    """
    return np.array([0, 0, -np.pi / 2])


def get_hold_arm_euler():
    """
    获取夹持臂的欧拉角姿态
    
    返回：
        numpy.ndarray: 欧拉角 [roll, pitch, yaw]
    """
    return np.array([0, 0, -np.pi / 2])


def get_dual_arm_euler():
    """
    获取双臂系统的欧拉角姿态
    
    返回：
        tuple: (移动臂欧拉角, 夹持臂欧拉角)
    """
    return get_move_arm_euler(), get_hold_arm_euler()


def get_single_arm_euler():
    """
    获取单臂系统的欧拉角姿态
    
    返回：
        numpy.ndarray: 欧拉角 [roll, pitch, yaw]
    """
    return get_move_arm_euler()


def get_move_arm_box(arm_type):
    """
    获取移动臂的工作空间边界框
    
    参数：
        arm_type (str): 机械臂类型
    
    返回：
        tuple: (边界框最小值, 边界框最大值)
    """
    arm_pos = get_move_arm_pos(arm_type)
    return arm_pos - np.array([100.0, 100.0, 0.0]), arm_pos + np.array([30.0, 50.0, 80.0])


def get_hold_arm_box(arm_type):
    """
    获取夹持臂的工作空间边界框
    
    参数：
        arm_type (str): 机械臂类型
    
    返回：
        tuple: (边界框最小值, 边界框最大值)
    """
    arm_pos = get_hold_arm_pos(arm_type)
    return arm_pos - np.array([30.0, 100.0, 0.0]), arm_pos + np.array([100.0, 50.0, 80.0])


def get_dual_arm_box(arm_type):
    """
    获取双臂系统的工作空间边界框
    
    参数：
        arm_type (str): 机械臂类型
    
    返回：
        tuple: (移动臂边界框, 夹持臂边界框)
    """
    return get_move_arm_box(arm_type), get_hold_arm_box(arm_type)


def get_single_arm_box(arm_type):
    """
    获取单臂系统的工作空间边界框
    
    参数：
        arm_type (str): 机械臂类型
    
    返回：
        tuple: (边界框最小值, 边界框最大值)
    """
    return get_move_arm_box(arm_type)


def get_assembly_center(arm_type):
    """
    获取装配中心位置
    
    参数：
        arm_type (str): 机械臂类型
    
    返回：
        numpy.ndarray: 装配中心位置坐标 [x, y, z]
    """
    dx = get_board_dx()
    if arm_type == 'xarm7':
        return np.array([0, -6 * dx, 0])
    elif arm_type == 'panda':
        return np.array([0, -6 * dx, 0])
    elif arm_type == 'ur5e':
        return np.array([0, -6 * dx, 0])
    else:
        raise NotImplementedError


def get_fixture_min_y(arm_type):
    """
    获取夹具的最小Y坐标
    
    参数：
        arm_type (str): 机械臂类型
    
    返回：
        float: 夹具最小Y坐标
    """
    dx = get_board_dx()
    if arm_type == 'xarm7':
        return 6 * dx
    elif arm_type == 'panda':
        return 4 * dx
    elif arm_type == 'ur5e':
        return 6 * dx
    else:
        raise NotImplementedError