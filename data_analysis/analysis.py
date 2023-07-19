# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import pdb
import math
import matplotlib.pyplot as plt
import re

center_coord = [951,544]


def moving_average_filter(data, window_size = 5):
    filtered_data = []

    for i in range(window_size, len(data) - window_size):
        window = data[i - window_size: i + window_size + 1]
        avg = sum(window) / len(window)
        filtered_data.append(avg)

    # 補償損失的資料點
    compensate_points = window_size // 2
    compensated_data = data[:compensate_points] + filtered_data + data[-compensate_points:]

    return compensated_data

def median_filter(signal, window_size):
    """
    對一維訊號做中值濾波。
    :param signal: 一維訊號
    :param window_size: 窗口大小
    :return: 濾波後的訊號
    """
    filtered_signal = np.zeros_like(signal)
    for i in range(len(signal)):
        lower_bound = max(0, i - window_size // 2)
        upper_bound = min(len(signal), i + window_size // 2 + 1)
        filtered_signal[i] = np.median(signal[lower_bound:upper_bound])
    return filtered_signal
def calculate_angle_left_body(pointlist):

    p1, p2, p3 = pointlist[0],pointlist[1],pointlist[2]
    """
    Calculates the angle between three points in 2D space.
    Args:
        p1, p2, p3: Each point is a list [x, y]
    Returns:
        angle: angle in degrees
    """
    temp = p1
    p1 = p2
    p2 = temp
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1, 1))
    
    # Adjust the angle's range to [90, 270] degrees if needed.
    if np.cross(v1, v2) > 0:
        angle = 2*np.pi - angle
    
    return np.degrees(angle)
def calculate_angle_right_body(pointlist):
    """
    input a list which have 3 points
    [(hip),(shoulder),(knee)]
    The maximum bending angle that a person can achieve is 0~270 degrees5
    """
    p1 = pointlist[0]
    p2 = pointlist[1]
    p3 = pointlist[2]
    temp = p1
    p1 = p2
    p2 = temp
    # Calculate the angle between the vectors (p1, p2) and (p3, p2)
    angle = math.atan2(p3[1]-p2[1], p3[0]-p2[0]) - math.atan2(p1[1]-p2[1], p1[0]-p2[0])
    angle = math.degrees(angle)
    
    return angle+360 if angle < 0 else angle
def read_files(sub_folder):
    base_path = "output_results"
    sub_path = os.path.join(base_path, sub_folder)
    files = os.listdir(sub_path)

    fail_files = []
    success_files = []

    for file in files:
        if file.endswith('.json'):
            if "fail" in file:
                fail_files.append(os.path.join(sub_path, file))
            elif "success" in file:
                success_files.append(os.path.join(sub_path, file))

    return fail_files, success_files
def process_json_data(json_data):
    keypoints = ['l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
                 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    body_parts = {
        'l_eye': slice(3, 5),
        'r_eye': slice(6, 8),
        'l_ear': slice(9, 11),
        'r_ear': slice(12, 14),
        'l_shoulder': slice(15, 17),
        'r_shoulder': slice(18, 20),
        'l_elbow': slice(21, 23),
        'r_elbow': slice(24, 26),
        'l_wrist': slice(27, 29),
        'r_wrist': slice(30, 32),
        'l_hip': slice(33, 35),
        'r_hip': slice(36, 38),
        'l_knee': slice(39, 41),
        'r_knee': slice(42, 44),
        'l_ankle': slice(45, 47),
        'r_ankle': slice(48, 50)
    }

    keypoints_data = {part: [] for part in keypoints}
    angle_data = {'left_hip': [], 'left_shoulder': [], 'right_hip': [], 'right_shoulder': []}
    height = []

    for i in json_data:
        for part, indices in body_parts.items():
            keypoints_data[part].append(i['keypoints'][indices])

    for counter in range(len(json_data)):
        angle_data['left_hip'].append(calculate_angle_left_body([keypoints_data['l_hip'][counter],
                                                                 keypoints_data['l_knee'][counter],
                                                                 keypoints_data['l_shoulder'][counter]]))
        angle_data['left_shoulder'].append(calculate_angle_left_body([keypoints_data['l_shoulder'][counter],
                                                                      keypoints_data['l_wrist'][counter],
                                                                      keypoints_data['l_hip'][counter]]))
        angle_data['right_hip'].append(calculate_angle_right_body([keypoints_data['r_hip'][counter],
                                                                   keypoints_data['r_knee'][counter],
                                                                   keypoints_data['r_shoulder'][counter]]))
        angle_data['right_shoulder'].append(calculate_angle_right_body([keypoints_data['r_shoulder'][counter],
                                                                        keypoints_data['r_wrist'][counter],
                                                                        keypoints_data['r_hip'][counter]]))
        
    
    return angle_data, height


if __name__ == "__main__":
    # pdb.set_trace()
    # 讀取 'sub1' 的所有檔案
    subject = 'sub3'
    
    fail_files, success_files = read_files(subject)
    fail_files = sorted(fail_files, key=lambda x: int(re.search(f'{subject}_(\d+)_', x).group(1)))
    success_files = sorted(success_files, key=lambda x: int(re.search(f'{subject}_(\d+)_', x).group(1)))

    # fail_files = sorted(fail_files)
    # success_files = sorted(success_files)
     
    fail_json = []            
    success_json = []
    
    # read results of pose estimation.
    for file in success_files:
        print(file)
        with open(file) as f:    # 讀取每一幀的 pose keypoint 和 bbox(左上、右下) 的座標
            success_json.append(json.load(f))
    for file in fail_files:
        print(file)
        with open(file) as f:    # 讀取每一幀的 pose keypoint 和 bbox(左上、右下) 的座標
            fail_json.append(json.load(f))
            

     
    success_rhip_angle_data = []
    fail_rhip_angle_data = []
    success_lhip_angle_data = []
    fail_lhip_angle_data = []
    
    success_rshoulder_angle_data = []
    fail_rshoulder_angle_data = []
    success_lshoulder_angle_data = []
    fail_lshoulder_angle_data = []
    
    for json_data in success_json:
        angle_data,height = process_json_data(json_data)
        
        success_rhip_angle_data.append(angle_data['right_hip'])
        success_lhip_angle_data.append(angle_data['left_hip'])
        success_rshoulder_angle_data.append(angle_data['right_shoulder'])
        success_lshoulder_angle_data.append(angle_data['left_shoulder'])
    
    for json_data in fail_json:
        angle_data,height = process_json_data(json_data)
        fail_rhip_angle_data.append(angle_data['right_hip'])
        fail_lhip_angle_data.append(angle_data['left_hip'])
        fail_rshoulder_angle_data.append(angle_data['right_shoulder'])
        fail_lshoulder_angle_data.append(angle_data['left_shoulder'])   
        
    # 将所有数据列表放在一个列表中 (according to video to choose left or right )
    # sub3_list are set by observing the frame of film from hand stand to release.
    # [fail;fail;success;success]
    sub3_list = [[(0,735),(0,740),(0,690)],[(0,735),(0,740),(0,690)],[(0,715),(0,740),(0,751)],[(0,715),(0,740),(0,751)]] # for fail and success
    # sub3_list = [[(0,715),(0,740),(0,751)],[(0,715),(0,740),(0,751)]] # for succss
    sub2_list = [[(0,152),(0,163),(0,159)],[(0,152),(0,163),(0,159)]] # for succss
    sub1_list = [[(0,410)],[(0,410),(0,426),(0,440)],[(0,410)],[(0,410),(0,426),(0,440)]] # for fail and success
    # sub1_list = [[(0,410),(0,426),(0,440)],[(0,410),(0,426),(0,440)]] # for success
    # sub1_list = [[(0,410)],[(0,410)]] # for fail
    
    height = [[3.714,3.693,3.828],[3.714,3.693,3.828],[3.793,3.781,3.739],[3.793,3.781,3.739]]
    
    # 要按照hip、shoulder、hip、shoulder...排列。
    # sub1 : left hip,all right
    if subject=='sub1':
        data_lists = [
            fail_lhip_angle_data,
            fail_rshoulder_angle_data
            # success_lhip_angle_data,
            # success_rshoulder_angle_data
        ]
     # sub2 : left hip, rigth shoulder
    elif subject=='sub2':
        data_lists = [
            success_lhip_angle_data,
            success_rshoulder_angle_data
        ]
    elif subject== 'sub3':
    # sub3 : all left
        data_lists = [
            fail_lhip_angle_data,
            fail_lshoulder_angle_data,
            success_lhip_angle_data,
            success_lshoulder_angle_data
        ]
    # selecct sub1,sub2,sub3
    if subject=='sub1':
        sub_list = sub1_list
    elif subject=='sub2':
        sub_list = sub2_list
    elif subject== 'sub3':
        sub_list = sub3_list
        
    


            
            
            
            
            