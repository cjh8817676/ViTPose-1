# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pdb
import glob
import json
from PIL import Image
from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks


HUMAN_MASK = 1  # The number of the human mask
BAR_MASK   = 2  # The number of the bar mask
'''
read csv output from my gui and plot height.
尋找頭部最高點的時候，臀部的高度
'''
def find_horizontal_bar_mask(original_mask):
    # pdb.set_trace()
    w , h = original_mask.shape
    min_row = 4000
    min_col = 4000
    max_col = -1
    max_row = -1
    
    rows,cols = np.where(original_mask[:,:]==BAR_MASK)
    min_row = min(rows)
    max_row = max(rows)
    min_col = min(cols)
    max_col = max(cols)
                
    if len(rows) == 0:
        print("no bar mask!")
                
    top_of_bar = [min_col , min_row]
    bottom_of_bar = [max_col , max_row]
    
    print([top_of_bar, bottom_of_bar])
                
    return [top_of_bar, bottom_of_bar]

def read_files(sub_folder,action,order):
    base_path = "output_results"
    sub_path = os.path.join(base_path, sub_folder)
    files = os.listdir(sub_path)

    temp = []
    # pdb.set_trace()
    target_file = f"{sub_folder}_{order}_{action}_"
    for file in files:
        if file.endswith('.json'):
            if target_file in file:
                temp.append(os.path.join(sub_path, file))

    return temp

def plot_data(json_path,mask_path , start, end):
    # mask_path = os.path.join('./workspace/',video_mask, "masks/") 
    
    fnames = sorted(glob.glob(os.path.join(mask_path, '*.jpg')))
    if len(fnames) == 0:
        fnames = sorted(glob.glob(os.path.join(mask_path, '*.png')))
        
    frame_list = []
    for i, fname in enumerate(fnames):
        frame_list.append(np.array(Image.open(fname), dtype=np.uint8))
        break
    mask_data = np.stack(frame_list, axis=0)
    horizontal_bar_points = find_horizontal_bar_mask(mask_data[0])
    # pdb.set_trace()
    # 微調單槓中心點不微調
    center_bar = np.array([(horizontal_bar_points[0][0] + horizontal_bar_points[1][0]) / 2 , horizontal_bar_points[0][1]])
    print('center_bar',center_bar)
    
    # 2.6m / pixel_height_of_bar  為了髖
    pixel_height_ratio = 2.6/ (center_bar[1] - horizontal_bar_points[1][1]) 
    print('pixel_height_ratio',pixel_height_ratio)
    with open(json_path) as f:    # 讀取每一幀的 pose keypoint 和 bbox(左上、右下) 的座標
        pose_data = json.load(f)
        

    hip = []
    for i in pose_data:
        l_hip = i['keypoints'][33:35]
        r_hip = i['keypoints'][36:38]
        hip.append( [(l_hip[0]-r_hip[0])/2 , (l_hip[1]+r_hip[1]) / 2] )

    frames = [i for i in range(len(hip[start:end]))]
    time = [i/120 for i in range(len(hip[start:end]))]
    # pdb.set_trace()
    # for i, data in enumerate(hip_plot_data):
    #     hip_plot_data[i] += 2.6 
    
    head_height_data = []
    for i in hip:
        head_height_data.append(( i[1] - center_bar[1] ) * pixel_height_ratio * (9.6/8.3))
    
    # max_index = np.argmax(head_height_data[start:end])
    # max_time = time[max_index]
    # max_value = head_height_data[max_index]
    
    # pdb.set_trace()
    peaks, _ = find_peaks(head_height_data, height=0, distance=150)
    
    
    
    print('max heiht:',peaks, head_height_data[peaks[-1]])

    plt.figure()
    
    plt.plot(frames, head_height_data[start:end])
    # for i in peaks:
    plt.scatter(peaks[-1], head_height_data[peaks[-1]], c='red', label='Max Value')
    plt.ylabel('hip height(m)')
    plt.xlabel('time(s)')
    plt.legend(['hip height', 'max fly value'], loc='lower left')
    plt.show()
    

if __name__ == "__main__":
    # (951.544),  972,549
    subject = 'sub3'
    action = 'TKATCHEV'
    start = 1
    end = 12
    
    for i in range(start,end+1):
        order = str(i)
        
        files = read_files(subject,action,order)
        json_file = files[0]
        # pdb.set_trace()
        video_mask = json_file.split('/')[-1]
        video_mask = video_mask.replace('vis_','')
        video_mask = video_mask.replace('ViTPose_','')
        video_mask = video_mask.replace('.MOV.json','')
        mask_path = os.path.join('./workspace/',video_mask, "masks/")
        
    
        plot_data(json_file, mask_path, 0, -1)