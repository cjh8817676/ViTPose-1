#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:08:24 2023

@author: m11002125
"""

import os
import cv2
import sys

import cv2

def extract_video_segment(input_video, output_video, start_time, end_time):
    # 開啟原始影片檔案
    video = cv2.VideoCapture(input_video)

    # 確保影片檔案正確開啟
    if not video.isOpened():
        print("無法開啟影片檔案")
        return

    # 取得原始影片的基本資訊
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 計算開始和結束的幀數
    start_frame = start_time
    end_frame = end_time

    # 計算切割後影片的總幀數和總時間長度
    total_frames = end_frame - start_frame + 1
    total_time = total_frames / fps

    # 建立新的影片寫入物件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 將原始影片的當前幀設置到開始的位置
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 迭代原始影片中的幀，直到達到結束的幀數或影片結束
    while video.isOpened():
        # 讀取當前幀
        ret, frame = video.read()

        # 檢查幀是否成功讀取
        if not ret:
            break

        # 將幀寫入新的影片
        output.write(frame)

        # 檢查是否到達結束的幀數
        if video.get(cv2.CAP_PROP_POS_FRAMES) == end_frame:
            break

    # 釋放影片物件和影片寫入物件
    video.release()
    output.release()

    print("影片切割完成")
    
def read_files_in_path(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                # Process the content of the file as needed
                print(content)

if __name__ == '__main__':
    # 呼叫函式進行影片切割和保存
    video_name = "8.MOV"
    input_video = f"/home/m11002125/ViTPose/test_video/takedev_gt/{video_name}"
    output_video = f"/home/m11002125/ViTPose/vis_results/gym_lab/{video_name}"
    
    video_name = video_name.split('.')[0]
    mask_path = f"/home/m11002125/ViTPose/workspace/{video_name}/masks"
    
    all_file = sorted(os.listdir(mask_path))
    
    
    start_frame =  int(all_file[0].split('.')[0]) # 起始幀數
    end_frame =   int(all_file[-1].split('.')[0]) # 結束幀數

    # print(start_frame,',',end_frame)

    extract_video_segment(input_video, output_video, start_frame, end_frame)
