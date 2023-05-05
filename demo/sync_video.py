#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:53:38 2023

@author: m11002125
"""

# -*- coding: utf-8 -*-
'''
利用該程式碼，一幀一幀的比較模型輸出的差異
'''


# importing libraries
import cv2
import numpy as np
import pdb
# Create a VideoCapture object and read from input file
cap1 = cv2.VideoCapture("/home/m11002125/video1.mp4")
cap2 = cv2.VideoCapture('/home/m11002125/video2.mp4')
datalen1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
fourcc1 = int(cap1.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
fps1 = cap1.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
datalen2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
fourcc2 = int(cap2.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
fps2 = cap2.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))


# 找到影片的開始和結束時間
start_time = 0
end_time = total_frames1 / fps1

# 計算每一影格的時間間隔
frame_interval1 = 1 / fps1
frame_interval2 = 1 / fps2

# 計算影片的總時間
total_time_1 = total_frames1 / fps1
total_time_2 = total_frames2 / fps2

# 將兩個影片的當前影格設置為其開始影格
cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 創建一個視頻寫入器，將同步的影片寫入新的文件中
videoWriter = cv2.VideoWriter('synced_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps1, (2*int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2

num_frame=0
# 逐幀讀取兩個影片，直到其中一個影片已到達其結束
while True:
    # 從兩個影片中讀取當前幀
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    # 如果其中一個影片已經讀取到結束，則退出循環
    if not ret1 or not ret2:
        break
    
    # 計算兩個影片當前的時間戳
    timestamp1 = start_time + cap1.get(cv2.CAP_PROP_POS_FRAMES) / fps1
    timestamp2 = start_time + cap2.get(cv2.CAP_PROP_POS_FRAMES) / fps2
    
    numpy_horizontal_concat = np.concatenate((frame1, frame2 ), axis=1)
    # pdb.set_trace()

    image = cv2.putText(numpy_horizontal_concat, 'hello {}'.format(num_frame), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    
    # cv2.imshow('Frame', image)
    num_frame+=1
    # 如果兩個影片的時間戳相差不超過一個影格的時間間隔，則將這兩幀合併為一幀
    if abs(timestamp1 - timestamp2) < 1 / fps1:
        merged_frame = cv2.hconcat([frame1, frame2])
        # pdb.set_trace()
        videoWriter.write(merged_frame)
    # 如果兩個影片的時間戳相差超過一個影格的時間間隔，則只寫入其中一個影片的當前幀
    else:
        if timestamp1 < timestamp2:
            videoWriter.write(np.concatenate((frame1, frame1 ), axis=1))
        else:
            videoWriter.write(np.concatenate((frame2, frame2 ), axis=1))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
  
    
# 釋放資源
cap1.release()
cap2.release()
videoWriter.release()

cv2.destroyAllWindows()







