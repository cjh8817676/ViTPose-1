# -*- coding: utf-8 -*-
'''
利用該程式碼，一幀一幀的比較模型輸出的差異
'''


# importing libraries
import cv2
import numpy as np
  
# Create a VideoCapture object and read from input file
cap1 = cv2.VideoCapture("/home/m11002125/AlphaPose-1/ViTPose/vis_results/vis_cat_jump.mp4")
cap2 = cv2.VideoCapture('/home/m11002125/AlphaPose-1/ViTPose/vis_results/vis_hrnet_cat_jump.mp4')


frameSize = (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 影片長寬

datalen = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
fourcc = int(cap1.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
fps = cap1.get(cv2.CAP_PROP_FPS)                  # 查看 FPS


videoWriter = cv2.VideoWriter('./optput.mp4',fourcc,fps, (frameSize[0]*2,frameSize[1]))
        
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
# Check if camera opened successfully
if (cap1.isOpened()== False):
    print("Error opening video file")
  
# Read until video is completed
while(cap1.isOpened() and cap2.isOpened()):
      
# Capture frame-by-frame
    ret_hr, frame_hr = cap1.read()
    ret_res, frame_res = cap2.read()
    
    if ret_hr and ret_res:
    # Display the resulting frame
        
        numpy_horizontal_concat = np.concatenate((frame_hr, frame_res ), axis=1)
        
        image = cv2.putText(numpy_horizontal_concat, 'hello {}'.format(num_frame), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.imshow('Frame', image)
        videoWriter.write(image)
        num_frame+=1
    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
  
# Break the loop
    else:
        break
  
# When everything done, release
# the video capture object
cap1.release()
cap2.release()
videoWriter.release()
  
# Closes all the frames
cv2.destroyAllWindows()