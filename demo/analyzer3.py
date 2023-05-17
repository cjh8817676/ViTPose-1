import sys,os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import json
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from PyQt5 import uic
import pandas as pd
import pdb
import glob
from PIL import Image
file_path = os.path.dirname(__file__)
from numba import jit
from collections import defaultdict
from defaultlist import defaultlist



imu_data_pd = pd.DataFrame()
imu_data_len = 0
imu_sample_rate = 400
imu_data_gyrox = []
imu_data_gyroy = []
imu_data_gyroz = []
imu_data_accx = []
imu_data_accy = []
imu_data_accz = []
imu_data_haccx = []
imu_data_haccy = []
imu_data_haccz = []
imu_data_left = 0
imu_data_right = 1200



handoff_radius = 30 # 30 pixel

# coco pose
'''
[nose,leye,reye,lear,rear,lshoulder,rshoulder,lelbow,relbow,lwrist,rwrist,lhip,rhip,lknee,rknee,lankle,rankle]
'''

# y_true : np.array() ' predictions : list
def mae(y_true, predictions):
    # pdb.set_trace()
    while (y_true.shape[0] < len(predictions)):
        predictions.pop()
    
    while (y_true.shape[0] > len(predictions)):
        y_true = y_true[0:y_true.shape[0]-1]
    
    
    y_true, predictions = y_true, np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

# mask of human is red
# @jit(nopython=True)
def find_human_mask(original_mask): # shape (w,h,3)  RGB
    # pdb.set_trace()
    h, w, c = original_mask.shape
    # human_mask = np.zeros((w,h,3)) # plt.imshow(human_mask) to verify if we only get human mask
    human_mask_point = []
    

    # for i in range(0,h):    # row
    #     for j in range(0,w):  # col
    #         if np.array_equal(original_mask[i][j],np.array([128,0,0])):
    #             # human_mask[i][j] = np.array([128,0,0])
    #             human_mask_point.append([j,i])  # [col, row] 為了之後方便計算,倒過來放
    
    rows,cols = np.where((original_mask[:,:,0]==128) & (original_mask[:,:,1]==0) & (original_mask[:,:,2]==0))

    
    for i,j in zip(rows,cols):
        human_mask_point.append([j,i])
        # print(i,j)
    
                
    return human_mask_point

# mask of high bar is green
# @jit(nopython=True)
def find_horizontal_bar_mask(original_mask):
    # pdb.set_trace()

    w , h, c = original_mask.shape

    min_row = 4000
    min_col = 4000
    max_col = -1
    max_row = -1
    

    rows,cols = np.where((original_mask[:,:,0]==0) & (original_mask[:,:,1]==128) & (original_mask[:,:,2]==0))

    min_row = min(rows)
    max_row = max(rows)
    min_col = min(cols)
    max_col = max(cols)
                
    
    if len(rows) == 0:
        print("no bar mask!")
                
    top_of_bar = [min_col , min_row]
    bottom_of_bar = [max_col , max_row]
                
    return [top_of_bar, bottom_of_bar]

# get closet distance from a sets and a point
# @jit(nopython=True)
def get_mini_distance(point,mask_set):
    latest_distance =4000
    for i in mask_set:
        distance = np.sqrt(np.sum(np.square(point-i)))
        if latest_distance > distance:
            latest_distance = distance
    
    return latest_distance
# @jit(nopython=True)
def check_handoff(bar_point,human_hand,human_mask):
    # bar_point: left_top and roght bottom [[min_col,min_row],[max_col,max_row]] (row: h col : w)
    # human_mask: many points as mask, enery points is [col, row]
    # pdb.set_trace()
    bar_highest_point = np.array([ ( bar_point[0][0] + bar_point[1][0] ) / 2 , bar_point[0][1] + 5])  # (+2)降低一點單槓高度 [col,row]
    
            
    left_wrist = human_hand[0] # [col,row]
    right_wrist = human_hand[1] # [col,row]
    
    # (self.horizontal_bar_points[0][0] + self.horizontal_bar_points[1][0])/2 , self.horizontal_bar_points[0][1]+5), 15, (255,0,0), 2)

    # didtance between left wrist and bar. (unit : pixel )
    dblwb = ((left_wrist[0] - bar_highest_point[0])**2 + (left_wrist[1] - bar_highest_point[1])**2 )**0.5
    # didtance between right wrist and bar. (unit : pixel)
    dbrwb = ((right_wrist[0] - bar_highest_point[0])**2 + (right_wrist[1] - bar_highest_point[1])**2 )**0.5
    points_of_human = len(human_mask[0])
    
    num_of_over_bar = 0
    
    for i in human_mask[0]:
        if i[1] < bar_highest_point[1]:
            num_of_over_bar+=1
            
    '''
    "Human over bar" and "hand-bar distance long enough" 
    '''
    if num_of_over_bar / points_of_human > 0.95 and dblwb > handoff_radius and dbrwb > handoff_radius:
        return True
    else:
        return False


# pt1 = [x1,y1] , t2 = [x2,y2]    

def getGradient(pt1,pt2):
    return (pt2[1]-pt1[1]) / (pt2[0]-pt1[0])
    

def getAngle(pointlist):

    pt1,pt2,pt3 = pointlist
    
    m1 = getGradient(pt1, pt2)
    m2 = getGradient(pt1, pt3)
    
    angR =math.atan((m2-m1)/(1+(m2*m1)))
    
    angD = (math.degrees(angR))
    return -angD



# 多個 Widget 組成 Layout。  Lauout 又可以互相組合。
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("./demo/test_gui.ui", self)
        
        # start frame
        self.start_frame = 0
        self.end_frame = 0
        
        # human center list
        self.human_center_list = []
        self.human_feet = []
        
        # create Upload_video button
        self.Upload_video.clicked.connect(self.open_file)

        # Main canvas -> QLabel definition
        self.main_canvas.setSizePolicy(QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignCenter)
        # self.main_canvas.setMinimumSize(100, 100)
        self.main_canvas.setMouseTracking(True) # Required for all-time tracking

        # timeline slider
        self.tl_slider.valueChanged.connect(self.tl_slide)  # 只要被拖拉，或是被set_value就會執行self.tl_slide
        self.tl_slider.setTickPosition(QSlider.TicksBelow)
        
        # some buttons
        self.playBtn.setEnabled(False)
        self.playBtn.clicked.connect(self.on_play)

        # playing flag
        self.playing_flag = False
        self.curr_frame_dirty = False

        # Create label to display current position
        self.position_label.setText('Position: 0')

        # 紀錄播放到的幀數
        self.cursur = 0
        
        # button
        self.right_one_frame.setEnabled(False)
        self.right_one_frame.clicked.connect(self.on_next_frame)
        self.left_one_frame.setEnabled(False)
        self.left_one_frame.clicked.connect(self.on_prev_frame)
        

        # motion/action data
        self.mask_data = defaultlist()    # mask for every frame(Image)
        self.human_mask_data = defaultdict(list) # {num of frame :  points of mask (coordinate)}  
        self.horizontal_bar_points = defaultlist()  # [highest point(row,col) , lowest point] of mask of bar (coordinate)
        self.pose_data = defaultlist()    # keypoint bbox for every frame
        self.twist_data = defaultlist()   # speed of loop motion of every frame
        
        # cursur timer
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.on_time)

        #height pg
        self.gt_height_data = []
        self.head_height_data = [] # butt height of every frame
        self.count=0
        self.height_pg.setLabel('bottom','Time','s')
        self.height_pg.showGrid(x = True, y = True, alpha = 1) 
        self.height_pg.setLabel('left','meter(m)')
        # self.gt_height_curve = self.height_pg.plot(self.gt_height_data,pen=(255,0,0)) # 搭配 setData for indicate
        # self.height_curve = self.height_pg.plot(self.head_height_data) # 搭配 setData for indicate
        self.height_pg_timer = pg.QtCore.QTimer()
        self.height_pg_timer.timeout.connect(self.update_height)

        
        self.gt_belly_data = []
        self.twist_data = [0]                                          # no twist speed in first frame 
    

    def open_file(self):
        self.head_height_data = []
        self.hand_coordinate = []

        self.gt_belly_data = []

        self.twist_data = [0]                                          # no twist speed in first frame 
        self.hip_angle_data=[]
        self.handoff_data = []
        self.head_coord_data = []

        global imu_data_pd,imu_data_gyrox,imu_data_gyroy,imu_data_gyroz
        global imu_data_accx,imu_data_accy,imu_data_accz
        global imu_data_haccx,imu_data_haccy,imu_data_haccz, imu_data_len
        global imu_data_left,imu_data_right
        imu_data_left = 0
        imu_data_right = 1200

        self.filename, _ = QFileDialog.getOpenFileName(self, "Open Video")

        video_name = self.filename.split('x')[-1]
        self.video_name = video_name.split('.')[0]
        video_mask = self.video_name.split('/')[-1]
        video_mask = video_mask.replace('vis_','')
        self.mask_path = os.path.join('/home/m11002125/ViTPose/workspace/',video_mask, "masks/") 
        self.motion_json_path = os.path.join('/home/m11002125/ViTPose/vis_results/',video_mask+'.mp4.json')


        self.gt_head_path = os.path.join('/home/m11002125/ViTPose/ground_truth/',video_mask+'_head.xlsx')
        self.gt_belly_path = os.path.join('/home/m11002125/ViTPose/ground_truth/',video_mask+'_belly.xlsx')


        # read results of pose estimation.
        with open(self.motion_json_path) as f:    # 讀取每一幀的 pose keypoint 和 bbox(左上、右下) 的座標
            self.pose_data = json.load(f)
        
        # load mask to self.mask_data and find human mask
        self.load_mask()
        # load ground truth of parameter:
        # gt head:
        if os.path.isfile(self.gt_head_path):
            excel_data = pd.read_excel(self.gt_head_path)
            excel_data = excel_data.tail(-1) # remove first row
            excel_data.set_axis( ['step','frame','time','pixelx','pixely','x','y','r','theta_r','theta','w','wa'], axis='columns', inplace=True)
            self.gt_height_data = np.array(excel_data['y'].to_numpy(), dtype=float)
        else:
            self.gt_height_data = None
            print('no head ground truth')
        # pdb.set_trace()
        # gt belly:
        if os.path.isfile(self.gt_belly_path):
            belly_data = pd.read_excel(self.gt_belly_path)
            belly_data = belly_data.tail(-1) # remove first row
            belly_data.set_axis( ['step','frame','time','pixelx','pixely','x','y','r','theta_r','theta','w','wa'], axis='columns', inplace=True)

            self.gt_belly_data = np.array(belly_data['w'].to_numpy(), dtype=float) #　get angular speeds

        else:
            self.gt_belly_data = None
            print('no belly ground truth')


        # read all video and save every frame in stack
        stream = cv2.VideoCapture(self.filename)                    # 影像路徑
        self.num_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))          # fourcc:  編碼的種類 EX:(MPEG4 or H264)
        self.fps = stream.get(cv2.CAP_PROP_FPS)                     # 查看 FPS
        print('fps:',self.fps)
        print('num_of_frames:',self.num_frames)
        self.w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))          # 影片寬
        self.h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))         # 影片高

        video_frame = []
        counter = 0

        # center bar coord 
        # TODO: 單槓中心點的抓取方式，利用演算法 or 手動提供。
        center_bar = np.array([(self.horizontal_bar_points[0][0] + self.horizontal_bar_points[1][0]) / 2 , self.horizontal_bar_points[0][1] + 17]) # 微調單槓中心點
        
        # 2.8m / pixel_height_of_bar
        pixel_height_ratio = 2.8 / (center_bar[1] - self.horizontal_bar_points[1][1]) 


        # human center coord in start frame (butt center)
        human_center = np.array([(self.pose_data[counter]['keypoints'][33] + self.pose_data[counter]['keypoints'][36]) / 2,
                        (self.pose_data[counter]['keypoints'][34] + self.pose_data[counter]['keypoints'][37]) / 2])
        
        # distance between center bar and hip
        # radius = (center_bar**2 - human_center**2)**0.5
        
        # 3 points to calculate angle
        pointlist = [center_bar,human_center,np.array([])]
        delta_theta_list = []
        delta_t = 1 / self.fps
        # calculate motion parameter from every frame
        while(stream.isOpened()):
            _, frame = stream.read()
            if frame is None:
                break
            try:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frame.append(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                # height of gymnast (這裡座標要改成由左下(左下)往右上)，因為高度是由下往上
                center_head_x = (self.pose_data[counter]['keypoints'][3]+self.pose_data[counter]['keypoints'][6]+self.pose_data[counter]['keypoints'][9]+self.pose_data[counter]['keypoints'][12]) / 4
                center_head_y = (self.pose_data[counter]['keypoints'][4]+self.pose_data[counter]['keypoints'][7]+self.pose_data[counter]['keypoints'][10]+self.pose_data[counter]['keypoints'][13]) / 4
                # pdb.set_trace()
                self.head_coord_data.append(np.array([center_head_x,center_head_y]))
                
                self.head_height_data.append(( center_head_y - center_bar[1] ) * pixel_height_ratio)

                self.hand_coordinate.append([self.pose_data[counter]['keypoints'][27:29],self.pose_data[counter]['keypoints'][30:32]]) # wrist_coordinate
                
                self.human_center_list.append(np.array(np.array([(self.pose_data[counter]['keypoints'][33] + self.pose_data[counter]['keypoints'][36]) / 2,
                                (self.pose_data[counter]['keypoints'][34] + self.pose_data[counter]['keypoints'][37]) / 2]))
                )
                self.human_feet.append(np.array(np.array([(self.pose_data[counter]['keypoints'][45] + self.pose_data[counter]['keypoints'][48]) / 2,
                                (self.pose_data[counter]['keypoints'][46] + self.pose_data[counter]['keypoints'][49]) / 2]))
                )


                self.hip_angle_data.append(getAngle([(self.pose_data[counter]['keypoints'][36],self.pose_data[counter]['keypoints'][37]) ,(self.pose_data[counter]['keypoints'][18],self.pose_data[counter]['keypoints'][19]) ,(self.pose_data[counter]['keypoints'][42],self.pose_data[counter]['keypoints'][43])]))

                
                if check_handoff(self.horizontal_bar_points,[self.pose_data[counter]['keypoints'][27:29],self.pose_data[counter]['keypoints'][30:32]],self.human_mask_data[counter]): # check if handoff
                    self.handoff_data.append(1)  # handoff
                    counter +=1
                    self.twist_data.append(0)    # if human not on bar, then no speed
                else:
                    assert "sth wrong when process data"
                    self.handoff_data.append(0)  # not handoff
                    # first frame have no twist speed, so just record the next human center.
                    if len(pointlist[2]) == 0:
                        counter +=1
                        # add next human center to pointlist
                        pointlist[2] = np.array(np.array([(self.pose_data[counter]['keypoints'][33] + self.pose_data[counter]['keypoints'][36]) / 2,
                                        (self.pose_data[counter]['keypoints'][34] + self.pose_data[counter]['keypoints'][37]) / 2]))
                        continue
                    # pdb.set_trace()
                    # get angle.
                    delta_angle = getAngle(pointlist)
                    delta_theta_list.append(delta_angle)
                    
                    if len(delta_theta_list) == 2:
                        angular_speed = (delta_theta_list[0] + delta_theta_list[1] / 2) / delta_t
                        self.twist_data.append(angular_speed)
                        delta_theta_list.pop(0)
                        
                    if counter==self.num_frames - 1:
                        self.twist_data.append(0)
                    
                    counter +=1
                    pointlist[1] = pointlist[2]
                    # add next human center to pointlist
                    pointlist[2] = np.array(np.array([(self.pose_data[counter]['keypoints'][33] + self.pose_data[counter]['keypoints'][36]) / 2,
                                    (self.pose_data[counter]['keypoints'][34] + self.pose_data[counter]['keypoints'][37]) / 2]))
            except:
                video_frame.append(frame)

                assert "something wrong in data process"
                # break
        # pdb.set_trace()
        self.frames = np.stack(video_frame, axis=0)               # self.frames 儲存影片的所有幀

        self.position_label.setText('Position:0/{}'.format(self.num_frames))
        # bytesPerLine = 3 * self.w

        # set timeline slider
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(self.num_frames-1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TicksBelow)
        self.tl_slider.setTickInterval(1)
        self.playBtn.setEnabled(True)

        
        self.right_one_frame.setEnabled(True)
        self.left_one_frame.setEnabled(True)
        
        self.cursur = 0
        self.show_current_frame()


        # print('mae:',mae(self.gt_height_data,self.head_height_data))
        # pdb.set_trace()
        

        # gui timer 再讀完影片之後才啟動
        self.height_pg_timer.start() # 50ms
        # self.twist_pg_timer.start() # 50ms
        # self.hand_off_timer.start() # 50ms
        # self.imu_pg_timer.start() # 2ms  imu: 400Hz


    def update_interact_vis(self):
        frame = self.frames[self.cursur]
        if frame is not None:
            counter = 0
            # pdb.set_trace()
            if self.cursur < self.fps * 3:
                # for i in self.human_center_list:
                #     cv2.circle(frame,(int(i[0]) , int(i[1])), 2, (0,0,255), 2)
                #     counter += 1
                #     if counter >= self.cursur:
                #         break
                counter = 0
                for j in  self.head_coord_data:
                    cv2.circle(frame,(int(j[0]) , int(j[1])), 5, (255,0,0), 5)
                    counter += 1
                    if counter >= self.cursur:
                        break
            else:
                # pdb.set_trace()
                center_list = self.human_center_list[self.cursur-(int(self.fps) * 3) : self.cursur]
                feet_list =  self.head_coord_data[self.cursur-(int(self.fps) * 3) : self.cursur]
                # for i in center_list: 
                #     cv2.circle(frame,(int(i[0]) , int(i[1])), 2, (0,0,255), 2)
                for j in feet_list:
                    cv2.circle(frame,(int(j[0]) , int(j[1])), 5, (255,0,0), 5)
            # pdb.set_trace()
            cv2.line(frame, (math.ceil(self.head_coord_data[self.cursur][0]),math.ceil(self.head_coord_data[self.cursur][1])) , (math.ceil(self.head_coord_data[self.cursur][0]), self.horizontal_bar_points[0][1]), (0, 255, 255), thickness=3)
            cv2.line(frame, (0, self.horizontal_bar_points[0][1]), (self.w, self.horizontal_bar_points[0][1]), (0, 255, 0), thickness=1)
            # cv2.circle(frame,((self.horizontal_bar_points[0][0] + self.horizontal_bar_points[1][0])//2 , self.horizontal_bar_points[0][1]+5), handoff_radius, (255,0,0), 2)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            # self.setPixmap(QPixmap.fromImage(image))  #QLabel
            self.main_canvas.setPixmap(QPixmap(image.scaled(self.main_canvas.size(),    # canvas 的顯示
                Qt.KeepAspectRatio, Qt.FastTransformation)))
        # 播放的時候，也要更新slider cursur
        self.tl_slider.setValue(self.cursur)
    
    def tl_slide(self): # 只要slider(cursur)一改變，就要改變顯示的幀數，不管事正常播放或是拖拉
        self.cursur = self.tl_slider.value() #  改變顯示的幀數
        self.show_current_frame()            #  改變顯示的frame
        self.show_keypoint_position()        #  改變顯示的keypoint

    def show_current_frame(self):
        # Re-compute overlay and show the image
        # self.compose_current_im()
        t = time.time()
        self.update_interact_vis()
        
        self.position_label.setText('Position:{frame}/{all_frames}'.format(frame=int(self.cursur),all_frames=self.num_frames))
        self.tl_slider.setValue(self.cursur)
        print('update_vis and setText : ',time.time() - t)
    
    def show_keypoint_position(self):
        l_eye_x,l_eye_y = self.pose_data[self.cursur]['keypoints'][3:5]
        l_eye_x,l_eye_y = int(l_eye_x),int(l_eye_y)
        str_ = f"{l_eye_x},{l_eye_y}"
        self.l_eye_acc.setText(str_)
        r_eye_x,r_eye_y = self.pose_data[self.cursur]['keypoints'][6:8]
        r_eye_x,r_eye_y = int(r_eye_x),int(r_eye_y)
        str_ = f"{r_eye_x},{r_eye_y}"
        self.r_eye_acc.setText(str_)
        
        l_ear_x,l_ear_y = self.pose_data[self.cursur]['keypoints'][9:11]
        l_ear_x,l_ear_y = int(l_ear_x),int(l_ear_y)
        str_ = f"{l_ear_x},{l_ear_y}"
        self.l_ear_acc.setText(str_)
        r_ear_x,r_ear_y = self.pose_data[self.cursur]['keypoints'][12:14]
        r_ear_x,r_ear_y = int(r_ear_x),int(r_ear_y)
        str_ = f"{r_ear_x},{r_ear_y}"
        self.r_ear_acc.setText(str_)
        
        l_shoulder_x,l_shoulder_y = self.pose_data[self.cursur]['keypoints'][15:17]
        l_shoulder_x,l_shoulder_y = int(l_shoulder_x),int(l_shoulder_y)
        str_ = f"{l_shoulder_x},{l_shoulder_y}"
        self.l_shoulder_acc.setText(str_)
        r_shoulder_x,r_shoulder_y = self.pose_data[self.cursur]['keypoints'][18:20]
        r_shoulder_x,r_shoulder_y = int(r_shoulder_x),int(r_shoulder_y)
        str_ = f"{r_shoulder_x},{r_shoulder_y}"
        self.r_shoulder_acc.setText(str_)
        
        l_elbow_x,l_elbow_y = self.pose_data[self.cursur]['keypoints'][21:23]
        l_elbow_x,l_elbow_y = int(l_elbow_x),int(l_elbow_y)
        str_ = f"{l_elbow_x},{l_elbow_y}"
        self.l_elbow_acc.setText(str_)
        r_elbow_x,r_elbow_y = self.pose_data[self.cursur]['keypoints'][24:26]
        r_elbow_x,r_elbow_y = int(r_elbow_x),int(r_elbow_y)
        str_ = f"{r_elbow_x},{r_elbow_y}"
        self.r_elbow_acc.setText(str_)
        
        l_wrist_x,l_wrist_y = self.pose_data[self.cursur]['keypoints'][27:29]
        l_wrist_x,l_wrist_y = int(l_wrist_x),int(l_wrist_y)
        str_ = f"{l_wrist_x},{l_wrist_y}"
        self.l_wrist_acc.setText(str_)
        r_wrist_x,r_wrist_y = self.pose_data[self.cursur]['keypoints'][30:32]
        r_wrist_x,r_wrist_y = int(r_wrist_x),int(r_wrist_y)
        str_ = f"{r_wrist_x},{r_wrist_y}"
        self.r_wrist_acc.setText(str_)
        
        l_hip_x,l_hip_y = self.pose_data[self.cursur]['keypoints'][33:35]
        l_hip_x,l_hip_y = int(l_hip_x),int(l_hip_y)
        str_ = f"{l_hip_x},{l_hip_y}"
        self.l_hip_acc.setText(str_)
        r_hip_x,r_hip_y = self.pose_data[self.cursur]['keypoints'][36:38]
        r_hip_x,r_hip_y = int(r_hip_x),int(r_hip_y)
        str_ = f"{r_hip_x},{r_hip_y}"
        self.r_hip_acc.setText(str_)
        
        l_knee_x,l_knee_y = self.pose_data[self.cursur]['keypoints'][39:41]
        l_knee_x,l_knee_y = int(l_knee_x),int(l_knee_y)
        str_ = f"{l_knee_x},{l_knee_y}"
        self.l_knee_acc.setText(str_)
        r_knee_x,r_knee_y = self.pose_data[self.cursur]['keypoints'][42:44]
        r_knee_x,r_knee_y = int(r_knee_x),int(r_knee_y)
        str_ = f"{r_knee_x},{r_knee_y}"
        self.r_knee_acc.setText(str_)
        
        l_ankle_x,l_ankle_y = self.pose_data[self.cursur]['keypoints'][45:47]
        l_ankle_x,l_ankle_y = int(l_ankle_x),int(l_ankle_y)
        str_ = f"{l_ankle_x},{l_ankle_y}"
        self.l_ankle_acc.setText(str_)
        r_ankle_x,r_ankle_y = self.pose_data[self.cursur]['keypoints'][48:50]
        r_ankle_x,r_ankle_y = int(r_ankle_x),int(r_ankle_y)
        str_ = f"{r_ankle_x},{r_ankle_y}"
        self.r_ankle_acc.setText(str_)
    
    def on_play(self):                 
        if self.timer.isActive():
            self.timer.stop()
            self.playBtn.setText('Play')
        else:
            self.timer.start()   # timer 
            self.playBtn.setText('Pause')
            
    def on_time(self):                 # 更新 cursor
        self.cursur += 1
        if self.cursur > self.num_frames-1:
            self.cursur = 0
        self.tl_slider.setValue(self.cursur)
        
    def update_height(self):
        # pdb.set_trace()
        t = time.time()
        self.height_pg.clear()

        time_line = np.linspace(0.0, self.num_frames*(1/59.4), num=self.num_frames)
        self.height_pg.plot(y = self.head_height_data,  x = time_line)
        self.height_pg.plot([time_line[self.cursur]],[self.head_height_data[self.cursur]],pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')

        if self.gt_height_data is not None:
            time_line2 = np.linspace(0.0, len(self.gt_height_data)*(1/59.4), num=len(self.gt_height_data))
            self.height_pg.plot(self.gt_height_data,pen=(255,0,0), x = time_line2)
            self.height_pg.plot([time_line2[self.cursur]],[self.gt_height_data[self.cursur]],pen=(200,200,200), symbolBrush=(0,255,0), symbolPen='w')
            

        print('update_height:',time.time()-t)

    def load_mask(self):
        fnames = sorted(glob.glob(os.path.join(self.mask_path, '*.jpg')))
        if len(fnames) == 0:
            fnames = sorted(glob.glob(os.path.join(self.mask_path, '*.png')))
        frame_list = []
        # pdb.set_trace()
        t = time.time()
        for i, fname in enumerate(fnames):

            frame_list.append(np.array(Image.open(fname).convert('RGB'), dtype=np.uint8))

        print('reading mask cost', time.time() - t)
        
        self.mask_data = np.stack(frame_list, axis=0)
        
        # create human_mask 
        # mask_data有可能為圖片路徑，假如影片太長或解析度太高，我會將影片與mask都以路徑的方式讀取，而不是一次全都讀取到ram裡，
        # 除非 RAM 很大，否則讀取影片的方式就要將影片切割成圖片再一張一張讀取   

        t = time.time()
        for index,mask in enumerate(self.mask_data):
            self.human_mask_data[index].append(np.array(find_human_mask(mask)))
        print('find_human_mask cost', time.time() - t)
        # create horizontal mask only for 1st frame
        self.horizontal_bar_points = find_horizontal_bar_mask(self.mask_data[0])
        
        # pdb.set_trace()
        # return frames
        
    def on_prev_frame(self):
        # self.tl_slide will trigger on setValue
        self.cursur = max(0, self.cursur-1) # cursor 決定第幾幀
        self.tl_slider.setValue(self.cursur)

    def on_next_frame(self):  # 移至下一幀
        # self.tl_slide will trigger on setValue
        self.cursur = min(self.cursur+1, self.num_frames-1)
        self.tl_slider.setValue(self.cursur)

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.setWindowTitle('圖形化介面')
    Root.show()
    sys.exit(App.exec())# -*- coding: utf-8 -*-
