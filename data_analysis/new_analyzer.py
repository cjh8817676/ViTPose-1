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
from scipy.signal import find_peaks
from scipy import signal
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from pykalman import KalmanFilter


HUMAN_MASK = 1  # The number of the human mask
BAR_MASK   = 2  # The number of the bar mask

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
def find_human_mask(original_mask):
    # pdb.set_trace()
    h, w = original_mask.shape
    # human_mask = np.zeros((w,h,3)) # plt.imshow(human_mask) to verify if we only get human mask
    human_mask_point = []

    
    rows,cols = np.where((original_mask[:,:]==HUMAN_MASK))
    
    for i,j in zip(rows,cols):
        human_mask_point.append([j,i])
        # print(i,j)
    
    return human_mask_point

# mask of high bar is green
# @jit(nopython=True)
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

def getGradient(pt1,pt2):
    return (pt2[1]-pt1[1]) / (pt2[0]-pt1[0])
    
def getAngle(pointlist):
    """
    input a list which have 3 points
    two kinds of situation:
    1. calculate angle from Human_Com and center bar
        first point be axis
        clockwise => ang > 0
        counterclockwise => ang < 0
    """
    pt1,pt2,pt3 = pointlist
    m1 = getGradient(pt1, pt2)
    m2 = getGradient(pt1, pt3)
    angR =math.atan((m2-m1)/(1+(m2*m1)))
    angD = (math.degrees(angR))
    return -angD

def kalman_filter(data):
    # 定義觀測矩陣H，這是一個1x1的矩陣，因為觀測向量只有一維
    H = np.array([[1.0]])
    
    # 定義狀態轉移矩陣F，這也是一個1x1的矩陣，因為狀態向量只有一維
    F = np.array([[1.0]])
    
    # 定義狀態轉移噪聲協方差Q，這是一個1x1的矩陣
    Q = np.array([[1e-5]])
    
    # 定義觀測噪聲協方差R，這是一個1x1的矩陣
    R = np.array([[0.1]])
    
    # 初始化卡爾曼濾波器
    kf = KalmanFilter(transition_matrices=F, observation_matrices=H, transition_covariance=Q, observation_covariance=R)

    # 進行Kalman濾波
    filtered_data, _ = kf.filter(data)

    return filtered_data

def moving_average_filter(data, window_size):
    # 定義一個空的列表來存儲平均值
    filtered_data = []

    # 計算移動窗口中的平均值，並將其添加到filtered_data中
    for i in range(window_size, len(data) - window_size):
        window = data[i-window_size:i+window_size+1]
        avg = sum(window) / len(window)
        filtered_data.append(avg)

    return filtered_data

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

# 多個 Widget 組成 Layout。  Lauout 又可以互相組合。
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("./demo/new_gui.ui", self)
        
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
        # self.gt_height_curve = self.height_pg.plot(self.gt_height_data,pen=(255,0,0)) # 搭配 setData for indicate
        # self.height_curve = self.height_pg.plot(self.head_height_data) # 搭配 setData for indicate
        self.height_pg_timer = pg.QtCore.QTimer()
        self.height_pg_timer.timeout.connect(self.update_height)
    
        # twist pg
        self.gt_augular_speed_data = []
        self.twist_data = [0]                                          # no twist speed in first frame 
        self.twist_pg.showGrid(x = True, y = True, alpha = 1) 
        self.twist_pg.setLabel('bottom','Time','s')
        # self.twist_gt_curve1 = self.twist_pg.plot(self.gt_augular_speed_data,pen=(255,0,0)) # for indicate
        # self.twist_curve1 = self.twist_pg.plot(self.twist_data)                   # for indicate
        self.height_pg.setLabel('left','meter(m)')
        self.twist_pg_timer = pg.QtCore.QTimer()
        self.twist_pg_timer.timeout.connect(self.update_twist)
        
        # handoff pg
        self.handoff_data = defaultlist()    # true if hands on bars , false if hands not on bars for every frame.
        self.hand_off_pg.setLabel('bottom','Time','s')
        self.hand_off_pg.showGrid(x = True, y = True, alpha = 1)
        self.hand_off_curve = self.hand_off_pg.plot(self.handoff_data) # for indicate
        self.hand_off_timer = pg.QtCore.QTimer()
        self.hand_off_timer.timeout.connect(self.update_hand_off)
        
        self.comboBox.currentTextChanged.connect(self.set_RL_mode)
        self.body = 'Left_Body'

        # sensor pg
        # self.sensors_data =  np.random.rand(300)
        self.sensors_pg.setLabel('bottom','Time','s')
        self.sensors_pg.showGrid(x = True, y = True, alpha = 1)
        # self.sensors_curve = self.sensors_pg.plot(self.sensors_data)
        self.imu_pg_timer = pg.QtCore.QTimer()
        self.imu_pg_timer.timeout.connect(self.update_imu)
        self.l_eye       = []
        self.r_eye       = []
        self.l_ear       = []
        self.r_ear       = []
        self.l_shoulder  = []
        self.r_shoulder  = []
        self.l_elbow     = []
        self.r_elbow     = []
        self.l_wrist     = []
        self.r_wrist     = []
        self.l_hip       = []
        self.r_hip       = []
        self.l_knee      = []
        self.r_knee      = []
        self.l_ankle     = []
        self.r_ankle     = []
        
    def open_file(self):
        self.left_hip_angle_data = []
        self.left_shoulder_angle_data = []
        self.right_hip_angle_data = []
        self.right_shoulder_angle_data = []
        self.head_height_data = []
        self.hand_coordinate = []
        self.gt_augular_speed_data = []
        self.twist_data = [0]                                          # no twist speed in first frame 
        self.handoff_data = []
        self.head_coord_data = []
        global imu_data_pd,imu_data_gyrox,imu_data_gyroy,imu_data_gyroz
        global imu_data_accx,imu_data_accy,imu_data_accz
        global imu_data_haccx,imu_data_haccy,imu_data_haccz, imu_data_len
        global imu_data_left,imu_data_right
        imu_data_left = 0
        imu_data_right = 1200
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        self.file_path = os.path.dirname(self.filename)
        print('video_name:', self.filename)
        video_name = self.filename.split('x')[-1]
        video_form = video_name.split('.')[1]
        self.video_name = video_name.split('.')[0]
        video_mask = self.video_name.split('/')[-1]
        video_mask = video_mask.replace('vis_','')
        video_mask = video_mask.replace('ViTPose_','')
        self.mask_path = os.path.join('./workspace/',video_mask, "masks/") 
        # pdb.set_trace()
        if os.path.isfile(os.path.join(f'{self.file_path}','gt_' + video_mask+'.'+video_form+'.json')):
            self.motion_json_path = os.path.join(f'{self.file_path}','gt_' + video_mask+'.'+video_form+'.json')
        else:
            self.motion_json_path = os.path.join(f'{self.file_path}',video_mask+'.'+video_form+'.json')

        self.gt_head_path = os.path.join('/home/m11002125/ViTPose/ground_truth/',video_mask+'_head.xlsx')
        self.gt_belly_path = os.path.join('/home/m11002125/ViTPose/ground_truth/',video_mask+'_belly.xlsx')
        self.gt_joint_path = os.path.join('/home/m11002125/ViTPose/ground_truth/',video_mask+'_gt_joint.xlsx')
        self.gt_joint_angle_path = os.path.join('/home/m11002125/ViTPose/ground_truth/',video_mask+'_gt_jointangle.xlsx')
        
        # read results of pose estimation.
        with open(self.motion_json_path) as f:    # 讀取每一幀的 pose keypoint 和 bbox(左上、右下) 的座標
            self.pose_data = json.load(f)
        
        # pdb.set_trace()
        for i in self.pose_data:
            self.l_eye.append(i['keypoints'][3:5])
            self.r_eye.append(i['keypoints'][6:8])
            self.l_ear.append(i['keypoints'][9:11])
            self.r_ear.append(i['keypoints'][12:14])
            self.l_shoulder.append(i['keypoints'][15:17])
            self.r_shoulder.append(i['keypoints'][18:20])
            self.l_elbow.append(i['keypoints'][21:23])
            self.r_elbow.append(i['keypoints'][24:26])
            self.l_wrist.append(i['keypoints'][27:29])
            self.r_wrist.append(i['keypoints'][30:32])
            self.l_hip.append(i['keypoints'][33:35])
            self.r_hip.append(i['keypoints'][36:38])
            self.l_knee.append(i['keypoints'][39:41])
            self.r_knee.append(i['keypoints'][42:44])
            self.l_ankle.append(i['keypoints'][45:47])
            self.r_ankle.append(i['keypoints'][48:50])
        
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
            self.gt_augular_speed_data = np.array(belly_data['w'].to_numpy(), dtype=float) #　get angular speeds
        else:
            self.gt_augular_speed_data = None
            print('no belly ground truth')
            
        if os.path.isfile(self.gt_joint_angle_path):
            gt_jointangle_data = pd.read_excel(self.gt_joint_angle_path)
            self.gt_hip_angle_data = gt_jointangle_data['hip_angle'].to_numpy()
            self.gt_shoulder_angle_data = gt_jointangle_data['shoulder_angle']
            self.gt_angle_timeline = gt_jointangle_data.iloc[:,0:3]
            
            # # 設置Butterworth濾波器的參數
            # fs = 60  # 采樣率
            # fc = 15   # 截止頻率
            # order = 4  # 濾波器的阶数
            # # 設計低通Butterworth濾波器
            # b, a = signal.butter(order, fc / (fs / 2), 'low')
            # self.gt_hip_angle_data = signal.filtfilt(b, a, self.gt_hip_angle_data)
            # self.gt_shoulder_angle_data = signal.filtfilt(b, a, self.gt_shoulder_angle_data)
            
        else:
            self.gt_hip_angle_data = []
            self.gt_shoulder_angle_data = []
            print('no gt_jointangle.xlsx found')   
        if os.path.isfile(self.gt_joint_path):
            gt_joint_data = pd.read_excel(self.gt_joint_path)
            gt_joint_data.columns = gt_joint_data.iloc[0]
            gt_joint_data = gt_joint_data.drop(0)
            
            # record segment of ground truth
            self.gt_windows_step = []   # from video outputed from XMem
            self.gt_windows_frame = []  # from origin video 
            start = True
            for index,i in enumerate(gt_joint_data['step']):
                if start:
                    left_frame = gt_joint_data['frame'][index+1]
                    left = i
                    last = i
                    start = False
                    continue
                if i - last > 1:
                    right = last
                    self.gt_windows_step.append((left,right))
                    self.gt_windows_frame.append((left_frame,gt_joint_data['frame'][index]))
                    left_frame = gt_joint_data['frame'][index+1]
                    left = i
                if index == len(gt_joint_data['step'])-1:
                    right = i
                    self.gt_windows_step.append((left,right))
                    self.gt_windows_frame.append((left_frame,gt_joint_data['frame'][index+1]))
                last = i

            # 資料補None
            # Find out the timeline that complemented with None
            compliment = np.array([None] * 16)

            self.gt_joint_timeline = gt_joint_data.iloc[1:,0:3].to_numpy() # time,step,frame
            step_line = self.gt_joint_timeline[:,1]
            
            for j in self.gt_angle_timeline.iloc[0:].to_numpy():
                if j[1] not in step_line:
                    compliment[0] = j[0] 
                    compliment[1] = j[1]
                    compliment[2] = j[2]
                    gt_joint_data.loc[len(gt_joint_data)+1] = compliment
            
            gt_joint_data = gt_joint_data.sort_values(by='step').reset_index(drop=True)

            self.gt_head_coord = gt_joint_data.iloc[:,3:5].to_numpy()
            self.gt_height_data = gt_joint_data.iloc[:,5].to_numpy()
            self.gt_wrist_coord = gt_joint_data.iloc[:,6:8].to_numpy()
            self.gt_elbow_coord = gt_joint_data.iloc[:,8:10].to_numpy()
            self.gt_knee_coord = gt_joint_data.iloc[:,10:12].to_numpy()
            self.gt_hip_coord = gt_joint_data.iloc[:,12:14].to_numpy()
            self.gt_shoulder_coord = gt_joint_data.iloc[:,14:16].to_numpy()
        else:
            self.gt_height_data = None
            print('no gt_joint.xlsx found')   

        # read all video and save every frame in stack
        self.stream = cv2.VideoCapture(self.filename)                    # 影像路徑
        self.num_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
        self.fourcc = int(self.stream.get(cv2.CAP_PROP_FOURCC))          # fourcc:  編碼的種類 EX:(MPEG4 or H264)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)                     # 查看 FPS
        print('fps:',self.fps)
        print('num_of_frames:',self.num_frames)
        self.w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))          # 影片寬
        self.h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))         # 影片高

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
        # pdb.set_trace()
        # calculate motion parameter from every frame
        while(self.stream.isOpened()):
            _, frame = self.stream.read()
            if frame is None:
                break
            try:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # height of gymnast (這裡座標要改成由左下(左下)往右上)，因為高度是由下往上
                center_head_x = (self.l_ear[counter][0]+self.l_ear[counter][0]+self.l_eye[counter][0]+self.r_eye[counter][0]) / 4
                center_head_y = (self.l_ear[counter][1]+self.l_ear[counter][1]+self.l_eye[counter][1]+self.r_eye[counter][1]) / 4
                # pdb.set_trace()
                self.head_height_data.append(( center_head_y - center_bar[1] ) * pixel_height_ratio)

                self.hand_coordinate.append([self.l_wrist[counter],self.r_wrist[counter][30:32]]) # wrist_coordinate
                
                self.human_center_list.append(np.array(np.array([(self.l_hip[counter][0] + self.r_hip[counter][0]) / 2,
                                (self.l_hip[counter][1] + self.r_hip[counter][1]) / 2]))
                )
                self.human_feet.append(np.array(np.array([(self.r_ankle[counter][0] + self.l_ankle[counter][0]) / 2,
                                                            (self.r_ankle[counter][1] + self.l_ankle[counter][1]) / 2]))
                )
                # pdb.set_trace()
                self.left_hip_angle_data.append(calculate_angle_left_body([self.l_hip[counter] ,self.l_knee[counter] , self.l_shoulder[counter]]))
                self.left_shoulder_angle_data.append(calculate_angle_left_body([self.l_shoulder[counter], self.l_wrist[counter] , self.l_hip[counter]]))
                self.right_hip_angle_data.append(calculate_angle_right_body([self.r_hip[counter] , self.r_knee[counter], self.r_shoulder[counter]]))
                self.right_shoulder_angle_data.append(calculate_angle_right_body([self.r_shoulder[counter], self.r_wrist[counter] , self.r_hip[counter]]))
                
                
                if check_handoff(self.horizontal_bar_points,[self.l_wrist[counter],self.r_wrist[counter]],self.human_mask_data[counter]): # check if handoff
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
                        pointlist[2] = np.array(np.array([(self.l_hip[counter][0] + self.r_hip[counter][0]) / 2,
                                        (self.l_hip[counter][1] + self.r_hip[counter][1]) / 2]))
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
                    pointlist[2] = np.array(np.array([(self.l_hip[counter][0] + self.r_hip[counter][0]) / 2,
                                    (self.l_hip[counter][1] + self.r_hip[counter][1]) / 2]))
            except:

                print('sth error')
                assert "something wrong in data process"
                # break


        # median filter
        MVA_WINDOW_SIZE = 8
        self.head_height_data = median_filter(self.head_height_data,MVA_WINDOW_SIZE)
        self.right_hip_angle_data = median_filter(self.right_hip_angle_data,MVA_WINDOW_SIZE)  #data為要過濾的訊號
        self.left_shoulder_angle_data = median_filter(self.left_shoulder_angle_data,MVA_WINDOW_SIZE)  #data為要過濾的訊號
        self.left_hip_angle_data = median_filter(self.left_hip_angle_data,MVA_WINDOW_SIZE)  #data為要過濾的訊號
        self.right_shoulder_angle_data = median_filter(self.right_shoulder_angle_data,MVA_WINDOW_SIZE)  #data為要過濾的訊號

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

        # gui timer 再讀完影片之後才啟動
        self.height_pg_timer.start() # 50ms
        self.twist_pg_timer.start() # 50ms
        self.hand_off_timer.start() # 50ms
        self.imu_pg_timer.start() # 2ms  imu: 400Hz
    
    def set_RL_mode(self):
        self.body = self.comboBox.currentText()
        print(self.body)

    def update_interact_vis(self):
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, self.cursur)
        # frame = self.frames[self.cursur]
        if self.stream.isOpened():

            ret, frame = self.stream.read()
            counter = 0
            if ret:
                # pdb.set_trace()
                # if self.cursur < self.fps * 3:
                #     for i in self.human_center_list:
                #         cv2.circle(frame,(int(i[0]) , int(i[1])), 2, (0,0,255), 2)
                #         counter += 1
                #         if counter >= self.cursur:
                #             break
                #     counter = 0
                #     for j in self.human_feet:
                #         cv2.circle(frame,(int(j[0]) , int(j[1])), 2, (255,0,0), 2)
                #         counter += 1
                #         if counter >= self.cursur:
                #             break
                # else:
                    # pdb.set_trace()
                    # center_list = self.human_center_list[self.cursur-(int(self.fps) * 3) : self.cursur]
                    # feet_list = self.human_feet[self.cursur-(int(self.fps) * 3) : self.cursur]
                    # for i in center_list: 
                        # cv2.circle(frame,(int(i[0]) , int(i[1])), 2, (0,0,255), 2)
                    # for j in feet_list:
                        # cv2.circle(frame,(int(j[0]) , int(j[1])), 2, (255,0,0), 2)

                # cv2.line(frame, (0, self.horizontal_bar_points[0][1]), (self.w, self.horizontal_bar_points[0][1]), (0, 255, 0), thickness=1)
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
        self.update_interact_vis()
        self.position_label.setText('Position:{frame}/{all_frames}'.format(frame=int(self.cursur),all_frames=self.num_frames))
        self.tl_slider.setValue(self.cursur)
    
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
        else:
            self.timer.start()   # timer 
            
    def on_time(self):                 # 更新 cursor
        self.cursur += 1
        if self.cursur > self.num_frames-1:
            self.cursur = 0
        self.tl_slider.setValue(self.cursur)
        
    def update_height(self):
        # pdb.set_trace()
        self.height_pg.clear()
        time = np.linspace(0.0, self.num_frames/self.fps, num=len(self.head_height_data))
        self.height_pg.plot(y = self.head_height_data,  x = time)
        self.height_pg.plot([time[self.cursur]],[self.head_height_data[self.cursur]],pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')

        lr = pg.LinearRegionItem([1, 30], bounds=[0,100], movable=True)
        self.height_pg.addItem(lr)
        line = pg.InfiniteLine(angle=90, movable=True)
        self.height_pg.addItem(line)
        line.setBounds([0,200])

        if self.gt_height_data is not None:
            data = np.array(self.gt_height_data, dtype=float)
            time2 = np.linspace(0.0, len(self.gt_height_data)/self.fps, num=len(self.gt_height_data))
            
            self.height_pg.plot(data,pen=(255,0,0), x = time2)
            self.height_pg.plot([time2[self.cursur]],[data[self.cursur]],pen=(200,200,200), symbolBrush=(0,255,0), symbolPen='w')

    def update_twist(self):
        # pdb.set_trace()

        self.twist_pg.clear()
        if self.body == 'Left_Body':
            self.twist_pg.plot(self.left_shoulder_angle_data)
            self.twist_pg.plot([self.cursur],[self.left_shoulder_angle_data[self.cursur]],pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')
        elif self.body == 'Right_Body':
            self.twist_pg.plot(self.right_shoulder_angle_data)
            self.twist_pg.plot([self.cursur],[self.right_shoulder_angle_data[self.cursur]],pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')
        
        if len(self.gt_shoulder_angle_data) > 0:
            self.twist_pg.plot(self.gt_shoulder_angle_data,pen=(255,0,0))
            self.twist_pg.plot([self.cursur],[self.gt_shoulder_angle_data[self.cursur]],pen=(200,200,200), symbolBrush=(0,255,0), symbolPen='w')

        
    def update_hand_off(self):
        data = self.handoff_data[self.cursur:self.cursur+int(self.num_frames/self.fps)]
        self.hand_off_curve.setData(data)
        self.hand_off_curve.setPos(self.cursur, 0)
        

    def update_imu(self):
        # pdb.set_trace()
        self.sensors_pg.clear()
        if self.body == 'Left_Body':
            self.sensors_pg.plot(self.right_hip_angle_data)
            self.sensors_pg.plot([self.cursur],[self.right_hip_angle_data[self.cursur]],pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')
        elif self.body == 'Right_Body':
            self.sensors_pg.plot(self.left_hip_angle_data)
            self.sensors_pg.plot([self.cursur],[self.left_hip_angle_data[self.cursur]],pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')
        
        if len(self.gt_hip_angle_data) > 0:
            self.sensors_pg.plot(self.gt_hip_angle_data,pen=(255,0,0))
            self.sensors_pg.plot([self.cursur],[self.gt_hip_angle_data[self.cursur]],pen=(200,200,200), symbolBrush=(0,255,0), symbolPen='w')
        
    

    def rand(self,n):
        data = np.random.random(n)
        data[int(n*0.1):int(n*0.13)] += .5
        data[int(n*0.18)] += 2
        data[int(n*0.1):int(n*0.13)] *= 5
        data[int(n*0.18)] *= 20
        data *= 1e-12
        return data, np.arange(n, n+len(data)) / float(n)
    
    def load_mask(self):
        fnames = sorted(glob.glob(os.path.join(self.mask_path, '*.jpg')))
        if len(fnames) == 0:
            fnames = sorted(glob.glob(os.path.join(self.mask_path, '*.png')))
        frame_list = []
        # pdb.set_trace()
        t = time.time()
        for i, fname in enumerate(fnames):
            frame_list.append(np.array(Image.open(fname), dtype=np.uint8))
        print('reading mask cost', time.time() - t)
        
        self.mask_data = np.stack(frame_list, axis=0)
        
        # create human_mask 
        # mask_data有可能為圖片路徑，假如影片太長或解析度太高，我會將影片與mask都以路徑的方式讀取，而不是一次全都讀取到ram裡，
        # 除非 RAM 很大，否則讀取影片的方式就要將影片切割成圖片再一張一張讀取   
        # pdb.set_trace()
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

