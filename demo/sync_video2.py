import os
import cv2
import sys
# fix conflicts between qt5 and cv2
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from GUI.video_time_couplegui import Ui_MainWindow
import numpy as np
import tqdm

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.Upload_video.clicked.connect(self.open_file)
        self.ui.Upload_video2.clicked.connect(self.open_file2)

        # timeline slider
        self.ui.tl_slider.valueChanged.connect(self.tl_slide)  # 只要被拖拉，或是被set_value就會執行self.tl_slide
        self.ui.tl_slider.setTickPosition(QSlider.TicksBelow)
        
        # timeline slider
        self.ui.tl_slider2.valueChanged.connect(self.tl_slide2)  # 只要被拖拉，或是被set_value就會執行self.tl_slide
        self.ui.tl_slider2.setTickPosition(QSlider.TicksBelow)
        

        # some buttons
        self.ui.playBtn.setEnabled(False)
        self.ui.playBtn.clicked.connect(self.on_play)

        # some buttons
        self.ui.playBtn2.setEnabled(False)
        self.ui.playBtn2.clicked.connect(self.on_play2)

        # some buttons
        self.ui.doubleplay.setEnabled(False)
        self.ui.doubleplay.clicked.connect(self.on_double_play)

        # some buttons
        self.ui.couple_save.setEnabled(False)
        self.ui.couple_save.clicked.connect(self.save_couple_video)

        # right left button
        self.ui.right_one_frame.setEnabled(False)
        self.ui.right_one_frame.clicked.connect(self.on_next_frame)
        self.ui.left_one_frame.setEnabled(False)
        self.ui.left_one_frame.clicked.connect(self.on_prev_frame)

        self.ui.right_one_frame2.setEnabled(False)
        self.ui.right_one_frame2.clicked.connect(self.on_next_frame2)
        self.ui.left_one_frame2.setEnabled(False)
        self.ui.left_one_frame2.clicked.connect(self.on_prev_frame2)


        # cursur timer
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.on_time)

        # cursur timer
        self.timer2 = QTimer()
        self.timer2.setSingleShot(False)
        self.timer2.timeout.connect(self.on_time2)

        # double cursur timer
        self.doubletimer = QTimer()
        self.doubletimer.setSingleShot(False)
        self.doubletimer.timeout.connect(self.on_time_double)




    def open_file(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        print('video_name1:', self.filename)
        stream = cv2.VideoCapture(self.filename)                    # 影像路徑
        self.num_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))          # fourcc:  編碼的種類 EX:(MPEG4 or H264)
        self.fps = stream.get(cv2.CAP_PROP_FPS)                     # 查看 FPS
        print('fps:',self.fps)
        print('num_of_frames:',self.num_frames)
        self.w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))          # 影片寬
        self.h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))         # 影片高

        video_frame = []
        while(stream.isOpened()):
            _, frame = stream.read()
            if frame is None:
                break
            video_frame.append(frame)

        self.frames = np.array(video_frame)   

        self.ui.tl_slider.setMinimum(0)
        self.ui.tl_slider.setMaximum(self.num_frames-1)
        self.ui.tl_slider.setValue(0)
        self.ui.tl_slider.setTickPosition(QSlider.TicksBelow)
        self.ui.tl_slider.setTickInterval(1)
        self.ui.playBtn.setEnabled(True)
        self.ui.right_one_frame.setEnabled(True)
        self.ui.left_one_frame.setEnabled(True)
        self.cursur = 0
        self.show_current_frame()
        self.ui.playBtn.setEnabled(True)


    def open_file2(self):
        self.filename2, _ = QFileDialog.getOpenFileName(self, "Open Video")
        print('video_name2:', self.filename2)
        stream = cv2.VideoCapture(self.filename2)                    # 影像路徑
        self.num_frames2 = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
        self.fourcc2 = int(stream.get(cv2.CAP_PROP_FOURCC))          # fourcc:  編碼的種類 EX:(MPEG4 or H264)
        self.fps2 = stream.get(cv2.CAP_PROP_FPS)                     # 查看 FPS
        print('fps2:',self.fps2)
        print('num_of_frames2:',self.num_frames2)
        self.w_2 = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))          # 影片寬
        self.h_2 = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))         # 影片高

        video_frame = []
        while(stream.isOpened()):
            _, frame = stream.read()
            if frame is None:
                break
            video_frame.append(frame)

        self.frames2 = np.array(video_frame)   
        self.ui.tl_slider2.setMinimum(0)
        self.ui.tl_slider2.setMaximum(self.num_frames2-1)
        self.ui.tl_slider2.setValue(0)
        self.ui.tl_slider2.setTickPosition(QSlider.TicksBelow)
        self.ui.tl_slider2.setTickInterval(1)
        self.ui.playBtn.setEnabled(True)
        self.ui.right_one_frame2.setEnabled(True)
        self.ui.left_one_frame2.setEnabled(True)
        self.cursur2 = 0
        self.show_current_frame2()
        self.ui.playBtn2.setEnabled(True)
        self.ui.doubleplay.setEnabled(True)
        self.ui.couple_save.setEnabled(True)

    def tl_slide(self): # 只要slider(cursur)一改變，就要改變顯示的幀數，不管事正常播放或是拖拉
        self.cursur = self.ui.tl_slider.value() #  改變顯示的幀數
        self.show_current_frame()            #  改變顯示的frame


    def tl_slide2(self): # 只要slider(cursur)一改變，就要改變顯示的幀數，不管事正常播放或是拖拉
        self.cursur2 = self.ui.tl_slider2.value() #  改變顯示的幀數
        self.show_current_frame2()            #  改變顯示的frame

    
    def show_current_frame(self):
        # Re-compute overlay and show the image
        # self.compose_current_im()
        self.update_interact_vis()
        self.ui.position_label.setText('Position:{frame}/{all_frames}'.format(frame=int(self.cursur),all_frames=self.num_frames))
        self.ui.tl_slider.setValue(self.cursur)

    def show_current_frame2(self):
        # Re-compute overlay and show the image
        # self.compose_current_im()
        self.update_interact_vis2()
        self.ui.position_label2.setText('Position:{frame}/{all_frames}'.format(frame=int(self.cursur2),all_frames=self.num_frames2))
        self.ui.tl_slider2.setValue(self.cursur2)

    
    def update_interact_vis(self):
        frame = self.frames[self.cursur]
        if frame is not None:
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            # self.setPixmap(QPixmap.fromImage(image))  #QLabel
            self.ui.main_canvas.setPixmap(QPixmap(image.scaled(self.ui.main_canvas.size(),    # canvas 的顯示
                Qt.KeepAspectRatio, Qt.FastTransformation)))
        # 播放的時候，也要更新slider cursur
        self.ui.tl_slider.setValue(self.cursur)
        
    def update_interact_vis2(self):
        frame = self.frames2[self.cursur2]
        if frame is not None:
            # pdb.set_trace()
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            # self.setPixmap(QPixmap.fromImage(image))  #QLabel
            self.ui.main_canvas2.setPixmap(QPixmap(image.scaled(self.ui.main_canvas2.size(),    # canvas 的顯示
                Qt.KeepAspectRatio, Qt.FastTransformation)))
        # 播放的時候，也要更新slider cursur
        self.ui.tl_slider2.setValue(self.cursur2)

    def on_double_play(self):                 
        if self.doubletimer.isActive():
            self.doubletimer.stop()
        else:
            self.doubletimer.start()   # timer 

    def on_play(self):                 
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start()   # timer 

    def on_play2(self):                 
        if self.timer2.isActive():
            self.timer2.stop()
        else:
            self.timer2.start()   # timer 
    
                
    def on_time(self):                 # 更新 cursor
        self.cursur += 1
        if self.cursur > self.num_frames-1:
            self.cursur = 0
        self.ui.tl_slider.setValue(self.cursur)
                    
    def on_time2(self):                 # 更新 cursor
        self.cursur2 += 1
        if self.cursur2 > self.num_frames2-1:
            self.cursur2 = 0
        self.ui.tl_slider2.setValue(self.cursur2)

    def on_time_double(self):           # 更新 2種cursor
        self.on_time()
        self.on_time2()


    def on_prev_frame(self):
        # self.tl_slide will trigger on setValue
        self.cursur = max(0, self.cursur-1) # cursor 決定第幾幀
        self.ui.tl_slider.setValue(self.cursur)

    def on_next_frame(self):  # 移至下一幀
        # self.tl_slide will trigger on setValue
        self.cursur = min(self.cursur+1, self.num_frames-1)
        self.ui.tl_slider.setValue(self.cursur)

    def on_prev_frame2(self):
        # self.tl_slide will trigger on setValue
        self.cursur2 = max(0, self.cursur2-1) # cursor 決定第幾幀
        self.ui.tl_slider2.setValue(self.cursur2)

    def on_next_frame2(self):  # 移至下一幀
        # self.tl_slide will trigger on setValue
        self.cursur2 = min(self.cursur2+1, self.num_frames2-1)
        self.ui.tl_slider2.setValue(self.cursur2)
    
    def save_couple_video(self):
        cursur1 = self.cursur
        cursur2 = self.cursur2
        numframes = self.num_frames
        numframes2 = self.num_frames2

        right1 = numframes - cursur1
        right2 = numframes2 - cursur2

        stream = cv2.VideoCapture(self.filename)                      # 影像路徑
        stream2 = cv2.VideoCapture(self.filename2)                    # 影像路徑


        # 創建一個視頻寫入器，將同步的影片寫入新的文件中
        videoWriter = cv2.VideoWriter('synced_video1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        videoWriter2 = cv2.VideoWriter('synced_video2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.fps2, (int(stream2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream2.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        counter = 0
        counter2 = 0

        while True:
            # 從兩個影片中讀取當前幀
            if counter != cursur1:
                ret1, frame1 = stream.read()
                counter += 1
            if counter2 != cursur2:
                ret2, frame2 = stream2.read()
                counter2 += 1 

            if counter == cursur1 and counter2 == cursur2:
                break


        while True:

            ret1, frame1 = stream.read()
            ret2, frame2 = stream2.read()
            # 如果其中一個影片已經讀取到結束，則退出循環
            if not ret1 or not ret2:
                break
            videoWriter.write(frame1)
            videoWriter2.write(frame2)


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.setWindowTitle('圖形化介面')
    Root.show()
    sys.exit(App.exec())# -*- coding: utf-8 -*-
