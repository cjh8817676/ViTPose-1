# -*- coding: utf-8 -*-
import math
import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import os
import json
import sys
import pdb
import mmcv
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.color import color_val


class ZoomableLabel(QLabel):
    def __init__(self, scroll_area ,parent=None):
        super(ZoomableLabel, self).__init__(parent)
        self.scroll_area = scroll_area  # 保存对QScrollArea对象的引用
        self.scale_factor = 1.0
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.is_panning = False
        self.last_mouse_pos = QPoint()
        self.scroll_x = 0
        self.scroll_y = 0

        self.all_keypoints = np.array([[]]) # all keypoints (type is numpy)
        self.skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
        self.points = []            # record all keypoints (type is QPoint)
        self.dragging_index = None  # selected_keypoint 
        self.setMouseTracking(True)
    
    def set_skeleton(self, keypoints):
        self.all_keypoints = np.array([keypoints])
        # pdb.set_trace()
        self.update()  # 要求重繪，plot_skeleton()
        
    def setPixmap(self, pixmap):
        
        self.scroll_x = self.scroll_area.horizontalScrollBar().value()
        self.scroll_y = self.scroll_area.verticalScrollBar().value()
        
        self.original_pixmap = pixmap
        if self.original_pixmap:
            # 使用保存的縮放比例來縮放新的帧
            width = self.scale_factor * self.original_pixmap.width()
            height = self.scale_factor * self.original_pixmap.height()
            self.scaled_pixmap = self.original_pixmap.scaled(width, height, Qt.KeepAspectRatio)
            
            painter = QPainter(self.original_pixmap)
            painter.setPen(QPen(Qt.red, 10))  # 设置笔的颜色和大小
            painter.drawPoint(0, 0)  # 在左上角绘制点
            
            # 设定缩放后的Pixmap给QLabel
            super().setPixmap(self.scaled_pixmap)
            
            # Maintain the scroll position
            self.scroll_area.horizontalScrollBar().setValue(self.scroll_x)
            self.scroll_area.verticalScrollBar().setValue(self.scroll_y)

    def mousePressEvent(self, event):
        print( event.pos())
        
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            print('click:',event.pos())
            print('points:',self.points)
            for i, point in enumerate(self.points):
                print('click:',event.pos())
                if (event.pos() - point).manhattanLength() < 1:
                    self.dragging_index = i   # 找出第幾點被拖拉
                    break

    def mouseReleaseEvent(self, event):
        self.is_panning = False
        self.unsetCursor()
        self.dragging_index = None
        
    def wheelEvent(self, event):
        degrees = event.angleDelta().y() / 8  # get wheel rotation in degrees
        steps = degrees / 15  # each step is 15 degrees
        proposed_scale_factor = self.scale_factor * (1.0 + steps / 10)
    
        # Constrain the zoom level to reasonable values
        if 0.1 <= proposed_scale_factor <= 10.0:
            old_scale_factor = self.scale_factor
            self.scale_factor = proposed_scale_factor
    
            # Calculate the new position of the scroll bars
            old_pos = self.scroll_area.mapFromGlobal(QCursor.pos())
            dx = old_pos.x() * (self.scale_factor / old_scale_factor - 1)
            dy = old_pos.y() * (self.scale_factor / old_scale_factor - 1)
    
            # Apply the zoom level to the image
            if self.original_pixmap:
                width = self.scale_factor * self.original_pixmap.width()
                height = self.scale_factor * self.original_pixmap.height()
                self.scaled_pixmap = self.original_pixmap.scaled(width, height, Qt.KeepAspectRatio)
                super().setPixmap(self.scaled_pixmap)
    
            # Update the scroll bars
            self.scroll_area.horizontalScrollBar().setValue(self.scroll_area.horizontalScrollBar().value() + dx)
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().value() + dy)

    def mouseMoveEvent(self, event):
        if self.is_panning and event.buttons() & Qt.MiddleButton:
            diff = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self.scroll_area.horizontalScrollBar().setValue(self.scroll_area.horizontalScrollBar().value() - diff.x())
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().value() - diff.y())
        elif event.buttons() & Qt.LeftButton and self.dragging_index is not None:
            # Calculate the new position of the keypoint in the original coordinate system
            pixmap_pos = self.get_scaled_pixmap_position()  # Get the position of the top left corner of the pixmap
            new_x = (event.x() - pixmap_pos.x()) / self.scale_factor
            new_y = (event.y() - pixmap_pos.y()) / self.scale_factor
    
            # Update the keypoint and the corresponding point
            self.all_keypoints[0][self.dragging_index] = [new_x, new_y]
            self.points[self.dragging_index] = event.pos()
    
            self.update()  # Redraw the image and the skeleton
            
    def get_scaled_pixmap_position(self):
        # 獲取 ZoomableLabel 和 scaled_pixmap 的中心點
        label_center = self.rect().center()
        pixmap_center = self.scaled_pixmap.rect().center()

        # 計算 scaled_pixmap 的左上角座標，使其中心與 ZoomableLabel 的中心對齊
        top_left = label_center - pixmap_center
        print('top_left',top_left)

        return top_left
    
    def paintEvent(self, e):
        print('paint event')
        # Call the parent class's paintEvent to ensure the QLabel is painted correctly
        super().paintEvent(e)
        if self.scaled_pixmap is not None:
            painter = QPainter(self)
            self.plot_skeleton(painter)
            painter.end()
            
    def plot_skeleton(self, painter):
        print('plot skeleton')
        if self.all_keypoints.shape[0] != 0:
            self.points = []
            pixmap_pos = self.get_scaled_pixmap_position()  # Get the position of the top left corner of the pixmap
            for i, keypoint in enumerate(self.all_keypoints[0]):
                # Scale the keypoint's coordinates according to the current scaling factor
                scaled_x = int(keypoint[0] * self.scale_factor) + pixmap_pos.x()
                scaled_y = int(keypoint[1] * self.scale_factor) + pixmap_pos.y()
                point = QPoint(scaled_x, scaled_y)
                self.points.append(point)
                painter.setBrush(QBrush(Qt.green))
                painter.drawEllipse(point, 3, 3)
            for link in self.skeleton:
                painter.setPen(QPen(Qt.red, 2))
                painter.drawLine(self.points[link[0]], self.points[link[1]])
                
    def zoom_in(self):
        self.scale_factor *= 1.2
        self.rescale_image()

    def zoom_out(self):
        self.scale_factor /= 1.2
        self.rescale_image()
    
    def rescale_image(self):
        if self.original_pixmap is not None:
            self.scaled_pixmap = self.original_pixmap.scaled(self.size() * self.scale_factor, Qt.KeepAspectRatio)
            self.setPixmap(self.scaled_pixmap)
            self.update()
                        
            
class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player")
        # UI elements
        self.scroll_area = QScrollArea(self)

        self.video_label = ZoomableLabel(self.scroll_area)  # use our custom label
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.scroll_area.setWidget(self.video_label)
        self.scroll_area.setWidgetResizable(True)  # allow the widget to resize with the scroll area

        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)  # Connect sliderMoved signal to set_position method
        self.slider.sliderPressed.connect(self.slider_pressed)  # Connect sliderPressed signal to slider_pressed method
        self.slider.sliderReleased.connect(self.slider_released)  # Connect sliderReleased signal to slider_released method
        self.frame_label = QLabel("Frame: 0")
        
        self.skeletons = {}          # 記錄所有骨架

        # Open file button
        self.open_button = QPushButton("Open File")
        self.open_button.clicked.connect(self.open_file)

        # Play button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_pause)

        # Layout for controls
        self.controls_layout = QHBoxLayout()
        self.controls_layout.addWidget(self.open_button)
        self.controls_layout.addWidget(self.play_button)
        self.controls_layout.addWidget(self.slider)
        self.controls_layout.addWidget(self.frame_label)

        # Main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.scroll_area)
        self.main_layout.addLayout(self.controls_layout)

        # Container
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        # Timer for video frame fetching
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)

        # Frame control
        QShortcut(QKeySequence("F"), self, self.next_frame)
        QShortcut(QKeySequence("D"), self, self.prev_frame)
        QShortcut(QKeySequence("R"), self, self.reset_zoom) # Reset zoom shortcut

        self.cap = None
        self.total_frames = 0
        self.isPlaying = False
        self.isSliderPressed = False  # Add a variable to track if the slider is being dragged

    def scroll(self, dx, dy):
        scroll_bar_x = self.scroll_area.horizontalScrollBar()
        scroll_bar_y = self.scroll_area.verticalScrollBar()

        if scroll_bar_x:
            scroll_bar_x.setValue(max(min(scroll_bar_x.value() + dx, scroll_bar_x.maximum()), scroll_bar_x.minimum()))

        if scroll_bar_y:
            scroll_bar_y.setValue(max(min(scroll_bar_y.value() + dy, scroll_bar_y.maximum()), scroll_bar_y.minimum()))

    def open_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', 
            '/home/m11002125/ViTPose/vis_results',"Video files (*.mp4 *.avi *.MOV)")
        video_name = fname.split('/')[-1]
        video_name = video_name.replace('vis_','')
        if sys.platform == "linux":
            data_path = fname.split('/')[1:-1]
            self.video_name = video_name
            self.motion_json_path = ''
            for i in data_path:
                self.motion_json_path = os.path.join(self.motion_json_path,i)
            self.motion_json_path = os.path.join(self.motion_json_path,self.video_name + '.json')
        elif sys.platform == "win32":
            data_path = fname.split('/')[1:-1]
            self.video_name = video_name
            self.motion_json_path = 'c:\\'
            for i in data_path:
                self.motion_json_path = os.path.join(self.motion_json_path,i)
            self.motion_json_path = os.path.join(self.motion_json_path,self.video_name + '.json')

        
        if fname:
            self.cap = cv2.VideoCapture(fname)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setMaximum(self.total_frames - 1)
            with open(self.motion_json_path) as f:    # 讀取每一幀的 pose keypoint 和 bbox(左上、右下) 的座標
                self.pose_data = json.load(f)
            
        self.keypoints = {}
        self.pose_result = []
        # pdb.set_trace()  
        for i in self.pose_data:
            bodys_coord = []
            bodys_coord_score = []
            for j in range(0,len(i['keypoints']),3):
                bodys_coord.append((i['keypoints'][j],i['keypoints'][j+1]))
                bodys_coord_score.append([i['keypoints'][j],i['keypoints'][j+1]])
                
            self.pose_result.append(np.array(bodys_coord_score))
            self.keypoints[i['image_id']] = bodys_coord
        self.pose_result = np.array(self.pose_result)
        # pdb.set_trace()   
        print('finish')
        
    def play_pause(self):
        if self.isPlaying:
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(30)
            self.play_button.setText("Pause")
        self.isPlaying = not self.isPlaying

    def play_video(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            d_skeleton = None
            if ret:
                # pdb.set_trace()
                self.video_label.scroll_x = self.video_label.scroll_area.horizontalScrollBar().value()
                self.video_label.scroll_y = self.video_label.scroll_area.verticalScrollBar().value()
                
                d_skeleton =  self.pose_result[self.slider.value()]
                
                self.skeletons[str(self.slider.value())] = d_skeleton
                
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                # pdb.set_trace()
                self.video_label.setPixmap(pixmap)  # 直接设置新的pixmap，缩放将在setPixmap方法中处理。
                self.video_label.set_skeleton(d_skeleton)
                # Maintain the scroll position
                self.video_label.scroll_area.horizontalScrollBar().setValue(self.video_label.scroll_x)
                self.video_label.scroll_area.verticalScrollBar().setValue(self.video_label.scroll_y)
        
                if not self.isSliderPressed:  # Only update slider value when it's not being dragged
                    self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    
                self.frame_label.setText(f"Frame: {self.cap.get(cv2.CAP_PROP_POS_FRAMES)}")     

    def slider_pressed(self):
        self.isSliderPressed = True

    def slider_released(self):
        self.isSliderPressed = False
        self.set_position(self.slider.value())  # Ensure the video frame matches the slider position when released

    def set_position(self, position):
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.play_video()

    def next_frame(self):
        if self.cap.isOpened():
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame + 1 < self.total_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 1 -1 )
                self.play_video()

    def prev_frame(self):
        if self.cap.isOpened():
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame - 1 >= 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1 -1)
                self.play_video()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
            
    def reset_zoom(self):
        self.video_label.scale_factor = 1.0  # Reset the scale factor to 1.0
        self.video_label.setPixmap(self.video_label.original_pixmap)  # Apply the original pixmap without scaling
        
if __name__ == '__main__':
    app = QApplication([])
    player = VideoPlayer()
    player.show()
    app.exec_()

