import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class ZoomableLabel(QLabel):
    def __init__(self, scroll_area ,parent=None):
        super(ZoomableLabel, self).__init__(parent)
        self.scroll_area = scroll_area  # 保存对QScrollArea对象的引用
        self.scale_factor = 1.0
        self.original_pixmap = None
        self.is_panning = False
        self.last_mouse_pos = QPoint()

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        if self.original_pixmap:
            # 使用保存的縮放比例來縮放新的帧
            width = self.scale_factor * self.original_pixmap.width()
            height = self.scale_factor * self.original_pixmap.height()
            scaled_pixmap = self.original_pixmap.scaled(width, height, Qt.KeepAspectRatio)
            # 设定缩放后的Pixmap给QLabel
            super().setPixmap(scaled_pixmap)
            
            # Maintain the scroll position
            self.scroll_area.horizontalScrollBar().setValue(self.scroll_x)
            self.scroll_area.verticalScrollBar().setValue(self.scroll_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.unsetCursor()

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
                scaled_pixmap = self.original_pixmap.scaled(width, height, Qt.KeepAspectRatio)
                super().setPixmap(scaled_pixmap)

            # Update the scroll bars
            self.scroll_area.horizontalScrollBar().setValue(self.scroll_area.horizontalScrollBar().value() + dx)
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().value() + dy)

    def mouseMoveEvent(self, event):
        if self.is_panning:
            diff = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self.scroll_area.horizontalScrollBar().setValue(self.scroll_area.horizontalScrollBar().value() - diff.x())
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().value() - diff.y())

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
        
        if fname:
            self.cap = cv2.VideoCapture(fname)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setMaximum(self.total_frames - 1)

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
            if ret:
                # Record the scroll position
                self.video_label.scroll_x = self.video_label.scroll_area.horizontalScrollBar().value()
                self.video_label.scroll_y = self.video_label.scroll_area.verticalScrollBar().value()
        
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap)  # 直接设置新的pixmap，缩放将在setPixmap方法中处理
                
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

