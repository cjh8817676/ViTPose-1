import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, QHBoxLayout, QMainWindow, QScrollArea
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import pdb

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()

        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()

        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        self.frame_label = QLabel()
        self.layout.addWidget(self.frame_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)
        self.layout.addWidget(self.slider)

        self.open_file_button = QPushButton('Open File')
        self.open_file_button.clicked.connect(self.open_file_dialog)
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.play_pause)
        self.next_frame_button = QPushButton('Next Frame')
        self.next_frame_button.clicked.connect(self.next_frame)
        self.last_frame_button = QPushButton('Last Frame')
        self.last_frame_button.clicked.connect(self.last_frame)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.open_file_button)
        self.buttons_layout.addWidget(self.play_button)
        self.buttons_layout.addWidget(self.next_frame_button)
        self.buttons_layout.addWidget(self.last_frame_button)
        self.layout.addLayout(self.buttons_layout)

        self.setLayout(self.layout)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filepath, _ = QFileDialog.getOpenFileName(self, 'Open Video File', '', 'Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)', options=options)
        if filepath:
            self.open_video(filepath)

    def open_video(self, filepath):
        self.cap = cv2.VideoCapture(filepath)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, self.total_frames - 1)
        self.timer.start(30)

    def play_video(self):
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimage = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimage))

            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.setValue(current_frame)
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.setValue(current_frame)
            self.frame_label.setText(f"Frame: {current_frame}")
        else:
            self.timer.stop()

    def play_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(30)
            self.play_button.setText("Pause")

    def set_position(self, position):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)

    def next_frame(self):
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.set_position(current_frame + 1)

    def last_frame(self):
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.set_position(current_frame - 1)

class MultiVideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        # pdb.set_trace()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Multi Video Player")
        self.central_widget = QWidget()
        self.layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.video_players_widget = QWidget()
        self.video_players_layout = QVBoxLayout()

        for i in range(2):  # 您可以根據需要增加或減少播放器的數量
            video_player = VideoPlayer()
            self.video_players_layout.addWidget(video_player)

        self.video_players_widget.setLayout(self.video_players_layout)
        self.scroll_area.setWidget(self.video_players_widget)
        self.layout.addWidget(self.scroll_area)

        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = MultiVideoPlayer()
    window.show()
    sys.exit(app.exec_())
