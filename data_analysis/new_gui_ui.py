# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'new_gui.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QMainWindow, QMenu, QMenuBar, QPushButton,
    QSizePolicy, QSlider, QSpacerItem, QTextEdit,
    QToolBar, QVBoxLayout, QWidget)

from pyqtgraph import PlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1158, 777)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setLayoutDirection(Qt.LeftToRight)
        self.actionMivos = QAction(MainWindow)
        self.actionMivos.setObjectName(u"actionMivos")
        self.actionMotion_Analysis = QAction(MainWindow)
        self.actionMotion_Analysis.setObjectName(u"actionMotion_Analysis")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_5 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.button_widget = QWidget(self.centralwidget)
        self.button_widget.setObjectName(u"button_widget")
        self.button_widget.setEnabled(True)
        self.horizontalLayout_2 = QHBoxLayout(self.button_widget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.Upload_video = QPushButton(self.button_widget)
        self.Upload_video.setObjectName(u"Upload_video")

        self.horizontalLayout_2.addWidget(self.Upload_video)

        self.Use_camera = QPushButton(self.button_widget)
        self.Use_camera.setObjectName(u"Use_camera")

        self.horizontalLayout_2.addWidget(self.Use_camera)

        self.horizontalSpacer = QSpacerItem(60, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.left_one_frame = QPushButton(self.button_widget)
        self.left_one_frame.setObjectName(u"left_one_frame")

        self.horizontalLayout_2.addWidget(self.left_one_frame)

        self.right_one_frame = QPushButton(self.button_widget)
        self.right_one_frame.setObjectName(u"right_one_frame")

        self.horizontalLayout_2.addWidget(self.right_one_frame)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_3)

        self.playBtn = QPushButton(self.button_widget)
        self.playBtn.setObjectName(u"playBtn")

        self.horizontalLayout_2.addWidget(self.playBtn)

        self.stopBtn = QPushButton(self.button_widget)
        self.stopBtn.setObjectName(u"stopBtn")

        self.horizontalLayout_2.addWidget(self.stopBtn)


        self.verticalLayout_2.addWidget(self.button_widget)

        self.main_canvas = QLabel(self.centralwidget)
        self.main_canvas.setObjectName(u"main_canvas")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.main_canvas.sizePolicy().hasHeightForWidth())
        self.main_canvas.setSizePolicy(sizePolicy1)
        self.main_canvas.setMinimumSize(QSize(0, 0))
        self.main_canvas.setMouseTracking(False)
        self.main_canvas.setStyleSheet(u"background-color: rgb(0, 0, 0);")
        self.main_canvas.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.main_canvas)

        self.video_progress = QWidget(self.centralwidget)
        self.video_progress.setObjectName(u"video_progress")
        self.horizontalLayout_3 = QHBoxLayout(self.video_progress)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.tl_slider = QSlider(self.video_progress)
        self.tl_slider.setObjectName(u"tl_slider")
        self.tl_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_3.addWidget(self.tl_slider)

        self.horizontalSpacer_2 = QSpacerItem(50, 20, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_2)

        self.position_label = QLabel(self.video_progress)
        self.position_label.setObjectName(u"position_label")

        self.horizontalLayout_3.addWidget(self.position_label)


        self.verticalLayout_2.addWidget(self.video_progress)

        self.widget_3 = QWidget(self.centralwidget)
        self.widget_3.setObjectName(u"widget_3")
        self.horizontalLayout_4 = QHBoxLayout(self.widget_3)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.athelete_title = QLabel(self.widget_3)
        self.athelete_title.setObjectName(u"athelete_title")
        font = QFont()
        font.setPointSize(14)
        self.athelete_title.setFont(font)
        self.athelete_title.setMouseTracking(False)
        self.athelete_title.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.athelete_title.setMargin(-2)

        self.horizontalLayout_4.addWidget(self.athelete_title)

        self.athelete_title_2 = QLabel(self.widget_3)
        self.athelete_title_2.setObjectName(u"athelete_title_2")
        self.athelete_title_2.setFont(font)
        self.athelete_title_2.setMouseTracking(False)
        self.athelete_title_2.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.athelete_title_2.setMargin(-2)

        self.horizontalLayout_4.addWidget(self.athelete_title_2, 0, Qt.AlignHCenter)


        self.verticalLayout_2.addWidget(self.widget_3)

        self.athelete_data = QWidget(self.centralwidget)
        self.athelete_data.setObjectName(u"athelete_data")
        sizePolicy1.setHeightForWidth(self.athelete_data.sizePolicy().hasHeightForWidth())
        self.athelete_data.setSizePolicy(sizePolicy1)
        self.athelete_data.setMaximumSize(QSize(16777215, 360))
        self.athelete_data.setStyleSheet(u"background-color: rgb(192, 192, 192);")
        self.horizontalLayout = QHBoxLayout(self.athelete_data)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.widget = QWidget(self.athelete_data)
        self.widget.setObjectName(u"widget")
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy2)
        self.left_body = QVBoxLayout(self.widget)
        self.left_body.setObjectName(u"left_body")
        self.label_12 = QLabel(self.widget)
        self.label_12.setObjectName(u"label_12")
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy3)

        self.left_body.addWidget(self.label_12)

        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")
        sizePolicy3.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy3)

        self.left_body.addWidget(self.label)

        self.label_6 = QLabel(self.widget)
        self.label_6.setObjectName(u"label_6")
        sizePolicy3.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy3)

        self.left_body.addWidget(self.label_6)

        self.label_7 = QLabel(self.widget)
        self.label_7.setObjectName(u"label_7")
        sizePolicy3.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy3)

        self.left_body.addWidget(self.label_7)

        self.label_8 = QLabel(self.widget)
        self.label_8.setObjectName(u"label_8")
        sizePolicy3.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy3)

        self.left_body.addWidget(self.label_8)

        self.label_9 = QLabel(self.widget)
        self.label_9.setObjectName(u"label_9")
        sizePolicy3.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy3)

        self.left_body.addWidget(self.label_9)

        self.label_10 = QLabel(self.widget)
        self.label_10.setObjectName(u"label_10")
        sizePolicy3.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy3)

        self.left_body.addWidget(self.label_10)

        self.label_11 = QLabel(self.widget)
        self.label_11.setObjectName(u"label_11")
        sizePolicy3.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy3)

        self.left_body.addWidget(self.label_11)


        self.horizontalLayout.addWidget(self.widget)

        self.widget_2 = QWidget(self.athelete_data)
        self.widget_2.setObjectName(u"widget_2")
        sizePolicy4 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy4)
        self.left_body_acc = QVBoxLayout(self.widget_2)
        self.left_body_acc.setObjectName(u"left_body_acc")
        self.l_eye_acc = QTextEdit(self.widget_2)
        self.l_eye_acc.setObjectName(u"l_eye_acc")
        sizePolicy3.setHeightForWidth(self.l_eye_acc.sizePolicy().hasHeightForWidth())
        self.l_eye_acc.setSizePolicy(sizePolicy3)
        self.l_eye_acc.setMaximumSize(QSize(80, 16777215))

        self.left_body_acc.addWidget(self.l_eye_acc)

        self.l_ear_acc = QTextEdit(self.widget_2)
        self.l_ear_acc.setObjectName(u"l_ear_acc")
        sizePolicy3.setHeightForWidth(self.l_ear_acc.sizePolicy().hasHeightForWidth())
        self.l_ear_acc.setSizePolicy(sizePolicy3)
        self.l_ear_acc.setMaximumSize(QSize(80, 16777215))

        self.left_body_acc.addWidget(self.l_ear_acc)

        self.l_shoulder_acc = QTextEdit(self.widget_2)
        self.l_shoulder_acc.setObjectName(u"l_shoulder_acc")
        sizePolicy3.setHeightForWidth(self.l_shoulder_acc.sizePolicy().hasHeightForWidth())
        self.l_shoulder_acc.setSizePolicy(sizePolicy3)
        self.l_shoulder_acc.setMaximumSize(QSize(80, 16777215))

        self.left_body_acc.addWidget(self.l_shoulder_acc)

        self.l_elbow_acc = QTextEdit(self.widget_2)
        self.l_elbow_acc.setObjectName(u"l_elbow_acc")
        sizePolicy3.setHeightForWidth(self.l_elbow_acc.sizePolicy().hasHeightForWidth())
        self.l_elbow_acc.setSizePolicy(sizePolicy3)
        self.l_elbow_acc.setMaximumSize(QSize(80, 16777215))

        self.left_body_acc.addWidget(self.l_elbow_acc)

        self.l_wrist_acc = QTextEdit(self.widget_2)
        self.l_wrist_acc.setObjectName(u"l_wrist_acc")
        sizePolicy3.setHeightForWidth(self.l_wrist_acc.sizePolicy().hasHeightForWidth())
        self.l_wrist_acc.setSizePolicy(sizePolicy3)
        self.l_wrist_acc.setMaximumSize(QSize(80, 16777215))

        self.left_body_acc.addWidget(self.l_wrist_acc)

        self.l_hip_acc = QTextEdit(self.widget_2)
        self.l_hip_acc.setObjectName(u"l_hip_acc")
        sizePolicy3.setHeightForWidth(self.l_hip_acc.sizePolicy().hasHeightForWidth())
        self.l_hip_acc.setSizePolicy(sizePolicy3)
        self.l_hip_acc.setMaximumSize(QSize(80, 16777215))

        self.left_body_acc.addWidget(self.l_hip_acc)

        self.l_knee_acc = QTextEdit(self.widget_2)
        self.l_knee_acc.setObjectName(u"l_knee_acc")
        sizePolicy3.setHeightForWidth(self.l_knee_acc.sizePolicy().hasHeightForWidth())
        self.l_knee_acc.setSizePolicy(sizePolicy3)
        self.l_knee_acc.setMaximumSize(QSize(80, 16777215))

        self.left_body_acc.addWidget(self.l_knee_acc)

        self.l_ankle_acc = QTextEdit(self.widget_2)
        self.l_ankle_acc.setObjectName(u"l_ankle_acc")
        sizePolicy3.setHeightForWidth(self.l_ankle_acc.sizePolicy().hasHeightForWidth())
        self.l_ankle_acc.setSizePolicy(sizePolicy3)
        self.l_ankle_acc.setMaximumSize(QSize(80, 16777215))

        self.left_body_acc.addWidget(self.l_ankle_acc)


        self.horizontalLayout.addWidget(self.widget_2)

        self.widget1 = QWidget(self.athelete_data)
        self.widget1.setObjectName(u"widget1")
        sizePolicy4.setHeightForWidth(self.widget1.sizePolicy().hasHeightForWidth())
        self.widget1.setSizePolicy(sizePolicy4)
        self.right_body = QVBoxLayout(self.widget1)
        self.right_body.setObjectName(u"right_body")
        self.label_13 = QLabel(self.widget1)
        self.label_13.setObjectName(u"label_13")
        sizePolicy3.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy3)

        self.right_body.addWidget(self.label_13)

        self.label_14 = QLabel(self.widget1)
        self.label_14.setObjectName(u"label_14")
        sizePolicy3.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy3)

        self.right_body.addWidget(self.label_14)

        self.label_15 = QLabel(self.widget1)
        self.label_15.setObjectName(u"label_15")
        sizePolicy3.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy3)

        self.right_body.addWidget(self.label_15)

        self.label_16 = QLabel(self.widget1)
        self.label_16.setObjectName(u"label_16")
        sizePolicy3.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy3)

        self.right_body.addWidget(self.label_16)

        self.label_17 = QLabel(self.widget1)
        self.label_17.setObjectName(u"label_17")
        sizePolicy3.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy3)

        self.right_body.addWidget(self.label_17)

        self.label_18 = QLabel(self.widget1)
        self.label_18.setObjectName(u"label_18")
        sizePolicy3.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy3)

        self.right_body.addWidget(self.label_18)

        self.label_19 = QLabel(self.widget1)
        self.label_19.setObjectName(u"label_19")
        sizePolicy3.setHeightForWidth(self.label_19.sizePolicy().hasHeightForWidth())
        self.label_19.setSizePolicy(sizePolicy3)

        self.right_body.addWidget(self.label_19)

        self.label_20 = QLabel(self.widget1)
        self.label_20.setObjectName(u"label_20")
        sizePolicy3.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy3)

        self.right_body.addWidget(self.label_20)


        self.horizontalLayout.addWidget(self.widget1)

        self.widget2 = QWidget(self.athelete_data)
        self.widget2.setObjectName(u"widget2")
        sizePolicy4.setHeightForWidth(self.widget2.sizePolicy().hasHeightForWidth())
        self.widget2.setSizePolicy(sizePolicy4)
        self.right_body_acc = QVBoxLayout(self.widget2)
        self.right_body_acc.setObjectName(u"right_body_acc")
        self.r_eye_acc = QTextEdit(self.widget2)
        self.r_eye_acc.setObjectName(u"r_eye_acc")
        sizePolicy3.setHeightForWidth(self.r_eye_acc.sizePolicy().hasHeightForWidth())
        self.r_eye_acc.setSizePolicy(sizePolicy3)
        self.r_eye_acc.setMaximumSize(QSize(80, 16777215))

        self.right_body_acc.addWidget(self.r_eye_acc)

        self.r_ear_acc = QTextEdit(self.widget2)
        self.r_ear_acc.setObjectName(u"r_ear_acc")
        sizePolicy3.setHeightForWidth(self.r_ear_acc.sizePolicy().hasHeightForWidth())
        self.r_ear_acc.setSizePolicy(sizePolicy3)
        self.r_ear_acc.setMaximumSize(QSize(80, 16777215))

        self.right_body_acc.addWidget(self.r_ear_acc)

        self.r_shoulder_acc = QTextEdit(self.widget2)
        self.r_shoulder_acc.setObjectName(u"r_shoulder_acc")
        sizePolicy3.setHeightForWidth(self.r_shoulder_acc.sizePolicy().hasHeightForWidth())
        self.r_shoulder_acc.setSizePolicy(sizePolicy3)
        self.r_shoulder_acc.setMaximumSize(QSize(80, 16777215))

        self.right_body_acc.addWidget(self.r_shoulder_acc)

        self.r_elbow_acc = QTextEdit(self.widget2)
        self.r_elbow_acc.setObjectName(u"r_elbow_acc")
        sizePolicy3.setHeightForWidth(self.r_elbow_acc.sizePolicy().hasHeightForWidth())
        self.r_elbow_acc.setSizePolicy(sizePolicy3)
        self.r_elbow_acc.setMaximumSize(QSize(80, 16777215))

        self.right_body_acc.addWidget(self.r_elbow_acc)

        self.r_wrist_acc = QTextEdit(self.widget2)
        self.r_wrist_acc.setObjectName(u"r_wrist_acc")
        sizePolicy.setHeightForWidth(self.r_wrist_acc.sizePolicy().hasHeightForWidth())
        self.r_wrist_acc.setSizePolicy(sizePolicy)
        self.r_wrist_acc.setMaximumSize(QSize(80, 16777215))

        self.right_body_acc.addWidget(self.r_wrist_acc)

        self.r_hip_acc = QTextEdit(self.widget2)
        self.r_hip_acc.setObjectName(u"r_hip_acc")
        sizePolicy.setHeightForWidth(self.r_hip_acc.sizePolicy().hasHeightForWidth())
        self.r_hip_acc.setSizePolicy(sizePolicy)
        self.r_hip_acc.setMaximumSize(QSize(80, 16777215))

        self.right_body_acc.addWidget(self.r_hip_acc)

        self.r_knee_acc = QTextEdit(self.widget2)
        self.r_knee_acc.setObjectName(u"r_knee_acc")
        sizePolicy.setHeightForWidth(self.r_knee_acc.sizePolicy().hasHeightForWidth())
        self.r_knee_acc.setSizePolicy(sizePolicy)
        self.r_knee_acc.setMaximumSize(QSize(80, 16777215))

        self.right_body_acc.addWidget(self.r_knee_acc)

        self.r_ankle_acc = QTextEdit(self.widget2)
        self.r_ankle_acc.setObjectName(u"r_ankle_acc")
        sizePolicy.setHeightForWidth(self.r_ankle_acc.sizePolicy().hasHeightForWidth())
        self.r_ankle_acc.setSizePolicy(sizePolicy)
        self.r_ankle_acc.setMaximumSize(QSize(80, 16777215))

        self.right_body_acc.addWidget(self.r_ankle_acc)


        self.horizontalLayout.addWidget(self.widget2)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.sensors_pg = PlotWidget(self.athelete_data)
        self.sensors_pg.setObjectName(u"sensors_pg")
        self.sensors_pg.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.sensors_pg.sizePolicy().hasHeightForWidth())
        self.sensors_pg.setSizePolicy(sizePolicy1)
        self.sensors_pg.setMinimumSize(QSize(0, 300))
        font1 = QFont()
        font1.setBold(False)
        self.sensors_pg.setFont(font1)
        self.sensors_pg.setLayoutDirection(Qt.LeftToRight)
        self.sensors_pg.setStyleSheet(u"background-color: rgb(255, 255, 0);")

        self.verticalLayout.addWidget(self.sensors_pg)

        self.comboBox = QComboBox(self.athelete_data)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")
        sizePolicy5 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy5)

        self.verticalLayout.addWidget(self.comboBox)


        self.horizontalLayout.addLayout(self.verticalLayout)


        self.verticalLayout_2.addWidget(self.athelete_data)


        self.horizontalLayout_5.addLayout(self.verticalLayout_2)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        font2 = QFont()
        font2.setPointSize(11)
        self.label_2.setFont(font2)

        self.verticalLayout_3.addWidget(self.label_2, 0, Qt.AlignHCenter)

        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setFont(font)

        self.verticalLayout_3.addWidget(self.label_3, 0, Qt.AlignHCenter)

        self.height_pg = PlotWidget(self.centralwidget)
        self.height_pg.setObjectName(u"height_pg")
        sizePolicy1.setHeightForWidth(self.height_pg.sizePolicy().hasHeightForWidth())
        self.height_pg.setSizePolicy(sizePolicy1)
        self.height_pg.setStyleSheet(u"background-color: rgb(255, 0, 255);")

        self.verticalLayout_3.addWidget(self.height_pg)

        self.label_21 = QLabel(self.centralwidget)
        self.label_21.setObjectName(u"label_21")
        sizePolicy.setHeightForWidth(self.label_21.sizePolicy().hasHeightForWidth())
        self.label_21.setSizePolicy(sizePolicy)
        self.label_21.setFont(font)

        self.verticalLayout_3.addWidget(self.label_21, 0, Qt.AlignHCenter)

        self.twist_pg = PlotWidget(self.centralwidget)
        self.twist_pg.setObjectName(u"twist_pg")
        sizePolicy6 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.twist_pg.sizePolicy().hasHeightForWidth())
        self.twist_pg.setSizePolicy(sizePolicy6)
        self.twist_pg.setStyleSheet(u"background-color: rgb(0, 255, 0);")

        self.verticalLayout_3.addWidget(self.twist_pg)

        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setFont(font)

        self.verticalLayout_3.addWidget(self.label_4, 0, Qt.AlignHCenter)

        self.hand_off_pg = PlotWidget(self.centralwidget)
        self.hand_off_pg.setObjectName(u"hand_off_pg")
        sizePolicy6.setHeightForWidth(self.hand_off_pg.sizePolicy().hasHeightForWidth())
        self.hand_off_pg.setSizePolicy(sizePolicy6)
        self.hand_off_pg.setMinimumSize(QSize(0, 0))
        self.hand_off_pg.setStyleSheet(u"background-color: rgb(0, 0, 255);")

        self.verticalLayout_3.addWidget(self.hand_off_pg)


        self.horizontalLayout_5.addLayout(self.verticalLayout_3)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 1158, 20))
        self.menu = QMenu(self.menuBar)
        self.menu.setObjectName(u"menu")
        self.menuWindows = QMenu(self.menuBar)
        self.menuWindows.setObjectName(u"menuWindows")
        self.menuExit = QMenu(self.menuBar)
        self.menuExit.setObjectName(u"menuExit")
        MainWindow.setMenuBar(self.menuBar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.menuBar.addAction(self.menu.menuAction())
        self.menuBar.addAction(self.menuWindows.menuAction())
        self.menuBar.addAction(self.menuExit.menuAction())
        self.menuWindows.addAction(self.actionMivos)
        self.menuWindows.addAction(self.actionMotion_Analysis)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionMivos.setText(QCoreApplication.translate("MainWindow", u"Mivos", None))
        self.actionMotion_Analysis.setText(QCoreApplication.translate("MainWindow", u"Motion Analysis", None))
        self.Upload_video.setText(QCoreApplication.translate("MainWindow", u"Upload Video", None))
        self.Use_camera.setText(QCoreApplication.translate("MainWindow", u"Use Camera", None))
        self.left_one_frame.setText(QCoreApplication.translate("MainWindow", u"Left", None))
        self.right_one_frame.setText(QCoreApplication.translate("MainWindow", u"Right", None))
        self.playBtn.setText(QCoreApplication.translate("MainWindow", u"START", None))
        self.stopBtn.setText(QCoreApplication.translate("MainWindow", u"STOP", None))
        self.main_canvas.setText(QCoreApplication.translate("MainWindow", u"indicate video", None))
        self.position_label.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.athelete_title.setText(QCoreApplication.translate("MainWindow", u" Athlete and Video Information", None))
        self.athelete_title_2.setText(QCoreApplication.translate("MainWindow", u"hip angle", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"l-eye", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"l-ear", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"l-shoulder", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"l-elbow", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"l-wrist", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"l-hip", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"l-knee", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"l-ankle", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"r-eye", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"r-ear", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"r-shoulder", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"r-elbow", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"r-wrist", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"r-hip", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"r-knee", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"r-ankle", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Left_Body", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Right_Body", None))

        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Parameters", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u9ad8\u5ea6 (Height)", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"\u8f49\u901f (TWIST & TURN ROTATION SPEED)", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u6eef\u7a7a\u6642\u9593 (HAND OFF TIME)", None))
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuWindows.setTitle(QCoreApplication.translate("MainWindow", u"Windows", None))
        self.menuExit.setTitle(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

