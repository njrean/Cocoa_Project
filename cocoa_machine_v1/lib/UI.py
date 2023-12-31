import typing
import cv2
import os
from datetime import datetime

import serial
import serial.tools.list_ports
import time
import heapq
import threading
from threading import Timer, Thread

from collections import defaultdict, OrderedDict
from queue import Queue

import numpy as np
from PyQt5.QtCore import QLibraryInfo, QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QImage, QPixmap
from PyQt5.QtWidgets import (QMainWindow,
                            QWidget,
                            QDialog,
                            QVBoxLayout,
                            QHBoxLayout,
                            QGridLayout,
                            QGroupBox,
                            QLabel,
                            QSlider,
                            QSpinBox,
                            QLineEdit,
                            QCheckBox,
                            QComboBox,
                            QPushButton,
                            QApplication, 
                            QStyleFactory,
                            )

from pyqtgraph import PlotWidget, plot

from lib.Camera import Camera
from lib.Proprocessing import Preprocessing
from lib.Bayesian_Segmentation import Bayesian_Segmentation
from lib.Bayesian_Classification import Bayesian_Classsification
from lib.Tracker import Tracker
# from lib.Scheduler import Scheduler
from lib.function import crop_bean, find_centroid_conner

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

font = cv2.FONT_HERSHEY_SIMPLEX

#Variable which use in multithreads
ids = []
classes = []
centroids = []
timestamp = 0
queue_collect = Queue()
queue_check = Queue()
heap_stamp = []

class MainWindow(QMainWindow): 
    def __init__(self, webcam:Camera, 
                preprocessing:Preprocessing, 
                model_segment:Bayesian_Segmentation,
                model_classify:Bayesian_Classsification,
                tracker:Tracker):
        
        super(MainWindow, self).__init__()

        main_widget = Main_widget(webcam, preprocessing, 
                                  model_segment, 
                                  model_classify, 
                                  tracker)
        
        self.setCentralWidget(main_widget)

class Main_widget(QWidget):
    def __init__(self, webcam:Camera, 
                preprocessing:Preprocessing,
                model_segment:Bayesian_Segmentation,
                model_classify:Bayesian_Classsification,
                tracker:Tracker):
        
        super(Main_widget, self).__init__()

        self.webcam = webcam
        self.preprocessing = preprocessing
        self.model_segment = model_segment
        self.model_classify = model_classify
        self.tracker = tracker
        
        #Stream Video
        self.disply_width = int(preprocessing.original_image_w/2)
        self.display_height = preprocessing.original_image_h
        self.video_label = QLabel(self)
        self.video_label.resize(self.disply_width, self.display_height)

        self.thread = VideoThread(webcam, preprocessing)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        
        self.scheduler = Scheduler(self.preprocessing.ref_point1_x, self.preprocessing.unit)
        self.scheduler.start()

        #graph plot 
        self.graphWidget = PlotWidget()

        #Group of control
        group_botton = QGroupBox('Control')
        layout_botton = QVBoxLayout()

        ##button for open setting window
        self.botton_setting = QPushButton('setting', group_botton)
        self.botton_setting.clicked.connect(self.setting_on_click)
        layout_botton.addWidget(self.botton_setting)

        ##button for refresh camera parameter
        self.botton_camera_update = QPushButton('refresh camera', group_botton)
        self.botton_camera_update.clicked.connect(self.update_camera_on_click)
        layout_botton.addWidget(self.botton_camera_update)

        ##button for take a picture
        self.image_crop_save = np.zeros((preprocessing.original_image_w, preprocessing.original_image_h))
        self.image_tran_save = np.zeros((preprocessing.original_image_w, preprocessing.original_image_h))
        self.botton_cupture = QPushButton('capture', group_botton)
        self.botton_cupture.clicked.connect(self.save_picture)
        layout_botton.addWidget(self.botton_cupture)

        ##Checker for activate sorting system
        self.activate_sorting = CheckBox("Sorting system", True)
        self.activate_sorting.addWidget(layout_botton)

        ##motor status text and control button
        self.motor_status_widget = TextBox('Motor status', 'deactivate')
        self.motor_state = 0
        layout_botton.addWidget(self.motor_status_widget.label)
        layout_botton.addWidget(self.motor_status_widget.box)
        
        self.botton_motor = QPushButton('motor control', group_botton)
        self.botton_motor.clicked.connect(self.motor_send)
        layout_botton.addWidget(self.botton_motor)

        ##activate image segmentation for showing or not for image mode
        self.activate_segment = ComboBox('Shown segmentated image', ['Transform', 'Probability Map', 'Mask', 'Apply Mask', 'Mode set reference', 'Transform'])
        self.activate_segment.addWidget(layout_botton)

        group_botton.setLayout(layout_botton)

        #MainLayoutf
        main_layout = QGridLayout()
        main_layout.addWidget(self.video_label, 0, 0, 6, 3)
        main_layout.addWidget(group_botton, 3, 4, 1, 1)
        self.setLayout(main_layout)
        
    def closeEvent(self, event):
        print("Close")
        self.scheduler.stop()
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):

        global ids
        global classes
        global centroids
        global timestamp
        global queue_collect

        timestamp = time.time()

        img_tran, img_crop, img_extract = self.preprocessing.preprocess_pipeline(cv_img)

        self.image_crop_save = img_crop
        self.image_tran_save = img_tran

        mask, prob_map = self.model_segment.segment(img_extract, 
                                                    k=[0.1, 0.1, 0, 0.3, 0.4, 0.3], 
                                                    threshold=0.35, 
                                                    filter=True)
        _, bounds, sep_masks, flag_found = crop_bean(mask) #mask separate bean

        match self.activate_segment.box.currentText():
            case 'Mode set reference':
                img_show = img_tran
                reference_point1 = (self.preprocessing.ref_point1_x, self.preprocessing.ref_point1_y)
                reference_point2 = (self.preprocessing.ref_point2_x, self.preprocessing.ref_point2_y)
                img_show = cv2.circle(img_show, reference_point1, 3, (255,0, 255), -1)
                img_show = cv2.circle(img_show, reference_point2, 3, (0,255, 255), -1)
            case 'Transform':
                img_show = img_crop
                img_show = np.pad(img_show, ((self.preprocessing.ROI_up, cv_img.shape[0]-self.preprocessing.ROI_down), 
                                            (self.preprocessing.ROI_left, cv_img.shape[1]-self.preprocessing.ROI_right),
                                            (0, 0)), 'constant')
            case 'Mask':
                img_show = np.repeat((mask*255).astype(np.uint8), 3, axis=1).reshape(img_crop.shape)
                img_show = np.pad(img_show, ((self.preprocessing.ROI_up, cv_img.shape[0]-self.preprocessing.ROI_down), 
                                            (self.preprocessing.ROI_left, cv_img.shape[1]-self.preprocessing.ROI_right),
                                            (0, 0)), 'constant')
            case 'Probability Map':
                img_show = np.repeat((prob_map*255).astype(np.uint8), 3, axis=1).reshape(img_crop.shape)
                img_show = np.pad(img_show, ((self.preprocessing.ROI_up, cv_img.shape[0]-self.preprocessing.ROI_down), 
                                            (self.preprocessing.ROI_left, cv_img.shape[1]-self.preprocessing.ROI_right),
                                            (0, 0)), 'constant')
            case 'Apply Mask':
                mask_repeat = np.repeat((mask*255).astype(np.uint8), 3, axis=1).reshape(img_crop.shape)
                img_show = cv2.bitwise_and(img_crop, mask_repeat)
                img_show = np.pad(img_show, ((self.preprocessing.ROI_up, cv_img.shape[0]-self.preprocessing.ROI_down), 
                                            (self.preprocessing.ROI_left, cv_img.shape[1]-self.preprocessing.ROI_right),
                                            (0, 0)), 'constant')


        centroids = []
        classes = []

        #Get offset to draw in padded image
        y_offset = self.preprocessing.ROI_up
        x_offset =self.preprocessing.ROI_left

        if self.activate_sorting.box.isChecked():

            if flag_found:
                for i, bean_bound in enumerate(bounds):
                    _, centroid = find_centroid_conner(sep_masks[i])
                    #Cen x and y relate to global frame
                    cen_x = int(centroid[1]) + bean_bound[0] + x_offset #need to add offset in x axis in local mask and global image
                    cen_y = int(centroid[0]) + y_offset 
                    centroids.append([cen_x, cen_y])
                bean_prob, classes = self.model_classify.predict(sep_masks)

            ids = self.tracker.update(centroids)
            queue_collect.put((ids, centroids, classes, timestamp))
            
            #Draw bounding x axis / centroid / class left to right bean
            if flag_found:
                for i, bean_bound in enumerate(bounds):
                    img_show[centroids[i][1]-4:centroids[i][1]+4, centroids[i][0]-4:centroids[i][0]+4] = [0, 255, 0]

                    text = "ID:{} {} P:{:.2f}".format(self.tracker.IDs[i] , self.model_classify.index2label[classes[i]], bean_prob[i][classes[i]])

                    img_show = cv2.putText(img_show, text, 
                                        (bean_bound[0]+3+x_offset, y_offset-20), 
                                        font, 
                                        0.7, 
                                        (255, 0, 0), 
                                        2, 
                                        cv2.LINE_AA)

        stack_image = np.concatenate((cv_img, img_show), axis=0)

        qt_img = self.convert_cv_qt(stack_image)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, img_show):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)
     
    def setting_on_click(self):
        config_window = ConfigWindow(self.webcam, self.preprocessing)
        config_window.exec()

    def update_camera_on_click(self):
        self.webcam.set_camera_profile()

    def motor_send(self):
        data = calculate_data_send([4])
        send_UART(data)
        self.motor_state = (self.motor_state+1)%2

        self.motor_status_widget.box.setText("activate" if self.motor_state else "deactivate")

    def save_picture(self):
        date = datetime.now()
        flag1 = cv2.imwrite('./image_save/transform/{}.png'.format(date), self.image_tran_save)
        flag2 = cv2.imwrite('./image_save/crop/{}.png'.format(date), self.image_crop_save)
        if flag1 and flag2:
            print("Capture Sucess")

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, webcam:Camera, preprocessing:Preprocessing):
        super().__init__()
        self.preprocessing = preprocessing
        self.__run__flag = True
        self.webcam = webcam

    def run(self):
        # capture from web cam
        self.webcam.camera_start()

        while self.__run__flag:
            ret, img_read = self.webcam.capture()

            if ret:
                self.change_pixmap_signal.emit(img_read)

        self.webcam.camera_stop()

    def stop(self):
        self.__run__flag = False
        self.wait()

class ConfigWindow(QDialog):
    def __init__(self, webcam:Camera, preprocessing:Preprocessing):
        super(ConfigWindow, self).__init__()

        self.webcam = webcam
        self.camera_prof = dict()

        self.preprocessing = preprocessing

        #Create Group
        self.create_image_transforming(preprocessing.data)
        self.create_camera_setting(webcam.get_camera_info())
        self.create_model_parameter_setting()

        #Layout
        main_Layout = QGridLayout()
        main_Layout.addWidget(self.image_transform_group, 0, 0, 1, 1)
        main_Layout.addWidget(self.cameraSetting_group, 0, 1, 5, 5)
        main_Layout.addWidget(self.model_parameter_group, 1, 0)
        
        self.setLayout(main_Layout)

        #set main config of window
        self.setWindowTitle("Machine Configuration")
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        QApplication.setPalette(QApplication.style().standardPalette())

    def create_camera_setting(self, camera_info):

        self.cameraSetting_group = QGroupBox("Camera Setting")

        self.camera_config_widgets = []

        for info_key in camera_info:
            self.camera_prof[info_key] = int(camera_info[info_key]["value"])

            if camera_info[info_key]["type"] == "(int)":
                widget = ValueSliderLink(self.cameraSetting_group, 
                                        int(camera_info[info_key]["min"]),
                                        int(camera_info[info_key]["max"]), 
                                        int(camera_info[info_key]["value"]),
                                        info_key)
                widget.box.valueChanged.connect(self.update_camera)
                
            elif camera_info[info_key]["type"] == "(bool)":
                widget = CheckBox(info_key, int(camera_info[info_key]["value"]))
                widget.box.stateChanged.connect(self.update_camera)

            elif camera_info[info_key]["type"] == "(menu)":
                widget = ComboBox(info_key, camera_info[info_key])
                widget.box.currentTextChanged.connect(self.update_camera)

            self.camera_config_widgets.append(widget)

        layout = QGridLayout()

        for i, widget in enumerate(self.camera_config_widgets):
            widget.addWidget(layout, i)

        save_botton = QPushButton('save camera profile')
        save_botton.clicked.connect(self.save_camera_profile_on_click)
        layout.addWidget(save_botton, i+1, 0)

        self.cameraSetting_group.setLayout(layout)

    def create_image_transforming(self, image_parameter):

        half_h = int(np.floor(image_parameter["original_image_h"]/2))
        half_w = int(np.floor(image_parameter["original_image_w"]/2))

        self.image_parameter_info = {"original_image_h": {"min": 1, "max": image_parameter["original_image_h"], "type":"box"},
                                     "original_image_w": {"min": 1, "max": image_parameter["original_image_w"], "type":"box"},
                                     "Rotation_z": {"min": -1800, "max": 1800, "type":"slide"},
                                     "Translation_x": {"min": -image_parameter["original_image_w"], "max": image_parameter["original_image_w"], "type":"slide"},
                                     "Translation_y": {"min": -image_parameter["original_image_h"], "max": image_parameter["original_image_h"], "type":"slide"},
                                     "Boundary_UP": {"min": 0, "max": half_h, "type":"slide"},
                                     "Boundary_DOWN": {"min": half_h+1, "max": image_parameter["original_image_h"], "type":"slide"},
                                     "Boundary_LEFT": {"min": 0, "max": half_w, "type":"slide"},
                                     "Boundary_RIGHT": {"min": half_w+1, "max": image_parameter["original_image_w"], "type":"slide"},
                                     "reference_point1_x": {"min": 0, "max": image_parameter["original_image_w"], "type":"slide"},
                                     "reference_point1_y": {"min": 0, "max": image_parameter["original_image_h"], "type":"slide"},
                                     "reference_point2_x": {"min": 0, "max": image_parameter["original_image_w"], "type":"slide"},
                                     "reference_point2_y": {"min": 0, "max": image_parameter["original_image_h"], "type":"slide"},
                                     "unit_transition": {"min": 0, "max": 100, "type":"box"}}
        
        self.image_parameter_value = {"original_image_h": image_parameter["original_image_h"],
                                     "original_image_w": image_parameter["original_image_w"],
                                     "Rotation_z": image_parameter["Rotation_z"],
                                     "Translation_x": image_parameter["Translation_x"],
                                     "Translation_y": image_parameter["Translation_y"],
                                     "Boundary_UP": image_parameter['Boundary_UP'],
                                     "Boundary_DOWN": image_parameter['Boundary_DOWN'],
                                     "Boundary_LEFT": image_parameter['Boundary_LEFT'],
                                     "Boundary_RIGHT": image_parameter['Boundary_RIGHT'],
                                     "reference_point1_x": image_parameter['reference_point1_x'],
                                     "reference_point1_y": image_parameter['reference_point1_y'],
                                     "reference_point2_x": image_parameter['reference_point2_x'],
                                     "reference_point2_y": image_parameter['reference_point2_y'],
                                     "unit_transition": image_parameter['unit_transition']}
        
        self.image_transform_group = QGroupBox("Image Transform")
        layout = QGridLayout()
        
        self.image_parameter_widgets = []

        for i, parameter in enumerate(self.image_parameter_info):
            min = self.image_parameter_info[parameter]["min"]
            max = self.image_parameter_info[parameter]["max"]
            value = image_parameter[parameter]
            if self.image_parameter_info[parameter]["type"] == "slide":
                widget  = ValueSliderLink(self.image_transform_group, min, max, value, parameter)
                widget.box.valueChanged.connect(self.update_image_transformation)
                
            else:
                widget = NumberBox(parameter, 'float', min, max, "{:.2f}".format(value))
                widget.box.textChanged.connect(self.update_image_transformation)
                widget.box.setReadOnly(True)

            widget.addWidget(layout, i)
            self.image_parameter_widgets.append(widget)

        save_botton = QPushButton('save image parameter')
        save_botton.clicked.connect(self.save_image_parameter_on_click)
        layout.addWidget(save_botton, 15, 0)

        self.image_transform_group.setLayout(layout)

    def create_model_parameter_setting(self):
        self.model_parameter_group = QGroupBox("Segmentation Model Parameter Setting")

        k_Y = NumberBox('k (Y)', 'float', 0, 1, '0')
        k_Cb = NumberBox('k (Cb)', 'float', 0, 1, '0')
        k_Cr = NumberBox('k (Cr)', 'float', 0, 1, '0')
        k_B = NumberBox('k (B)', 'float', 0, 1, '0')
        k_G = NumberBox('k (G)', 'float', 0, 1, '0')
        k_R = NumberBox('k (R)', 'float', 0, 1, '0')
        segment_thres = NumberBox('Segmentation Threshold', 'float', 0, 1, '0.6')

        save_botton = QPushButton('save model parameter')
        save_botton.clicked.connect(self.save_model_parameter_on_click)
        
        layout = QGridLayout()
        k_Y.addWidget(layout, 0)
        k_Cb.addWidget(layout, 1)
        k_Cr.addWidget(layout, 2)
        k_B.addWidget(layout, 3)
        k_G.addWidget(layout, 4)
        k_R.addWidget(layout, 5)
        segment_thres.addWidget(layout, 6)
        layout.addWidget(save_botton)
        self.model_parameter_group.setLayout(layout)

    def update_camera(self):
        camera_prof_check = dict()
        for i, info in enumerate(self.webcam.camera_info):
            if self.webcam.camera_info[info]["type"] == "(int)":
                camera_prof_check[info] = self.camera_config_widgets[i].box.value()
            elif self.webcam.camera_info[info]["type"] == "(bool)":
                camera_prof_check[info] = int(self.camera_config_widgets[i].box.isChecked())
            elif self.webcam.camera_info[info]["type"] == "(menu)":
                camera_prof_check[info] = int(self.camera_config_widgets[i].box.currentText())
        
        if self.camera_prof != camera_prof_check:
            self.camera_prof = camera_prof_check
            self.webcam.update_camera_profile(self.camera_prof)

    def update_image_transformation(self):
        for i, parameter in enumerate(self.image_parameter_value):
            if self.image_parameter_info[parameter]["type"] == "slide":
                self.image_parameter_value[parameter] = self.image_parameter_widgets[i].box.value()
            # else: self.image_parameter_value[parameter] = int(self.image_parameter_widgets[i].box.text())
        self.image_parameter_value['unit_transition'] = 80/(self.image_parameter_value['reference_point1_x']-self.image_parameter_value['reference_point2_x']) #2reference point diff 80 mm , unit in mm/pixel
        self.image_parameter_widgets[13].box.setText("{:.2f}".format(self.image_parameter_value['unit_transition']))
        self.preprocessing.data = self.image_parameter_value
        self.preprocessing.update_parameter()
        

    def save_camera_profile_on_click(self):
        self.webcam.save_camera_prof()

    def save_image_parameter_on_click(self):
        self.preprocessing.save_image_parameter()

    def save_model_parameter_on_click(self):
        print('click')

class ValueSliderLink():
    def __init__(self, GroupBox:QGroupBox, minimum:int, maximum:int, initial_val, label:str):
        self.label = QLabel(label)
        self.slider = QSlider(Qt.Orientation.Horizontal, GroupBox)
        self.box = QSpinBox()
        self.box.setRange(minimum, maximum)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.valueChanged.connect(self.box.setValue)
        self.box.valueChanged.connect(self.slider.setValue)
        self.slider.setValue(initial_val)

    def addWidget(self, layout, row):
        layout.addWidget(self.label, row, 0)
        layout.addWidget(self.box, row, 1)
        layout.addWidget(self.slider, row, 2, 2, 4)
        layout.rowStretch(2)

class NumberBox():
    def __init__(self, label, type, minimum, maximum, initial_value):
        self.label = QLabel(label)
        self.box = QLineEdit()
        if type == "integer":
            self.box.setValidator(QIntValidator(1, 1080))
        else: self.box.setValidator(QDoubleValidator(minimum, maximum, 2))
        self.box.setText(initial_value)

    def addWidget(self, layout, row):
        layout.addWidget(self.label, row, 0)
        layout.addWidget(self.box, row, 1)

class TextBox():
    def __init__(self, label, initial_value):
        self.label = QLabel(label)
        self.box = QLineEdit()
        self.box.setText(initial_value)

class CheckBox():
    def __init__(self, label:str, defalt_set:bool):
        self.box = QCheckBox(label)
        self.box.setChecked(defalt_set)

    def addWidget(self, layout, row=None):
        if row != None:
            layout.addWidget(self.box, row, 0)
        else: layout.addWidget(self.box)

class ComboBox():

    def __init__(self, label:str, config_info):
        self.box = QComboBox()
        self.label = QLabel(label)
        if type(config_info) == dict:
            choice_list = [str(i) for i in range(int(config_info["min"]), int(config_info["max"]))]
            self.box.addItems(choice_list)
            self.box.setCurrentText(config_info["value"])
        elif type(config_info) == list:
            self.box.addItems(config_info[:-1])
            self.box.setCurrentText(config_info[-1])

    def addWidget(self, layout, row=None):
        if row != None:
            layout.addWidget(self.label, row, 0)
            layout.addWidget(self.box, row, 1)
        else:
            group = QGroupBox()
            sublayout = QHBoxLayout()
            sublayout.addWidget(self.label)
            sublayout.addWidget(self.box)
            group.setLayout(sublayout)
            layout.addWidget(group)

ports = serial.tools.list_ports.comports()
serial_UNO = serial.Serial(ports[0].device, 
                            baudrate=115200,
                            bytesize=8, 
                            parity="N", 
                            stopbits=1)

class Scheduler(QThread):
    def __init__(self, x_reference_point, unit):
        super(Scheduler, self).__init__()
        
        self.data_send = 0b0
        self.id_class = defaultdict(list)
        self.wait_beanID = 0 #bean ID which wait to set timestamp

        self.xpoint_ref = x_reference_point
        self.timeadd = 0.8 #delay time to reach stataion 0 from reference point
        self.timebt = [0, 0.3, 0.67, 0.98] #delay time between station
        self.s_ref20 =  131 #distance between reference point to station 0
        self.v_belt = 170 #velocity of belt in mm/s
        self.unit_trans = unit #transform unit pixel to mm

        self.queue_tick = Queue()

        self.planner = Planner(self.xpoint_ref, self.timeadd, self.timebt, self.s_ref20, self.v_belt, self.unit_trans)
        # self.checker = Checker()

        self.flag_schedule = True

    def run(self):
        self.planner.start()
        # self.checker.start()

    def stop(self):
        self.planner.stop()
        # self.checker.stop()

class Planner(QThread):
    def __init__(self, xpoint_ref, timeadd, timebt, s_ref20, v_belt, unit_trans) -> None:
        super(Planner, self).__init__()

        self.xpoint_ref = xpoint_ref
        self.timeadd = timeadd #delay time to reach stataion 0 from reference point
        self.timebt = timebt #delay time between station
        self.s_ref20 = s_ref20 #distance between reference point to station 0
        self.v_belt = v_belt #velocity of belt in mm/s
        self.unit_trans = unit_trans #transform unit pixel to mm

        self.flag_schedule = True
        self.queue_tick = Queue()
        self.wait_beanID = 0
        self.data_send = 0b0
        self.id_class = defaultdict(list)

    def collect_callback(self):
        global queue_collect
 
        next_data = queue_collect.get()
        ids = next_data[0][::-1]
        classes = next_data[2][::-1]
        centroids = next_data[1][::-1]

        for id, cl, cen in zip(ids, classes, centroids):
            if id >= self.wait_beanID:
                self.id_class[id].append(cl)

                print(id, cen, self.xpoint_ref)

                if cen[0] >= self.xpoint_ref:
                    self.queue_tick.put((id, cen[0], next_data[-1]))
                    self.wait_beanID += 1

    def ticker_callback(self):
        global queue_check
        #if considered bean in position
        if not(self.queue_tick.empty()):

            data = self.queue_tick.get()
            
            stack_classes = self.id_class[data[0]]
            bean_class = min(stack_classes)
            now = time.time()
            delay = ((self.s_ref20-((data[1]-self.xpoint_ref)*self.unit_trans))/self.v_belt) - now + data[2] + self.timebt[bean_class]
            # delay = 0.84 + (self.timebt * bean_class) - now + self.id_xCentroid[self.wait_beanID][1]

            time_set = Timer(delay, interupt_Handeler, args=[[bean_class], delay+now, data[1]])
            time_set.start()

            # queue_check.put((now+delay, [bean_class], time_set))

    def run(self):
        while self.flag_schedule:
            self.collect_callback()
            self.ticker_callback()

    def stop(self):
        self.flag_schedule = False
        self.wait()

class Checker(QThread):
    def __init__(self) -> None:
        super(Checker, self).__init__()
        self.flag_checker = True

    def scan(self):
        global queue_check
        global heap_stamp

        data = queue_check.get()
        # heapq.heappush(heap_stamp, data)
        heap_stamp.append(data)

        heap_stamp = [heapq.heappop(heap_stamp) for _ in range(len(heap_stamp))]
        heap_stamp = [item for item in heap_stamp if item[2].is_alive()]
        time_stamp = np.array([t for t, _, _ in heap_stamp])
        time_stamp = np.triu((time_stamp[:, np.newaxis] - time_stamp).T)

        ls = []
        exist = []

        for i in range(len(time_stamp)-1):
            find = time_stamp[i][i+1:] <= 0.0065
            if any(find):
                stop = np.max(np.where(find)[0])+i+1
                if not(stop in exist):
                    ls.append(heap_stamp[i:stop+1])
                    exist += list(range(i, stop+1))

        if ls != []: 
            for group in ls:

                group_classes = group[0][1]
                time_new = group[0][0]
                group[0][2].cancel()

                for bean in group[1:]:
                    if not(any(x in bean[1] for x in group_classes)):
                        bean[2].cancel()
                        group_classes += bean[1]
                       
                delay = time_new
                print("Classes", group_classes)
                print("New delay", delay)
                print("New delay", time.time())
                time_set = Timer(delay, interupt_Handeler, args=[group_classes, time_new, 0])
                time_set.start()
                newdata = (time_new, group_classes, time_set)
                heap_stamp.append(newdata)

    def run(self):
        global heap_stamp
        while self.flag_checker:
            self.scan()

    def stop(self):
        self.flag_checker = False
        self.wait()

#Function for timer thread interrupt
def calculate_data_send(cl):
        data = 0b0
        for i in cl:
            data |= 0b1 << i
        return int(data)

def send_UART(data:int):
    byte_number = data.bit_length() + 7 // 8
    text = data.to_bytes(byte_number, 'big').decode()
    serial_UNO.write(bytearray(text,'ascii'))
    # time.sleep(1)

def interupt_Handeler(cl, need_t, cen_x):
    data = calculate_data_send(cl)
    send_UART(data)

    print("Push Class {}".format(cl))
    print("Cen x:", cen_x)
    print("Need Time {}".format(need_t))
    print("Now Time {}".format(time.time()))

    