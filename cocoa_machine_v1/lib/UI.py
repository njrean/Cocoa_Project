import cv2
import os

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

from lib.Camera import Camera
from lib.Proprocessing import Preprocessing
from lib.Bayesian_Segmentation import Bayesian_Segmentation

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

class MainWindow(QMainWindow): 
    def __init__(self, webcam:Camera, 
                 preprocessing:Preprocessing, 
                 model_segment:Bayesian_Segmentation):
        
        super(MainWindow, self).__init__()
        main_widget = Main_widget(webcam, preprocessing, model_segment)
        self.setCentralWidget(main_widget)

class Main_widget(QWidget):
    def __init__(self, webcam:Camera, 
                 preprocessing:Preprocessing,
                 model_segment:Bayesian_Segmentation):
        
        super(Main_widget, self).__init__()
        self.webcam = webcam
        self.preprocessing = preprocessing
        self.model_segment = model_segment

        #Stream Video
        self.disply_width = preprocessing.original_image_w
        self.display_height = preprocessing.original_image_h * 2
        self.video_label = QLabel(self)
        self.video_label.resize(self.disply_width, self.display_height)

        self.thread = VideoThread(webcam, preprocessing)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

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
        ##activate image segmentation for showing or not for image mode
        self.activate_segment = ComboBox('Shown segmentated image', ['Transform', 'Probability Map', 'Mask', 'Apply Mask', 'Transform'])
        self.activate_segment.addWidget(layout_botton)

        group_botton.setLayout(layout_botton)

        #MainLayoutf
        main_layout = QGridLayout()
        main_layout.addWidget(self.video_label, 0, 0, 6, 3)
        main_layout.addWidget(group_botton, 3, 4, 1, 1)
        self.setLayout(main_layout)
        
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):

        flag_show = self.activate_segment.box.currentText()

        _, img_crop, img_extract = self.preprocessing.preprocess_pipeline(cv_img)
        # print(img_extract.shape)
        mask, prob_map = self.model_segment.segment(img_extract, 
                                                    k=[0.1, 0.1, 0, 0.3, 0.4, 0.3], 
                                                    threshold=0.35, 
                                                    filter=True)

        if flag_show == 'Transform': 
            img_show = img_crop

        elif flag_show == 'Mask':
            img_show = np.repeat((mask*255).astype(np.uint8), 3, axis=1).reshape(img_crop.shape)

        elif flag_show == 'Probability Map':
            img_show = np.repeat((prob_map*255).astype(np.uint8), 3, axis=1).reshape(img_crop.shape)

        elif flag_show == 'Apply Mask':
            mask_repeat = np.repeat((mask*255).astype(np.uint8), 3, axis=1).reshape(img_crop.shape)
            img_show = cv2.bitwise_and(img_crop, mask_repeat)

        img_show = np.pad(img_show, ((self.preprocessing.ROI_up, cv_img.shape[0]-self.preprocessing.ROI_down), 
                                            (self.preprocessing.ROI_left, cv_img.shape[1]-self.preprocessing.ROI_right),
                                            (0, 0)), 'constant')

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
                                     "reference_point_x": {"min": 0, "max": half_w, "type":"slide"},
                                     "reference_point_y": {"min": 0, "max": half_w, "type":"slide"},
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
                                     "reference_point_x": image_parameter['reference_point_x'],
                                     "reference_point_y": image_parameter['reference_point_y'],
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
                widget = NumberBox(parameter, 'integer', min, max, str(value))
                widget.box.textChanged.connect(self.update_image_transformation)
                widget.box.setReadOnly(True)

            widget.addWidget(layout, i)
            self.image_parameter_widgets.append(widget)

        save_botton = QPushButton('save image parameter')
        save_botton.clicked.connect(self.save_image_parameter_on_click)
        layout.addWidget(save_botton, 13, 0)

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
            else: self.image_parameter_value[parameter] = int(self.image_parameter_widgets[i].box.text())
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