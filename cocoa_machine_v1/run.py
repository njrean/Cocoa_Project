import os
import sys
import cv2
import numpy as np
import time

from PyQt5.QtCore import QLibraryInfo
from PyQt5.QtWidgets import QApplication

from lib.Camera import Camera
from lib.Proprocessing import Preprocessing
from lib.Bayesian_Segmentation import Bayesian_Segmentation, load_bayesian_obj
from lib.UI import MainWindow, ConfigWindow

camera_prof_path = './lib/config/camera_profile.yaml'
image_config_path = './lib/config/image_config.yaml'
model_parameter_path = './lib/config/model_parameter.yaml'
model_segmentation_path = './lib/config/model_segmentation.pkl'
model_classification_path = './lib/config/model_classification.pkl'

if __name__ == "__main__":

    ##Preprocessing obj
    prep = Preprocessing(image_config_path=image_config_path)

    ##Camera obj
    webcam = Camera(camera_idx=4, img_w=1280, img_h=720, camera_prof_path=camera_prof_path)

    ##Segmentation obj
    model_segmentation = Bayesian_Segmentation(parameter_path=model_parameter_path)
    model_segmentation = load_bayesian_obj(model_segmentation_path)

    ##GUI
    app = QApplication(sys.argv)
    a = MainWindow(webcam, prep, model_segmentation)
    a.show()
    sys.exit(app.exec())
