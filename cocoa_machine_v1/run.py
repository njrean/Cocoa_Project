import sys
import numpy as np

from PyQt5.QtWidgets import QApplication
import multiprocessing

from lib.Camera import Camera
from lib.Proprocessing import Preprocessing
from lib.Bayesian_Segmentation import Bayesian_Segmentation, load_bayesian_obj
from lib.Bayesian_Classification import Bayesian_Classsification
from lib.Scheduler import Scheduler
from lib.UI import MainWindow
from lib.Tracker import Tracker

camera_prof_path = './lib/config/camera_profile.yaml'
image_config_path = './lib/config/image_config.yaml'
model_parameter_path = './lib/config/model_parameter.yaml'
model_segmentation_path = './lib/config/model_segmentation.pkl'
model_classification_path = './lib/config/model_classification.pkl'

if __name__ == "__main__":

    ##Preprocessing obj
    prep = Preprocessing(image_config_path=image_config_path)

    ##Camera obj
    webcam = Camera(camera_idx=4, 
                    img_w=prep.original_image_w, 
                    img_h=prep.original_image_h, 
                    camera_prof_path=camera_prof_path)

    ##Segmentation obj
    model_segmentation = Bayesian_Segmentation(parameter_path=model_parameter_path)
    model_segmentation = load_bayesian_obj(model_segmentation_path)

    ##Classification obj
    model_classification = Bayesian_Classsification()

    ##Tracker
    centroidTracker = Tracker()

    #Scheduler
    scheduler = Scheduler(prep.ref_point_x)

    ##GUI
    app = QApplication(sys.argv)
    a = MainWindow(webcam, 
                   prep, 
                   model_segmentation, 
                   model_classification, 
                   centroidTracker,
                   scheduler)
#     a.show()
#     sys.exit(app.exec())

# while(1):
#     scheduler = Scheduler(100)
#     value = input("Type integer 0, 1, 2, 3")
#     scheduler.buffer_test = int(value)
#     scheduler.calculate_data_send()
#     scheduler.send_UART()