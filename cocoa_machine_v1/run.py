import sys
import numpy as np
import heapq
from threading import Timer

from PyQt5.QtWidgets import QApplication

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


# def call(i):
#     print('Hi', i)

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
    # scheduler = Scheduler(prep.ref_point_x)

    ##GUI
    app = QApplication(sys.argv)
    a = MainWindow(webcam, 
                   prep, 
                   model_segmentation, 
                   model_classification, 
                   centroidTracker)
    
    # print(threading.active_count())
    a.show()
    sys.exit(app.exec())

# li = []
# timer1 = Timer(5, call, args=[5])
# timer2 = Timer(2, call, args=[2])
# timer1.start()
# timer2.start()
# timer3 = Timer(3, call, args=[3])
# timer3.start()

# heapq.heappush(li, 5)
# heapq.heappush(li, 2)
# heapq.heappush(li, 3)
# heapq.heappush(li, 4)
# heapq.heappush(li, 1)
# heapq.heappush(li, 6)
# li = [heapq.heappop(li) for i in range(len(li))]
# print(li)
# del li[1:3]
# print(li)
# timer2.join()
# if li[1][2].is_alive():
#     li[1][2].cancel()

# print(li)

# print(heapq.heappop(li))
# x = np.array([0.8, 1.2, 1.3, 1.6, 1.8])
# x = np.triu((x[:, np.newaxis] - x).T)
# print(x)
# # x = np.cumsum(x, axis=1)

# ls = []
# exist = []

# for i in range(len(x)-1):
#     find = x[i][i+1:] <= 0.26
#     print(find)
#     if any(find):
#         stop = np.max(np.where(find)[0])+i+1
#         if not(stop in exist):
#             ls.append((i, stop))
#             exist += list(range(i, stop+1))
        
# print(ls)
# print(exist)