import cv2
import os
import sys
sys.path.append('/home/admin01/Github/Cocoa_Project/cocoa_machine_v1')
from lib.Camera import *

camera_idx = 4
h = 1920
w = 1280
camera_prof_path = '/home/admin01/Github/Cocoa_Project/cocoa_machine_v1/lib/config/camera_profile.yaml'

image_id = 0
save_directory = '/home/admin01/Github/Cocoa_Project/cocoa_machine_v1/file_creation/dataset/image'

center_point_activate = 0

webcam = Camera(camera_idx=camera_idx, img_w=w, img_h=h, camera_prof_path=camera_prof_path)
webcam.set_camera_profile()
webcam.camera_start()

while(True):
    webcam.set_camera_profile()
    _, frame = webcam.capture()
    img_show = frame

    if center_point_activate == 1:
        img_show[:,:,:] = (0, 0, 255)

    cv2.imshow("Capturing", img_show)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        cv2.destroyWindow("Capturing")
        break
    
    elif cv2.waitKey(50) & 0xFF == ord('c'):
        file_name = save_directory+'/{}{}{}'.format('collect1_', image_id, '.png')
        print('Saved!!!', file_name)
        cv2.imwrite(file_name, frame)
        image_id += 1

    elif cv2.waitKey(50) & 0xFF == ord('o'):
        center_point_activate += 1
        center_point_activate %= 2

webcam.camera_stop()