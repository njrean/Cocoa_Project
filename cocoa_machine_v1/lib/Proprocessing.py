import numpy as np
import cv2
import math

from lib.function import read_yaml, save_yaml

class Preprocessing():
    def __init__(self, image_config_path):
        self.image_config_path = image_config_path
        self.update_preset()

#Function update preset
    def update_preset(self):
        self.data = read_yaml(self.image_config_path)

        self.original_image_h = self.data['original_image_h']
        self.original_image_w = self.data['original_image_w']

        self.rotz = self.data['Rotation_z']
        self.angle = self.rotz/10
        self.transx = self.data['Translation_x']
        self.transy = self.data['Translation_y']

        self.ROI_up = self.data['Boundary_UP']
        self.ROI_down = self.data['Boundary_DOWN']
        self.ROI_left = self.data['Boundary_LEFT']
        self.ROI_right = self.data['Boundary_RIGHT']

        self.ref_point1_x = self.data['reference_point1_x']
        self.ref_point1_y = self.data['reference_point1_y']
        self.ref_point2_x = self.data['reference_point2_x']
        self.ref_point2_y = self.data['reference_point2_y']
        self.unit = self.data['unit_transition']
        
        self.center_x = math.floor(self.original_image_w/2)
        self.center_y = math.floor(self.original_image_h/2)
        self.affine_matrix_preset = self.create_affine_matrix(self.angle, self.transx, self.transy)


#Function create affine matric from angle and offset
    def create_affine_matrix(self, rotz, transx, transy):
        cosz = np.cos(np.radians(rotz))
        sinz = np.sin(np.radians(rotz))
        a =  self.center_x * (1 - cosz) - (self.center_y * sinz)
        b =  self.center_y * (1 - cosz) + (self.center_x * sinz)
        R = np.array([[cosz, sinz, a], [-sinz, cosz, b], [0, 0, 1]])
        T = np.array([[1, 0, transx],[0, 1, transy], [0, 0, 1]])
        return (T @ R)[:2,:]

#Function transform image from angle and offset
    def transform(self, img, rotz, transx, transy):
        affine_matrix = self.create_affine_matrix(rotz, transx, transy)
        return cv2.warpAffine(img, affine_matrix, 
                              (self.original_image_w, self.original_image_h))

#Function transform image from presset
    def transform_from_preset(self, img):     
        #cv2.warpAffine(image, Affine_matrix, (w, h))
        return cv2.warpAffine(img, self.affine_matrix_preset,
                              (self.original_image_w, self.original_image_h))

#Function crop image from RoI
    def crop(self, img, ROI_up, ROI_down, ROI_left, ROI_right):
        return img[ROI_up:ROI_down, ROI_left:ROI_right, :]

#Extract color from BGR image to BGR+YCbCr image
    def extract_color(self, img_bgr):
        #input is numpy array shape(w, h, 3) BGR model (order => B G R)
        #output is numpy array shape(w, h, 6) BGR+YCbCr model (order => B G R Y Cb Cr)
        
        img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
        image_stack = np.zeros((img_bgr.shape[0], img_bgr.shape[1], 6))
        image_stack[:,:,:3] = img_bgr
        image_stack[:,:,3:6] = img_ycbcr

        return image_stack

#Combine preprocessing step by preset
    def preprocess_pipeline(self, img):
        #preprocessing sequence 1. transformation (rotation+translation)
        #                       2. crop by Region of Interest (RoI)
        #                       3. extract color model

        img_trans = self.transform_from_preset(img)
        img_crop = self.crop(img_trans, self.ROI_up, self.ROI_down, self.ROI_left, self.ROI_right)
        img_extract = self.extract_color(img_crop)

        return img_trans, img_crop, img_extract
    
    def apply_mask(self, img, mask):
        mask_reapeat = np.repeat((mask*255).astype(np.uint8), 3, axis=1).reshape(img.shape)
        return cv2.bitwise_and(img, mask_reapeat)
    
    def update_parameter(self):
        self.original_image_h = self.data['original_image_h']
        self.original_image_w = self.data['original_image_w']

        self.rotz = self.data['Rotation_z']
        self.angle = self.rotz/10
        self.transx = self.data['Translation_x']
        self.transy = self.data['Translation_y']

        self.ROI_up = self.data['Boundary_UP']
        self.ROI_down = self.data['Boundary_DOWN']
        self.ROI_left = self.data['Boundary_LEFT']
        self.ROI_right = self.data['Boundary_RIGHT']

        self.ref_point1_x = self.data['reference_point1_x']
        self.ref_point1_y = self.data['reference_point1_y']
        self.ref_point2_x = self.data['reference_point2_x']
        self.ref_point2_y = self.data['reference_point2_y']
        self.unit = self.data['unit_transition']
        
        self.center_x = math.floor(self.original_image_w/2)
        self.center_y = math.floor(self.original_image_h/2)
        self.affine_matrix_preset = self.create_affine_matrix(self.angle, self.transx, self.transy)
    
    def save_image_parameter(self):
        save_yaml(self.data, self.image_config_path)
