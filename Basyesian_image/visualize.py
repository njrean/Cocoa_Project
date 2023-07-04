import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time
import os
import bayesian_lib

os.chdir(r"C:\Users\natch\Github\Cocoa_Project")

annotation_file_path = 'dataset/annotation/annotation_test3.json'
raw_image_folder = 'dataset/raw/'
name_features = ['B', 'G', 'R']

BS_model_rgb = bayesian_lib.Bayesian(name_features=name_features)
BS_model_rgb.fit(raw_image_folder, annotation_file_path, RoI=[200, 200, 115, 115], mode='RGB')
BS_model_rgb.plot_features_distribution(attr=['cocoa', 'shadow'], mode='3d', elev=45, azim=-45, roll=0)