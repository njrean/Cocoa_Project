import sys
sys.path.append('/home/admin01/Github/Cocoa_Project/cocoa_machine_v1')
from lib.Bayesian_Segmentation import *

annotation_file_path = '/home/admin01/Github/Cocoa_Project/cocoa_machine_v1/file_creation/dataset/annotation/annotation_test5.json'
raw_image_folder = '/home/admin01/Github/Cocoa_Project/cocoa_machine_v1/file_creation/dataset/image/'
save_segmentation_model_path = '/home/admin01/Github/Cocoa_Project/cocoa_machine_v1/lib/config/'
model_name = 'model_segmentation.pkl'

name_features = ['B', 'G', 'R', 'Y', 'Cr', 'Cb']
roi = [280, 330, 144, 144]

model = Bayesian_Segmentation(n_features=6, name_features=name_features)
model.fit(raw_image_folder, annotation_file_path, RoI=roi, mode='combine')

save_bayesian_obj(save_segmentation_model_path+model_name, model)