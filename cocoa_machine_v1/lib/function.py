import yaml
from yaml.loader import SafeLoader
import numpy as np

def read_yaml(file_path):
    with open(file_path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    f.close()
    return data

def save_yaml(data, file_path):
     with open(file_path, 'w') as f:
        yaml.dump(data, f)

def lowpass_kernel(layer:int):
        kernel = [0, 1, 2, 1, 0]
    
        for _ in range(layer - 3):
            new_kernel = [0]
            for j in range(1, len(kernel)):
                new_kernel.append(kernel[j-1] + kernel[j])

            new_kernel.append(0)
            kernel = new_kernel
        
        kernel = np.array(kernel[1 : -1])
        kernel = kernel / np.sum(kernel)
            
        return kernel

lowpass_filter = lowpass_kernel(500)

def crop_bean(mask, value_consider = 0.04):

        mask = mask.T

        mean_vector = np.mean(mask, axis=1)
        low_pass_vector = np.convolve(mean_vector, lowpass_filter, 'same')

        start_idx = []
        stop_idx = []
        
        flag_found = 0
        flag = 0
        v_old = 0

        for i, v in enumerate(low_pass_vector):
            
            if flag == 0 and v >= value_consider:
                start_idx.append(i)
                flag = 1

            elif flag == 1 and v < v_old:
                flag = 2

            elif flag == 2 and v <= value_consider:
                stop_idx.append(i)
                flag_found = 1
                flag = 0

            v_old = v
        
        start_idx = start_idx[:len(stop_idx)]
        bound = np.array([start_idx, stop_idx]).T
        sep_mask = []

        for i, b in enumerate(bound):
            sep_mask.append(mask[b[0]: b[1], :])

        return low_pass_vector, bound, np.array(sep_mask, dtype=object), flag_found

#Find centroid and conner of a bean
def find_centroid_conner(bean_mask):
    count = np.asarray(bean_mask >= 1).nonzero()
    xmin = np.min(count[1])
    xmax = np.max(count[1])
    ymin = np.min(count[0])
    ymax = np.max(count[0])
    cen_x = np.mean(count[1])
    cen_y = np.mean(count[0])

    return np.array([xmin, xmax, ymin, ymax]), np.round(np.array([cen_x, cen_y]))
    

# def draw_box_centroid(img, bound, ):

#     pass