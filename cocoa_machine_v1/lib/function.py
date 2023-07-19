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
value_start = 0.04

def crop(mask, RoI=[], Horizon=False):
        offset = 0

        if RoI != []:
            mask = mask[RoI[0]:-RoI[1], RoI[2]:-RoI[3]]
            offset = 0

        if Horizon == True:
            mask = mask.T

        mean_vector = np.mean(mask, axis=1)
        low_pass_vector = np.convolve(mean_vector, lowpass_filter, 'same')

        start_idx = []
        stop_idx = []

        flag = 0
        v_old = 0

        for i, v in enumerate(low_pass_vector):
            if flag == 0 and v >= value_start:
                start_idx.append(i+offset)
                flag = 1

            elif flag == 1 and v < v_old:
                flag = 2

            elif flag == 2 and v <= value_start:
                stop_idx.append(i+offset)
                flag = 0

            v_old = v
        
        start_idx = start_idx[:len(stop_idx)]
        bound = np.array([start_idx, stop_idx]).T
        sep_mask = []

        for i, b in enumerate(bound):
            sep_mask.append(mask[b[0]: b[1], :])
       
        return np.array(sep_mask, dtype=object)