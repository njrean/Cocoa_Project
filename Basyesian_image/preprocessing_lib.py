import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml
import math

def lowpass_kernel(layer:int):
        kernel = [0,1,2,1,0]
    
        for i in range(layer - 3):
            new_kernel = [0]
            for j in range(1, len(kernel)):
                new_kernel.append(kernel[j-1]+kernel[j])

            new_kernel.append(0)
            kernel = new_kernel
        
        kernel = np.array(kernel[1:-1])
        kernel = kernel/np.sum(kernel)
            
        return kernel

class Prep_cocoa():
    def __init__(self, low_pass_length):
        self.low_pass_kernel = lowpass_kernel(low_pass_length)
        self.derivative_kernel = np.array([10, 0, -10])
        self.value_start = 0.04
        self.half_stop = -0.06
    
    def crop(self, mask, RoI=[], Horizon=False):
        offset = 0

        if RoI != []:
            mask = mask[RoI[0]:-RoI[1], RoI[2]:-RoI[3]]
            offset = 0

        if Horizon == True:
            mask = mask.T

        mean_vector = np.mean(mask, axis=1)
        derivative_kernel = np.array([10, 0, -10])
        low_pass_vector = np.convolve(mean_vector, self.low_pass_kernel, 'same')
        dif_vector = np.convolve(low_pass_vector, derivative_kernel, 'same')

        start_idx = []
        stop_idx = []

        flag = 0
        v_old = 0

        # for i, v in enumerate(dif_vector):
        #     if flag == 0 and v >= self.value_start:
        #         start_idx.append(i+offset)
        #         flag = 1

        #     elif flag == 1 and v < self.half_stop and v > v_old:
        #         flag = 2

        #     elif flag == 2 and v >= -0.0005:
        #         stop_idx.append(i+offset)
        #         flag = 0

        #     v_old = v

        for i, v in enumerate(low_pass_vector):
            if flag == 0 and v >= self.value_start:
                start_idx.append(i+offset)
                flag = 1

            elif flag == 1 and v < v_old:
                flag = 2

            elif flag == 2 and v <= self.value_start:
                stop_idx.append(i+offset)
                flag = 0

            v_old = v
        
        start_idx = start_idx[:len(stop_idx)]
        bound = np.array([start_idx, stop_idx]).T
        sep_mask = []

        for i, b in enumerate(bound):
            sep_mask.append(mask[b[0]: b[1],:])
       
        return np.array(sep_mask, dtype=object)
    
    def plot(self, image, mask, RoI=[], Rot=False):

        if RoI != []:
            mask = mask[RoI[0]:-RoI[1], RoI[2]:-RoI[3]]
            image = image [RoI[0]:-RoI[1], RoI[2]:-RoI[3]]

        if Rot == True:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: mask= mask.T

        mean_vector = np.mean(mask, axis=1)
        low_pass_vector = np.convolve(mean_vector, self.low_pass_kernel, 'same')
        dif_vector = np.convolve(low_pass_vector, self.derivative_kernel, 'same')

        fig, ax = plt.subplots(2,1)
        ax[0].imshow(image)
        ax[1].plot(mean_vector, label='mean')
        ax[1].plot(low_pass_vector, label='low pass')
        ax[1].plot(dif_vector, label='derivative')
        ax[1].legend()
        ax[1].set_xlim(xmin=0, xmax=image.shape[1])
        ax[1].set_xlabel("postion")
        ax[1].set_ylabel("value")

        plt.show()

class Camera():
    def __init__(self, camera_idx:int, img_w:int, img_h:int):
        self.camera_idx = camera_idx
        self.camera_matrix = np.zeros((3,3))
        self.coeff = np.zeros((1,5))
        self.w = img_w
        self.h = img_h

    def camera_calculate_intrinsic_param(self, r:int, c:int, n_data=10, save_file_path='./parameter/camera_calibration.yaml'):
        #initial needed variable

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((r*c,3), np.float32)
        objp[:,:2] = np.mgrid[0:r,0:c].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #Run video for collecting data
        cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

        found = 0

        while(True):
            ret, img = cap.read()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            check, corners = cv2.findChessboardCorners(img, (r,c), None)

            # If found, add object points, image points (after refining them)
            if check == True:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (r,c), corners2, check)
                
                if found < n_data and cv2.waitKey(30) & 0xFF == ord('c'):
                    objpoints.append(objp)   # Certainly, every loop objp is the same, in 3D.
                    imgpoints.append(corners2)
                    found += 1
                    print(found)

            cv2.imshow('Chessboard Finding', img)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print('Sucessful Calculate Intrinsic Parameter')
        self.save_intrinsic_param(mtx, dist, save_file_path)

    def Test(self):
        cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self. h)

        while(True):
            _, img = cap.read()

            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.coeff, (self.w, self.h), 1, (self.w, self.h))
            undist = cv2.undistort(img, self.camera_matrix, self.coeff, None, newcameramtx)
            stack = np.concatenate((img, undist), axis=1)
            stack = cv2.resize(stack, dsize=(1280, 360), interpolation=cv2.INTER_LINEAR)

            cv2.imshow('Test undistortion', stack)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()

    def undistort(self, image):
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.coeff, (self.w, self.h), 1, (self.w, self.h))
        undist = cv2.undistort(image, self.camera_matrix, self.coeff, None, newcameramtx)
        return undist

    def save_intrinsic_param(self, matrix, coeff, file_path):
        data = {'camera_matrix': matrix.tolist(),
                'dist_coeff': coeff.tolist()}
        with open(file_path, "w") as f:
            yaml.dump(data, f)
        f.close()

    def load_intrinsic_param(self, parameter_file_path):
        with open(parameter_file_path, 'r') as f:
            loadeddict = yaml.safe_load(f)
        self.camera_matrix = np.array(loadeddict.get('camera_matrix'))
        self.coeff = np.array(loadeddict.get('dist_coeff'))
        f.close()
