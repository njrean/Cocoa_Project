import numpy as np
import cv2
import subprocess

from lib.function import read_yaml, save_yaml

class Camera():

    def __init__(self, camera_idx:int, img_w:int, img_h:int, camera_prof_path:str):
        
        self.camera_idx = camera_idx
        self.cap = None                             #cv2 object for video capture
        
        self.w = img_w                              #wide of capture image
        self.h = img_h                              #height of capture image

        self.camera_prof_path = camera_prof_path    #path to load profile
        self.camera_prof = dict()                   #dictionary of camera profile
        self.camera_info = dict()

        self.camera_matrix = np.zeros((3,3))
        self.coeff = np.zeros((1,5))

    def get_camera_info(self):
        info = subprocess.run(['v4l2-ctl -d /dev/video{} --list-ctrls'.format(self.camera_idx)], 
                              shell=True, capture_output=True, text=True)
        
        list_line = info.stdout.split("\n")
        list_line.remove("User Controls")
        list_line.remove("Camera Controls")
        list_line = list(filter(None, list_line))

        camera_prof = dict()

        for config in list_line:
            config_info = config.strip().split(" ")
            config_info.pop(1)
            config_info.remove(":")
            config_info = list(filter(None, config_info))
            self.camera_info[config_info[0]] = dict()
            self.camera_info[config_info[0]]['type'] = config_info[1]
            for characteristic in config_info[2:]:
                if not(any(i.isdigit() for i in characteristic)) or any(i=='(' for i in characteristic): 
                    break
                ls = characteristic.split("=")
                self.camera_info[config_info[0]][ls[0]] = ls[1]
                if ls[0] == 'value':
                    camera_prof[config_info[0]] = ls[1]

        self.camera_prof = camera_prof

        return self.camera_info
    
    def update_camera_profile(self, camera_prof):
        self.camera_prof = camera_prof
        for config in camera_prof:
            if type(camera_prof[config]) != None :
                subprocess.call(['v4l2-ctl -d /dev/video{} -c {}={}'.format(self.camera_idx, 
                                                                            config, 
                                                                            str(camera_prof[config]))], 
                                                                            shell=True)

    def set_camera_profile(self, new_path=None):
        #add new path then set new path to being a camera profile path
        if new_path != None:
            self.camera_prof_path = new_path

        self.camera_prof = read_yaml(self.camera_prof_path)

        #run cmd to set camera parameter
        self.update_camera_profile(self.camera_prof)

    def save_camera_prof(self):
        save_yaml(self.camera_prof, self.camera_prof_path)
    
    def camera_start(self):
        api_preference = cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(self.camera_idx, api_preference)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

    def capture(self):
        flag, frame = self.cap.read()
        return flag, frame

    def camera_stop(self):
        self.cap.release()
    
"""
FUNCTION for UNDISTORT (fix intrinsic parameter) 
=> need to consider again because it would be not effect to image

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
        cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_V4L2)
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
"""