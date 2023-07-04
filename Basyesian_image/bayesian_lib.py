import json
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

#Bayesain Segmentation for binary class
class Bayesian():
    def __init__(self, n_features=3, name_features=['Blue channel', 'Green channel', 'Red channel']):
        self.n_features = n_features
        self.name_features = name_features

        self.model_cc = np.zeros((n_features, 2))
        self.model_bg = np.zeros((n_features, 2))

        self.n_cc_samples = 0
        self.n_bg_samples = 0
        self.n_sh_samples = 0
        self.n_sample = 0

        self.cc_samples = [[] for i in range(n_features)]
        self.bg_samples = [[] for i in range(n_features)]
        self.sh_samples = [[] for i in range(n_features)]

        self.cc_value_collect = []
        self.bg_value_collect = []
        self.sh_value_collect = [] #collect shadow pixel sample

        self.P_cocoa = 0
        self.P_bg = 0

        self.mode = ''

        self.n_image_sample = 0

        self.kernel = (1/256)*(np.array([[1, 4, 6, 4, 1],
                                        [4, 16, 24, 16, 4],
                                        [6, 24, 36, 24, 6],
                                        [4, 16, 24, 16, 4],
                                        [1, 4, 6, 4, 1]]))

    def print_detail(self):
        print('Detail')
        print('------------------------------------------------')
        print('Mode: {}'.format(self.mode))
        print('Number of images: {}'.format(self.n_image_sample))
        print('Number of Feature: {} -> {}'.format(self.n_features, self.name_features))
        print('------------------------------------------------')
        print('Probability of Cocoa:\t\t{}'.format(self.P_cocoa))
        print('Probability of Back ground:\t{}'.format(self.P_bg))
        print('------------------------------------------------')
        print('Number of cocao sample:\t\t{}'.format(self.n_cc_samples))
        print('Number of background sample:\t{}'.format(self.n_bg_samples))
        print('Number of shadow sample:\t{}'.format(self.n_sh_samples))
        print('Number of fit sample:\t\t{}'.format(self.n_sample))
        print('------------------------------------------------')
        print('Model Background')

        for i in range(self.n_features):
            print('\t{} \tmean: {} \tstd: {}'.format(self.name_features[i], self.model_bg[i][0], self.model_bg[i][1]))
        print('------------------------------------------------')
        print('Model Cocoa')
        for i in range(self.n_features):
            print('\t{} \tmean: {} \tstd: {}'.format(self.name_features[i], self.model_cc[i][0], self.model_cc[i][1]))
       
    def fit(self, image_folder_path, annotation_file_path, RoI=[], mode='RGB'):
        #prepare annotation
        annotation_file = open(annotation_file_path)
        annotation_data = json.load(annotation_file)
        annotation_file.close()

        img_names = [annotation_data[i]['filename'] for i in list(annotation_data.keys())]
        object_names = list(annotation_data.keys())

        P_cc = []
        self.mode = mode
        self.n_image_sample = 0

        #loop image to collect sample pixels
        for i, im_name in enumerate(img_names):

            self.n_image_sample += 1

            raw_image = cv2.imread(image_folder_path+im_name) 

            h = raw_image.shape[0]
            w = raw_image.shape[1]

            if mode == 'YCrCb': #default is BGR
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2YCR_CB) #Read as YCR_CB
            
            elif mode == 'combine':
                ycbcr_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2YCR_CB)
                buffer = np.zeros((h, w, raw_image.shape[2]+ycbcr_image.shape[2]))
                buffer[:, :, :raw_image.shape[2]] = raw_image
                buffer[:, :, raw_image.shape[2]:raw_image.shape[2]+ycbcr_image.shape[2]] = ycbcr_image
                raw_image = buffer

            labels_cc = np.zeros((h, w), dtype=np.bool_)
            labels_sh = np.zeros((h, w), dtype=np.bool_)
            regions = annotation_data[object_names[i]]['regions']

            for r in regions:
                if r['region_attributes']['object'] == 'cocoa':
                    cc = np.zeros((h, w))
                    x_points = r['shape_attributes']['all_points_x']
                    y_points = r['shape_attributes']['all_points_y']
                    contour = np.vstack((x_points, y_points)).T
                    cv2.fillPoly(cc, pts = [contour], color =[255])
                    labels_cc = np.logical_or(labels_cc, cc)

                elif r['region_attributes']['object'] == 'shadow':
                    sh = np.zeros((h, w))
                    x_points = r['shape_attributes']['all_points_x']
                    y_points = r['shape_attributes']['all_points_y']
                    contour = np.vstack((x_points, y_points)).T
                    cv2.fillPoly(sh, pts = [contour], color =[255])
                    labels_sh = np.logical_or(labels_sh, sh)

            #crop RoI
            if RoI != []:
                over = RoI[0]
                below = RoI[1]
                left = RoI[2]
                right = RoI[3]
                raw_image = raw_image[over:-below, left:-right, :]
                labels_cc = labels_cc[over:-below, left:-right]
                labels_sh = labels_sh[over:-below, left:-right]

            label_bg = np.invert(labels_cc)

            for f in range(self.n_features):
                self.cc_samples[f].append((raw_image[:, :, f]/255)[labels_cc])
                self.sh_samples[f].append((raw_image[:, :, f]/255)[labels_sh])
                self.bg_samples[f].append((raw_image[:, :, f]/255)[label_bg])
            
            #find probability of cocoa (P(cocoa))
            random_pixel = np.random.choice(labels_cc.flatten(), math.floor(labels_cc.shape[0]*labels_cc.shape[1]*0.5))
            P_cc.append(np.sum(random_pixel)/(labels_cc.shape[0]*labels_cc.shape[1]*0.5))

        self.n_cc_samples = len([v for array in self.cc_samples[0] for v in array])
        self.n_bg_samples = len([v for array in self.bg_samples[0] for v in array])
        self.n_sh_samples = len([v for array in self.sh_samples[0] for v in array])
        self.n_sample = min(self.n_cc_samples, self.n_bg_samples)

        self.cc_value_collect = np.zeros((self.n_features, self.n_cc_samples))
        self.bg_value_collect = np.zeros((self.n_features, self.n_bg_samples))
        self.sh_value_collect = np.zeros((self.n_features, self.n_sh_samples))

        #flatten list and change into numpy array
        for f in range(self.n_features):
            self.cc_value_collect[f] = np.array([v for array in self.cc_samples[f] for v in array])
            self.bg_value_collect[f] = np.array([v for array in self.bg_samples[f] for v in array])
            self.sh_value_collect[f] = np.array([v for array in self.sh_samples[f] for v in array])

        #resample 
        if self.n_bg_samples > self.n_sample:
            self.bg_value_collect =  self.bg_value_collect[:, np.random.choice(self.bg_value_collect.shape[1], self.n_sample, replace=False)]
        else:
            self.cc_value_collect =  self.cc_value_collect[:, np.random.choice(self.cc_value_collect.shape[1], self.n_sample, replace=False)]
            
        #calculate mean and std
        for f in range(self.n_features):
            self.model_cc[f][0] = np.mean(self.cc_value_collect[f])
            self.model_cc[f][1] = np.std(self.cc_value_collect[f])

            self.model_bg[f][0] = np.mean(self.bg_value_collect[f])
            self.model_bg[f][1] = np.std(self.bg_value_collect[f])

        self.P_cocoa = np.mean(np.array(P_cc))
        self.P_bg = 1 - self.P_cocoa

        #print detail
        print("Complete Fit data -------------")
        self.print_detail()

    def predict(self, image_arr, k, threshold=0.9, RoI=[], filter=False):

        if RoI != []:
            image_arr = image_arr[RoI[0]:-RoI[1], RoI[2]:-RoI[3], :]

        h = image_arr.shape[0]
        w = image_arr.shape[1]

        f_cc = (image_arr/255)-np.tile(self.model_cc[:, 0], h*w).reshape(h, w, self.n_features)
        f_cc = ((f_cc/np.tile(self.model_cc[:, 1], h*w).reshape(h, w, self.n_features))**2)*(-0.5)
        f_cc = math.e**(f_cc)
        
        f_bg = (image_arr/255)-np.tile(self.model_bg[:, 0], h*w).reshape(h, w, self.n_features)
        f_bg = ((f_bg/np.tile(self.model_bg[:, 1], h*w).reshape(h, w, self.n_features))**2)*(-0.5)
        f_bg = math.e**(f_bg)

        f = 1/(1+(((1-self.P_cocoa)*f_bg)/(self.P_cocoa*f_cc)))

        prob_map = np.sum(f * np.tile(np.array(k), h*w).reshape(h, w, self.n_features), axis=2)
        
        if filter==True:
            prob_map = cv2.filter2D(src=prob_map, ddepth=-1, kernel=self.kernel)

        if RoI != []:
            prob_map = np.pad(prob_map, [(RoI[0], RoI[1]), (RoI[2], RoI[3])], mode='constant', constant_values=0)
        
        mask = np.where( prob_map > threshold, 1, 0)

        return mask, prob_map

    def plot_features_distribution(self, attr=['bg', 'cocoa', 'shadow'], mode='hist', elev=0, azim=0, roll=0):
        labels = []
        data = []

        if 'bg' in attr:
            labels.append('bg')
            data.append(self.bg_value_collect)

        if 'cocoa' in attr:
            labels.append('cocoa')
            cc_data =  self.cc_value_collect[:, np.random.choice(self.cc_value_collect.shape[1], math.floor(self.n_sample*0.2), replace=False)]
            data.append(cc_data)

        if 'shadow' in attr:
            labels.append('shadow')
            data.append(self.sh_value_collect)

        if self.n_cc_samples != 0:
            
            if mode == 'hist':
                fig, ax = plt.subplots(1, self.n_features, figsize=(4*self.n_features, 4))
                bins = np.arange(0, 1, 0.015) # fixed bin size
                color = np.reshape(plt.cm.rainbow(np.linspace(0, 1, self.n_features*len(attr))), (self.n_features, len(attr), 4))
                
                for f, c in zip(range(self.n_features), color):
                    for i in range(len(attr)):
                            ax[f].hist(data[i][f], bins=bins, color=c[i], alpha=0.4, label=labels[i])

                    ax[f].legend()
                    ax[f].set_title('{} {}'.format('Feature', self.name_features[f]))
                    ax[f].set_xlabel('Value')
                    ax[f].set_ylabel('count')

            elif mode == '3d':
                plt.figure(figsize=(10,10))
                ax = plt.axes(projection='3d')
                for d in range(len(attr)):
                    ax.scatter3D(data[d][0], data[d][1], data[d][2], label=labels[d])
                ax.legend()

                if self.mode == 'RGB':
                    ax.set_xlabel('B')
                    ax.set_ylabel('G')
                    ax.set_zlabel('R')

                elif self.mode == 'YCrCb':
                    ax.set_xlabel('Y')
                    ax.set_ylabel('Cr')
                    ax.set_zlabel('Cb')

                ax.set_title('Plot {} 3D'.format(self.mode))
                # ax.view_init(elev, azim, roll)

            plt.show()
            
        else: print('Not it data yet')

    def load_obj():
        pass

    def save_obj():
        pass