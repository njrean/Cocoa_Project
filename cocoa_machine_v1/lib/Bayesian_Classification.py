import numpy as np

#Bayesain Classification
class Bayesian_Classsification():
    def __init__(self) -> None:

        self.model = {'L': {'mean': 9700.23, 'std': 787.95},
                      'M': {'mean': 7600.28, 'std': 560.41},
                      'S': {'mean': 6500.15, 'std': 476.62},
                      'Out': {'mean': 3560.47, 'std': 727.66}}
        
        self.index2label = {0: 'L',
                            1:'M',
                            2:'S',
                            3:'Out'}
        
        self.update_model()
        self.data = []

    def fit(self):
        pass

    def update_model(self):
        self.n_class = len(self.model)
        self.mean_mat = np.array([self.model[cl]['mean'] for cl in list(self.model.keys())])
        self.std_mat = np.array([self.model[cl]['std'] for cl in list(self.model.keys())])

    def predict(self, sep_masks):
        #input => separate mask (numpy array) shape (number of bean, h_mask, w_mak) mask value is 0 or 1 
        #order left to right
        #output => class right to left bean

        sum_masks = np.array([np.sum(mask, axis=(0,1)) for mask in sep_masks]) #sum area and not reverse right left to right order
        prob_mat = (np.e**((((sum_masks[None, :] - self.mean_mat[:, None])/self.std_mat[:,None])**2)*(-0.5))).T
        select_mat = np.argmax(prob_mat, axis=1)

        return prob_mat, select_mat

    def load_obj(self):
        pass

    def save_obj(self):
        pass
