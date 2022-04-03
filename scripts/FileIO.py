import os
from re import sub
import pickle
class FileIO:
    def __init__(self):
        self.data_path = os.path.dirname(__file__)+'/../data/'
        self.subjects = [i.split('.')[0] for i in os.listdir(self.data_path)]

    def get_experiments(self,subjecti):
        return os.listdir(os.path.join(self.data_path,subjecti))
    
    def load_experiment(self,subjecti):
        return pickle.load(open(os.path.join(self.data_path,subjecti+'.pkl'),'rb'))
    
    # def get_emg_data(self,subjecti):
    #     experiment = pickle.load(open(os.path.join(self.data_path,subjecti+'.pkl'),'rb'))
    #     return experiment['emg']
