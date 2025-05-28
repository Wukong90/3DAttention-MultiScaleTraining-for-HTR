"""
@author : Zi-Rui Wang
@time : 2024
@github : https://github.com/Wukong90
"""

from configure import IAM_utils
from configure import Preprocessing
from torch.utils.data import Dataset

class IAMDataset(Dataset):
    def __init__(self, set='train', set_wid=False, data_h=32):
        self.data_h  = data_h
        self.set = set
        self.data = IAM_utils.iam_main_loader(set,set_wid)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item][0]
        gt  = self.data[item][1]
        wid = self.data[item][2]

        img,widt = Preprocessing.preprocessing(img, data_h = self.data_h, set = self.set)

        return img,widt,gt,wid,self.set