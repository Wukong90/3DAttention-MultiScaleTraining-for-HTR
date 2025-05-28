'''
@author : Zi-Rui Wang
@time : 2024
@github : https://github.com/Wukong90
'''
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import cv2
import math

#Data augment
def tfactor(img):

    factor = np.random.randint(0, 4)
    if factor == 0:
        img[:, :] = img[:, :] * (0.7 + np.random.random() * 0.3)
    elif factor == 1:
        img[:, :] = img[:, :] * (0.8 + np.random.random() * 0.2)
    elif factor == 2:
        img[:, :] = img[:, :] * (0.6 + np.random.random() * 0.4)
    else:
        img[:, :] = img[:, :] * (0.5 + np.random.random() * 0.5)

    return img

def AddGauss(img, level):

    level = np.random.randint(3, 7)

    return cv2.blur(img, (level + r(1), level + r(1)))

def r(val):
    return int(np.random.random() * val)

def transform_data(image,width):

    new_image = Image.new('L',(width,96),255)
    randseed1 = np.random.randint(1, 3)

    if(randseed1 == 1):
        location_width = (width - image.width)/2
        location_height = (96 - image.height)/2 
    else:
        location_width = np.random.randint(max(int(width - image.width) + 1,1))
        location_height = np.random.randint(max(int(96 - image.height) + 1,1))
    
    new_image.paste(image,(int(location_width),int(location_height)))
    image = np.array(new_image)
    #t
    new_image = np.array(new_image)
    randseed2 = np.random.randint(1, 5)

    if randseed2 == 1:
        image = np.array(new_image)
    elif (randseed2 == 2):
        image = cv2.bitwise_not(new_image.copy())
    elif(randseed2 == 3):
        image = tfactor(new_image.copy())
    else:
        image = AddGauss(new_image.copy(), 0)

    image = image.reshape((1,96,width))
    image = (image / 127.5) - 1.0

    image = torch.FloatTensor(image)

    return image

#Bulid HCCDocDataset
class HCCDocDataset(Dataset):

    def __init__(self, root_dir=None, mode=None, data_path=None, label_path=None):

        paths, texts = self._load_from_raw_files(root_dir,data_path,label_path, mode)

        self.paths = paths
        self.texts = texts

    def _load_from_raw_files(self, root_dir,data_path,label_path, mode):
        mapping = {}

        paths_file = None

        paths = []
        texts = []

        with open(data_path, 'r') as fr:
        
            for line in fr.readlines():
                line=line.strip('\n')
                file_path = root_dir + line
                paths.append(file_path)ak
                
        with open(label_path, 'r') as fr:
            
            for line in fr.readlines():
                labels=line.strip('\n')
                texts.append(labels)

        return paths, texts

    def __len__(self):           
        return len(self.paths)

    def __getitem__(self, index):
        
        paths = self.paths[index]
        ctexts = self.texts[index]
        img_path = paths + ".jpg"
        image = Image.open(img_path).convert('L')

        randseed = np.random.randint(1, 6)

        if(randseed == 1): #Data augment
            ang = np.random.randint(-6, 7)                
            image = image.rotate(ang,expand=True,fillcolor=255)

        widt,heig = image.size
        
        if ((image.height > 96)):
            new_width = (float(image.width) / float(image.height)) * 96.0
            new_width = int(new_width)
            image = image.resize(
            (new_width, 96), resample=Image.BILINEAR)
            widt = new_width

        target = ctexts.split(' ')
        target_length = [len(target)] 
        target_ = list(map(int, target))
        target = torch.LongTensor(target_)
        target_length = torch.LongTensor(target_length)

        return image, target, target_length, widt   

#Load batch
def ept_collate_fn(batch):

    images, targets, target_lengths, widts = zip(*batch)
    max_w = max(widts)


    if(max_w<141):
        max_w = 141
    else:
        max_w_t = float(int((max_w-3)/2) + 1)
        max_w_t = float(int(max_w_t/2) + 1)
        while(max_w_t % 12 != 0.0):
            max_w = max_w + 1
            max_w_t = float(int((max_w-3)/2) + 1)
            max_w_t = float(int(max_w_t/2) + 1)
        max_w = int(max_w)
    new_images = []

    for num in range(0, len(images)):
        cur_img = images[num]
        cur_img = transform_data(cur_img,max_w)
        new_images.append(cur_img)

    images = torch.stack(new_images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)

    return images, targets, target_lengths

'''
#Build EPTDataset and Load batch
class EPTDataset(Dataset):

    def __init__(self, root_dir=None, mode=None, data_path=None, label_path=None):

        paths, texts = self._load_from_raw_files(root_dir,data_path,label_path, mode)


        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir,data_path,label_path, mode):
        mapping = {}

        paths_file = None

        paths = []
        texts = []

        with open(data_path, 'r') as fr:
            for line in fr.readlines():
                line=line.strip('\n')
                file_path = root_dir + line
                paths.append(file_path)              
                
        with open(label_path, 'r') as fr:
            for line in fr.readlines():
                labels=line.strip('\n')
                texts.append(labels)              

        return paths, texts

    def __len__(self):           
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img_path = path + ".jpg"

        try:
            image = Image.open(img_path).convert('L')
            randseed = np.random.randint(1, 6)
            if(randseed == 1):
                ang = np.random.randint(-6, 7)                
                image = image.rotate(ang,expand=True,fillcolor=255)
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if ((image.height > 96)):
            new_width = (float(image.width) / float(image.height)) * 96.0
            new_width = int(new_width)
            image = image.resize(
            (new_width, 96), resample=Image.BILINEAR)

        new_image = Image.new('L',(1440,96),255)

        randseed1 = np.random.randint(1, 3)

        if(randseed1 == 1):

            location_width = (1440 - image.width)/2
            location_height = (96 - image.height)/2
            
        else:
            location_width = np.random.randint(max(int(1440 - image.width) + 1,1))
            location_height = np.random.randint(max(int(96 - image.height) + 1,1))


        new_image.paste(image,(int(location_width),int(location_height)))

        image = np.array(new_image)

        new_image = np.array(new_image)

        randseed2 = np.random.randint(1, 5)

        if randseed2 == 1:
            image = np.array(new_image)
        elif (randseed2 == 2):
            image = cv2.bitwise_not(new_image.copy())
        elif(randseed2 == 3):
            image = tfactor(new_image.copy())
        else:
            image = AddGauss(new_image.copy(), 0)
    
        image = image.reshape((1,96,1440))
        image = (image / 127.5) - 1.0  #no char

        image = torch.FloatTensor(image) #no char

        if self.texts:
            text = self.texts[index]
            target = text.split(' ')
            target_length = [len(target)] 
            target_ = list(map(int, target))
            target = torch.LongTensor(target_)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def ept_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)

    return images, targets, target_lengths
'''