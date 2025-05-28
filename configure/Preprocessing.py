"""
@author : Zi-Rui Wang
@time : 2024
@github : https://github.com/Wukong90
"""

import numpy as np
from skimage.transform import resize
import random
import cv2
import torch

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

    new_image = Image.new('L',(width,124),255)
    randseed1 = np.random.randint(1, 3)

    if(randseed1 == 1):
        location_width = (width - image.width)/2
        location_height = (124 - image.height)/2 
    else:
        location_width = np.random.randint(max(int(width - image.width) + 1,1))
        location_height = np.random.randint(max(int(124 - image.height) + 1,1))
    
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

    image = image.reshape((1,124,width))
    image = (image / 127.5) - 1.0
    image = torch.FloatTensor(image)

    return image

def transform_data2(image,width):

    new_image = Image.new('L',(width,124),255)


    location_width = (width - image.width)/2
    location_height = (124 - image.height)/2 
    
    new_image.paste(image,(int(location_width),int(location_height)))

    image = np.array(new_image)

    image = image.reshape((1,124,width))
    image = (image / 127.5) - 1.0
    image = torch.FloatTensor(image)

    return image

def ept_collate_fn(batch):

    images, widts, targets, wid, set_= zip(*batch)

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
    targets_ = []
    target_lengths_ = []


    for num in range(0, len(images)):
        cur_img = images[num]
        if(set_[num] == "train"):
            cur_img = transform_data(cur_img,max_w)
        else:
            cur_img = transform_data2(cur_img,max_w)
        new_images.append(cur_img)

    images = torch.stack(new_images, 0)

    return images, targets

def img_resize_n(image, hei=32, if_aug=None):

    if(if_aug == "train"):
        
        randseed = np.random.randint(1, 5)

        if(randseed == 1):
            ang = np.random.uniform(-6,6 + 1e-15)
            image = image.rotate(ang,expand=True,fillcolor=255)
        elif(randseed == 2):
            sher = np.random.uniform(-0.5,0.5 + 1e-15)
            image = image.transform(image.size, Image.AFFINE, data=(1,sher,0,0,1,0))
        elif(randseed == 3):
            scal = np.random.uniform(1.1,1.3 + 1e-15)
            image = image.transform(image.size, Image.AFFINE, data=(scal,0,0,0,1,0))

        widt,heig = image.size

        if ((image.height > hei)):
            new_width = (float(image.width) / float(image.height)) * float(hei)
            new_width = int(new_width)
            image = image.resize((new_width, hei), resample=Image.BILINEAR)  #need to revise
            widt = new_width

    else:

        widt,heig = image.size

        if ((image.height > hei)):
            new_width = (float(image.width) / float(image.height)) * float(hei)
            new_width = int(new_width)
            image = image.resize((new_width, hei), resample=Image.BILINEAR)  #need to revise
            widt = new_width

    return image,widt

def preprocessing(img, data_h=32,set = "train"):

    img,widt = img_resize_n(img, height=data_h, if_aug = set)

    return img,widt
