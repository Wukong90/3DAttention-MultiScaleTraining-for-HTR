'''
@author : Zi-Rui Wang
@time : 2024
@github : https://github.com/Wukong90
'''
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss


from configure.config import train_config as config
from configure import HCCDocDataset,ept_collate_fn
#from configure import EPTDataset,ept_collate_fn
from model import TDMTNet
from model.ctc_decoder import ctc_decode
import editdistance
import nltk
import cv2
import math

#For HCCDoc Dataset
out_f = open('./train_loss/HCCDoc_TDMSNet_onlyCTC.txt','w')
save_model_dir = './weights/HCCDoc_TDMSNet_onlyCTC/'
model_name = 'HCCDoc_TDMSNet_onlyCTC_'
'''
For EPT Dataset
out_f = open('./train_loss/EPT_TDMSNet_onlyCTC.txt','w')
save_model_dir = './weights/EPT_TDMSNet_onlyCTC/'
model_name = 'EPT_TDMSNet_onlyCTC_'
'''
def load_image_HCCDoc(image_path, img_width=100, img_height=32):
    image = Image.open(image_path).convert('L')
    widt,heig = image.size

    if ((image.height > 96)):
        new_width = (float(image.width) / float(image.height)) * 96.0
        new_width = int(new_width)
        image = image.resize(
        (new_width, 96), resample=Image.BILINEAR)
        widt = new_width

    if(widt<141):
        widt = 141
    else:
        widt_t = float(int((widt-3)/2) + 1)
        widt_t = float(int(widt_t/2) + 1)
        while(widt_t % 12 != 0.0):
            widt = widt + 1
            widt_t = float(int((widt-3)/2) + 1)
            widt_t = float(int(widt_t/2) + 1)
        widt = int(widt)

    new_image = Image.new('L',(widt,96),255)

    location_width = (widt - image.width)/2
    location_height = (96 - image.height)/2
    new_image.paste(image,(int(location_width),int(location_height)))
    image = np.array(new_image)
    image = image.reshape((1,1, 96, widt))
    image = (image / 127.5) - 1.0

    image = torch.FloatTensor(image)
    return image
'''
For EPT Dataset
def load_image_EPT(image_path, img_width=100, img_height=32):
    image = Image.open(image_path).convert('L')

    if ((image.height > 96)):
        new_width = (float(image.width) / float(image.height)) * 96.0
        new_width = int(new_width)
        image = image.resize(
        (new_width, 96), resample=Image.BILINEAR)  #need to revise

    new_image = Image.new('L',(1440,96),255)
    location_width = (1440 - image.width)/2
    location_height = (96 - image.height)/2
    new_image.paste(image,(int(location_width),int(location_height)))
    image = np.array(new_image)
    image = image.reshape((1,1, 96, 1440))
    image = (image / 127.5) - 1.0
    image = torch.FloatTensor(image)
    return image
'''
def compute_cer(results):

    tot_ec = 0.0
    tot_tc = 0.0
    for label, pre in results:
        ec = nltk.edit_distance(pre.split(' '), label.split(' '))
        tc = len(label.split(' '))
        tot_ec = tot_ec + float(ec)
        tot_tc = tot_tc + float(tc)
    return tot_ec/tot_tc

def evaluate(net, imagelist_path, label_path, imagedir_path,
                    decode_method="greedy",
                    beam_size=10):
    net=net.eval()
    device = 'cuda' if next(net.parameters()).is_cuda else 'cpu'
    total_results=[]
    total_results_=[]
    total_results__=[]

    with torch.no_grad():
        imglist_f = open(imagelist_path,'r')
        label_f = open(label_path,'r')

        while True:
              one_results=[]
              one_results_=[]
              one_results__=[]

              line = imglist_f.readline()
              line = line.strip('\n')
              one_label = label_f.readline()
              one_label = one_label.strip('\n')
              if not line:
                 break
              f_path = imagedir_path + line + ".jpg"
              image = load_image_HCCDoc(image_path=f_path).to(device)
              #image = load_image_EPT(image_path=f_path).to(device) #EPT
              logits,logits_,logits__ = net(image)

              log_probs = torch.nn.functional.log_softmax(logits, dim=2)
              preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,label2char=False)
              one_results.append(one_label)
              pre_one = " ".join([str(x) for x in preds[0]])
              one_results.append(pre_one)
              total_results.append(one_results)

              log_probs_ = torch.nn.functional.log_softmax(logits_, dim=2)
              preds_ = ctc_decode(log_probs_, method=decode_method, beam_size=beam_size,label2char=False)
              one_results_.append(one_label)
              pre_one = " ".join([str(x) for x in preds_[0]])
              one_results_.append(pre_one)
              total_results_.append(one_results_)

              log_probs__ = torch.nn.functional.log_softmax(logits__, dim=2)
              preds__ = ctc_decode(log_probs__, method=decode_method, beam_size=beam_size,label2char=False)
              one_results__.append(one_label)
              pre_one = " ".join([str(x) for x in preds_[0]])
              one_results__.append(pre_one)
              total_results__.append(one_results__)

        imglist_f.close()
        label_f.close()
    current_results = compute_cer(total_results)
    current_results_ = compute_cer(total_results_)
    current_results__ = compute_cer(total_results__)

    return current_results,current_results_,current_results__

def train_batch(net, data, optimizer, criterion, device):
    net.train()
    images, targets, target_lengths = [d.to(device) for d in data]
   
    logits1,logits2,logits3 = net(images)

    log_probs1 = torch.nn.functional.log_softmax(logits1, dim=2)
    batch_size = images.size(0)

    input_lengths1 = torch.LongTensor([logits1.size(0)] * batch_size)  #logits.size(0) denote the width of image

    target_lengths = torch.flatten(target_lengths) # no use

    loss1 = criterion(log_probs1, targets, input_lengths1, target_lengths)

    log_probs2 = torch.nn.functional.log_softmax(logits2, dim=2)
    input_lengths2 = torch.LongTensor([logits2.size(0)] * batch_size)  #logits.size(0) denote the width of image
    loss2 = criterion(log_probs2, targets, input_lengths2, target_lengths)

    log_probs3 = torch.nn.functional.log_softmax(logits3, dim=2)
    input_lengths3 = torch.LongTensor([logits3.size(0)] * batch_size)  #logits.size(0) denote the width of image
    loss3 = criterion(log_probs3, targets, input_lengths3, target_lengths)


    loss1 = torch.mul(loss1,1/3.0)
    loss2 = torch.mul(loss2,1/3.0)
    loss3 = torch.mul(loss3,1/3.0)

    loss_ = torch.add(loss1,loss2) 
    loss = torch.add(loss3,loss_) 

    if (loss > 1000000.0):
        print("wrong"," ",loss)
        optimizer.zero_grad()
        return 0
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size'] 
    eval_batch_size = config['eval_batch_size']
    lr = config['lr']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    cpu_workers = config['cpu_workers']

    data_dir = config['train_data_dir']
    data_list = config['train_list']
    label_list = config['train_label_list']

    test_data_dir = config['test_data_dir']
    test_data_list = config['test_list']
    test_label_list = config['test_label_list']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}',file=out_f)

    train_dataset = HCCDocDataset(root_dir=data_dir, data_path = data_list, label_path = label_list, mode='train')
    #train_dataset = EPTDataset(root_dir=data_dir, data_path = data_list, label_path = label_list, mode='train')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=ept_collate_fn)

    
    num_class = 5888 + 1  #HCCD
    #num_class = 4047 + 1 #EPT


    Net = TDMTNet(1, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'])

    Net.to(device)

    optimizer = optim.AdamW([{'params':Net.parameters()}])
    #optimizer = optim.Adam([{'params':crnn.parameters()}])  #EPT

    sche = optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.5)
    #sche = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5) #EPT

    criterion = CTCLoss()
    criterion.to(device)

    i = 1
    
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}',file=out_f)

        tot_train_loss = 0.
        tot_train_count = 0

        for train_data in train_loader:
            loss = train_batch(Net, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)
            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('current_train_batch_loss[', i, ']: ', loss / train_size,file=out_f)
            out_f.flush()
            i += 1
        
        save_model_path = save_model_dir + model_name + str(epoch)
        torch.save(Net.state_dict(), save_model_path)      
        i = 1
        print('train_loss: ', tot_train_loss / tot_train_count,file=out_f)

        if(epoch>=180): #For EPT dataset, epoch >= 300
            evaluation = evaluate(Net, test_data_list, test_label_list,test_data_dir,
                        decode_method=config['decode_method'],
                        beam_size=config['beam_size'])

            print('test_AR: ',evaluation, file=out_f)

        sche.step()

if __name__ == '__main__':
    main()
