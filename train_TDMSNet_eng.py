"""
@author : Zi-Rui Wang
@time : 2024
@github : https://github.com/Wukong90
"""

import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from torch.autograd import Variable
from configure import Preprocessing
from configure import IAMDataset
from model import TDMTNet
import nltk

out_f = open('./train_loss/IAM_TDMSNet_onlyCTC.txt','w')
save_model_dir = './weights/IAM_TDMSNet_onlyCTC/'
model_name = 'IAM_TDMSNet_onlyCTC_'

alphabet_t = """_!#&()*+,-.'"/0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz """
cdict = {c: i for i, c in enumerate(alphabet_t)}  # character -> int
icdict = {i: c for i, c in enumerate(alphabet_t)}  # int -> character

def CER(label, prediction):
    return nltk.edit_distance(label, prediction),len(label)

def train_batch(model, data, optimizer, criterion, device):
    model.train()

    img = data[0]
    images = data[0].cuda()
    targets = data[1]
    
    logits1,logits2,logits3 = model(images)

    log_probs1 = torch.nn.functional.log_softmax(logits1, dim=2)
    batch_size = images.size(0)

    input_lengths1 = torch.LongTensor([logits1.size(0)] * batch_size)  #logits.size(0) denote the width of image
    input_lengths1 = input_lengths1.cuda()

    labels = Variable(torch.LongTensor([cdict[c] for c in ''.join(targets)]))
    labels = labels.cuda()

    label_lengths = torch.LongTensor([len(t) for t in targets])
    label_lengths = label_lengths.cuda()

    loss1 = criterion(log_probs1, labels, input_lengths1, label_lengths)

    log_probs2 = torch.nn.functional.log_softmax(logits2, dim=2)
    input_lengths2 = torch.LongTensor([logits2.size(0)] * batch_size)  #logits.size(0) denote the width of image
    loss2 = criterion(log_probs2, labels, input_lengths2, label_lengths)

    log_probs3 = torch.nn.functional.log_softmax(logits3, dim=2)
    input_lengths3 = torch.LongTensor([logits3.size(0)] * batch_size)  #logits.size(0) denote the width of image
    loss3 = criterion(log_probs3, labels, input_lengths3, label_lengths)

    loss1 = torch.mul(loss1,1/3.0)
    loss2 = torch.mul(loss2,1/3.0)
    loss3 = torch.mul(loss3,1/3.0)

    loss_ = torch.add(loss1,loss2) 
    loss = torch.add(loss3,loss_) 

    if (loss > 1000000):
        print("wrong"," ",loss)
        optimizer.zero_grad()
        return 0.0

    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

def val(model, criterion, val_loader):
    model.eval()
    avg_CER = 0
    avg_WER = 0
    tot_CE = 0
    tot_WE = 0
    tot_Clen = 0
    tot_Wlen = 0


    for val_data in val_loader:
        images = val_data[0].cuda()
        transcr = val_data[1]

        preds,preds1,preds2 = model(images)
        tdec = preds.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        if tdec.ndim == 1:
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
            # Compute metrics
            cur_CE,cur_Clen = CER(transcr[0], dec_transcr)
            tot_CE = tot_CE + cur_CE
            tot_Clen = tot_Clen + cur_Clen

        else:
            for k in range(len(tdec)):
                tt = [v for j, v in enumerate(tdec[k]) if j == 0 or v != tdec[k][j - 1]]
                dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
                cur_CE,cur_Clen = CER(transcr[k], dec_transcr)
                tot_CE = tot_CE + cur_CE
                tot_Clen = tot_Clen + cur_Clen

    avg_CER = tot_CE / tot_Clen
    return avg_CER

def main():

    epochs = 400
    train_batch_size = 22
    show_interval = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = IAMDataset(data_h=124,set='train', set_wid=False) #set_wid is used for loading writer information, you can ignore it in this work.
    val1_set = IAMDataset(data_h=124,set='val', set_wid=False)
    val2_set = IAMDataset(data_h=124,set='val2', set_wid=False)
    test_set = IAMDataset(data_h=124,set='test', set_wid=False)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=Preprocessing.ept_collate_fn)

    val_loader = DataLoader(
        dataset=val1_set,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        collate_fn=Preprocessing.ept_collate_fn)

    val2_loader = DataLoader(
        dataset=val2_set,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        collate_fn=Preprocessing.ept_collate_fn)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        collate_fn=Preprocessing.ept_collate_fn)

    num_class = len(alphabet_t) + 1

    Net = TDMTNet(1, num_class, map_to_seq_hidden=128, rnn_hidden=256)
    Net.cuda()
    optimizer = optim.AdamW([{'params':Net.parameters(),}])

    criterion = CTCLoss(reduction='sum')
    criterion.cuda()

    sche = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)

    i = 1

    for epoch in range(1, epochs + 1):

        print(f'epoch: {epoch}',file=out_f)
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(Net, train_data, optimizer, criterion, device)
            train_size = train_batch_size
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

        if epoch > 200:
            val_CER =  val(Net, criterion, val_loader)
            val2_CER = val(Net, criterion, val2_loader)
            avg_val_CER = (val_CER + val2_CER) / 2.0
            test_CER = val(Net, criterion, test_loader)
            print('val1 CER ', val_CER, ' val2 CER ',  val2_CER, ' avg_val CER ', avg_val_CER , ' test CER ' , test_CER , ' epoch ' ,epoch, file=out_f)

        sche.step()

if __name__ == '__main__':
    main()
