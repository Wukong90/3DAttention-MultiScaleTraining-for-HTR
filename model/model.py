'''
@author : Zi-Rui Wang
@time : 2024
@github : https://github.com/Wukong90
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.encoder_layer import EncoderLayer
import numpy as np
import math
from math import sqrt

class SELayer(nn.Module): #Channel-wise attention
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def SimAM(X): #Point-wise attention
	# X: input feature [N, C, H, W]
	# lambda: coefficient λ in Eqn (5)
	n = X.shape[2] * X.shape[3] - 1
	# square of (t - u)
	d = (X - X.mean(dim=[2,3]).view(X.size(0),X.size(1),1,1)).pow(2)
	# d.sum() / n is channel variance
	v = d.sum(dim=[2,3]) / n 
	v = v.view(v.size(0),v.size(1),1,1)
	# E_inv groups all importance of X
	#E_inv = d / (4 * (v + lambda)) + 0.5
	E_inv = d / (4 * (v + 0.001)) + 0.5  #set λ=0.0001
	# return attended features
	return X * F.sigmoid(E_inv)

def Positionalencoding2d(d_model, height, width): #Two-dimensional positional encoding
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class TDMTNet(nn.Module):
    def __init__(self, img_channel, num_class, map_to_seq_hidden=128, rnn_hidden=256):
        super(TDMTNet, self).__init__()
        #CNN blocks
        self.b0_c1 = nn.Conv2d(1, 64, 3, 1,1)
        self.b0_b1 = nn.BatchNorm2d(64)
        self.b0_se = SELayer(64)
        self.b0_p = nn.MaxPool2d(kernel_size=3, stride=2)
        #Block1
        self.b1_c1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.b1_b1 = nn.BatchNorm2d(64)
        self.b1_c2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.b1_b2 = nn.BatchNorm2d(64)

        self.b1_se = SELayer(64)
        self.b1_p = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        #Block2
        self.b2_c1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.b2_b1 = nn.BatchNorm2d(128)

        self.b2_c2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.b2_b2 = nn.BatchNorm2d(128)

        self.b2_up1c = nn.Conv2d(64, 128, 1, 1)
        self.b2_up1b = nn.BatchNorm2d(128)

        self.b2_se = SELayer(128)
        #Block3
        self.b3_c1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.b3_b1 = nn.BatchNorm2d(256)

        self.b3_c2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.b3_b2 = nn.BatchNorm2d(256)

        self.b3_c3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.b3_b3 = nn.BatchNorm2d(256)

        self.b3_c4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.b3_b4 = nn.BatchNorm2d(256)

        self.b3_up1c = nn.Conv2d(128, 256, 1, 1)
        self.b3_up1b = nn.BatchNorm2d(256)

        self.b3_se = SELayer(256)

        #Three-dimensional attention
        self.temperature = 1.0
        self.proj_k = nn.Linear(256, 300)
        self.gen_weight = nn.Linear(300, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = 1
        self.en = EncoderLayer(d_model=256,ffn_hidden=256,n_head=self.num_head,drop_prob=0.0)

        #Linear transformation for visual features
        
        self.map2seq = nn.Linear(256, map_to_seq_hidden)
        self.map2seq_ = nn.Linear(256, map_to_seq_hidden)
        self.map2seq__ = nn.Linear(256, map_to_seq_hidden)

        #Multi-scale frame length
        self.spl = 3
        self.spl_ = 4
        self.spl__ = 2

        #EncoderLayer for gloabl-context information, rnn for local-context informationand and fusing visual and context features
        #For frame size 3
        self.en1 = EncoderLayer(d_model=map_to_seq_hidden,ffn_hidden=256,n_head=self.num_head,drop_prob=0.0)
        self.en2 = EncoderLayer(d_model=map_to_seq_hidden,ffn_hidden=256,n_head=self.num_head,drop_prob=0.0)
        self.en3 = EncoderLayer(d_model=map_to_seq_hidden,ffn_hidden=256,n_head=self.num_head,drop_prob=0.0)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(896, num_class)

        #For frame size 4
        self.en1_ = EncoderLayer(d_model=map_to_seq_hidden,ffn_hidden=256,n_head=self.num_head,drop_prob=0.0)
        self.en2_ = EncoderLayer(d_model=map_to_seq_hidden,ffn_hidden=256,n_head=self.num_head,drop_prob=0.0)
        self.en3_ = EncoderLayer(d_model=map_to_seq_hidden,ffn_hidden=256,n_head=self.num_head,drop_prob=0.0)

        self.rnn1_ = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2_ = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense_ = nn.Linear(896, num_class)
    
        #For frame size 2
        self.en1__ = EncoderLayer(d_model=map_to_seq_hidden,ffn_hidden=256,n_head=self.num_head,drop_prob=0.0)
        self.en2__ = EncoderLayer(d_model=map_to_seq_hidden,ffn_hidden=256,n_head=self.num_head,drop_prob=0.0)
        self.en3__ = EncoderLayer(d_model=map_to_seq_hidden,ffn_hidden=256,n_head=self.num_head,drop_prob=0.0)

        self.rnn1__ = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2__ = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense__ = nn.Linear(896, num_class)

    def _integration(self, weight, value):
        attn = weight / self.temperature
        attn = self.softmax(attn)  # BNxT
        output = torch.bmm(attn.unsqueeze(1), value).squeeze(
            1)  # BNxT x BNxTxD-> BNxD
        return output, attn

    def forward(self, x):  

        # CNN blocks
        out = self.b0_c1(x)
        out = self.b0_b1(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)
        out = self.b0_se(out)
        out = self.b0_p(out)
        #Block1
        identi1 = out

        out = self.b1_c1(out)
        out = self.b1_b1(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)   
        out = self.b1_c2(out)
        out = self.b1_b2(out)

        out = nn.ReLU(inplace=True)(identi1 + out)
        out = SimAM(out)   

        out = self.b1_se(out)  
        out = self.b1_p(out)
        #Block2
        identi2 = self.b2_up1c(out)
        identi2 = self.b2_up1b(identi2)

        out = self.b2_c1(out)
        out = self.b2_b1(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)   
        out = self.b2_c2(out)
        out = self.b2_b2(out)

        out = nn.ReLU(inplace=True)(identi2 + out)
        out = SimAM(out)   
        out = self.b2_se(out)

        #Block3
        identi31 = self.b3_up1c(out)
        identi31 = self.b3_up1b(identi31)

        out = self.b3_c1(out)
        out = self.b3_b1(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)   
        out = self.b3_c2(out)
        out = self.b3_b2(out)

        out = nn.ReLU(inplace=True)(identi31 + out)
        out = SimAM(out)   

        identi32 = out

        out = self.b3_c3(out)
        out = self.b3_b3(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)   
        out = self.b3_c4(out)
        out = self.b3_b4(out)

        out = nn.ReLU(inplace=True)(identi32 + out)
        out = SimAM(out)   
        out = self.b3_se(out)

        #
        bs = out.shape[0]

        #Multi-scale framing
        ml1 = int(out.shape[3]/self.spl)
        ml2 = int(out.shape[3]/self.spl_)
        ml3 = int(out.shape[3]/self.spl__)

        out = out.permute(0,1,3,2)
        
        #For frame size self.spl

        out1 = out.view(out.shape[0],out.shape[1],ml1,self.spl,out.shape[3])
        out1 = out1.permute(0,1,2,4,3)
        out1 = out1.permute(0,2,1,3,4)

        out1 = out1.reshape(out1.shape[0]*out1.shape[1],out1.shape[2],out1.shape[3],out1.shape[4])
        
        #3D Attention
        
        b_n, channel, height, width = out1.size()
        pe = Positionalencoding2d(channel, height, width)
        out1 = out1 + pe.cuda()

        v1 = out1.reshape(b_n, channel, height*width)
        v1 = v1.permute(0, 2, 1)

        v1 = self.en(v1)

        k1 = torch.tanh(self.proj_k(v1))
        weight1 = self.gen_weight(k1).squeeze(2)
        cont1, attn1 = self._integration(weight1, v1)
        vis_f1 = cont1.view(bs,ml1,channel)

        #Context features

        m_vis_f1 = self.map2seq(vis_f1)

        outputs1 = self.en1(m_vis_f1)
        outputs1 = self.en2(outputs1)
        outputs1 = self.en3(outputs1)

        outputs1 = outputs1.permute(1,0,2)
        vis_f1 = vis_f1.permute(1,0,2)
        m_vis_f1 = m_vis_f1.permute(1,0,2)

        outputs2, _ = self.rnn1(m_vis_f1)
        outputs2, _ = self.rnn2(outputs2)

        #Fusing features

        out_e1 = torch.cat((outputs1,outputs2,vis_f1),2)

        #For frame size self.spl_

        out2 = out.view(out.shape[0],out.shape[1],ml2,self.spl_,out.shape[3])
        out2 = out2.permute(0,1,2,4,3)
        out2 = out2.permute(0,2,1,3,4)

        out2 = out2.reshape(out2.shape[0]*out2.shape[1],out2.shape[2],out2.shape[3],out2.shape[4])

        #3D Attention

        b_n_, channel_, height_, width_ = out2.size()
        pe_ = Positionalencoding2d(channel_, height_, width_)
        out2 = out2 + pe_.cuda()

        v2 = out2.reshape(b_n_, channel_, height_*width_)
        v2 = v2.permute(0, 2, 1)

        v2 = self.en(v2)

        k2 = torch.tanh(self.proj_k(v2))
        weight2 = self.gen_weight(k2).squeeze(2)
        cont2, attn2 = self._integration(weight2, v2)
        vis_f2 = cont2.view(bs,ml2,channel_)

        #Context features
        
        m_vis_f2 = self.map2seq_(vis_f2)
        outputs1_ = self.en1_(m_vis_f2)
        outputs1_ = self.en2_(outputs1_)
        outputs1_ = self.en3_(outputs1_)
        outputs1_ = outputs1_.permute(1,0,2)
        vis_f2 = vis_f2.permute(1,0,2)
        m_vis_f2 = m_vis_f2.permute(1,0,2)
        outputs2_, __ = self.rnn1_(m_vis_f2)
        outputs2_, __ = self.rnn2_(outputs2_)

        #Fusing features

        out_e2 = torch.cat((outputs1_,outputs2_,vis_f2),2)

        #For frame size self.spl__

        out3 = out.view(out.shape[0],out.shape[1],ml3,self.spl__,out.shape[3])
        out3 = out3.permute(0,1,2,4,3)
        out3 = out3.permute(0,2,1,3,4)
        out3 = out3.reshape(out3.shape[0]*out3.shape[1],out3.shape[2],out3.shape[3],out3.shape[4])

        #3D Attention

        b_n__, channel__, height__, width__ = out3.size()
        pe__ = Positionalencoding2d(channel__, height__, width__)
        out3 = out3 + pe__.cuda()

        v3 = out3.reshape(b_n__, channel__, height__*width__)
        v3 = v3.permute(0, 2, 1)

        v3 = self.en(v3)

        k3 = torch.tanh(self.proj_k(v3))
        weight3 = self.gen_weight(k3).squeeze(2)
        cont3, attn3 = self._integration(weight3, v3)
        vis_f3 = cont3.view(bs,ml3,channel__)

        #Context features

        m_vis_f3 = self.map2seq__(vis_f3)
        outputs1__ = self.en1__(m_vis_f3)
        outputs1__ = self.en2__(outputs1__)
        outputs1__ = self.en3__(outputs1__)
        outputs1__ = outputs1__.permute(1,0,2)
        vis_f3 = vis_f3.permute(1,0,2)
        m_vis_f3 = m_vis_f3.permute(1,0,2)
        outputs2__, ___ = self.rnn1__(m_vis_f3)
        outputs2__, ___ = self.rnn2__(outputs2__)

        #Fusing features
        out_e3 = torch.cat((outputs1__,outputs2__,vis_f3),2)

        #Outputs
        out_l1 = self.dense(out_e1)
        out_l2 = self.dense_(out_e2)
        out_l3 = self.dense__(out_e3)

        return out_l1,out_l2,out_l3
