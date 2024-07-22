import sys
sys.path.insert(0, '../Utilities/')

import torch
from collections import OrderedDict
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings
import random
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import pandas as pd
import math
import onnx
import time
from thop import profile
from scipy.special import iv
import copy



# ## 1.calculate core loss
#
#
# class Projector(nn.Module):
#     def __init__(self, operation_dim, output_dim):
#         super(Projector, self).__init__()
#
#         self.operation_dim = operation_dim
#         self.output_dim = output_dim
#
#         self.out = nn.Sequential(
#             nn.Linear(self.operation_dim, 48),
#             nn.LeakyReLU(),
#             nn.Linear(48, 32),
#             nn.LeakyReLU(),
#             nn.Linear(32, self.output_dim))
#
#     def forward(self, x):
#         y = self.out(x)
#         y = y.unsqueeze(0)
#         return y
#
# class Seq2Seq(nn.Module):
#     def __init__(self, projector,device):
#         super().__init__()
#
#
#         self.projector = projector
#         self.device = device
#
#     def forward(self,   operation,teacher_forcing_ratio=0.5):
#         batch_size=operation.size(dim=0)
#         # material_out = self.projector_material(material)
#         # predicted_param = self.projector(torch.cat((material_out.repeat(batch_size,1),operation),dim=1))
#         predicted_param = self.projector( operation)
#         predicted_param=predicted_param.squeeze(0)
#         predicted_param=predicted_param.unsqueeze(1)
#         # predicted_param=predicted_param.view(-1,1,3)
#         return predicted_param
#
# device = torch.device("cpu")
# projector = Projector(operation_dim=19, output_dim=10)
# net = Seq2Seq(projector,device)
#
# net.load_state_dict(torch.load('core_loss_parameters'))
# net.eval()
#
# #parameters
# std_B=0.0696
# std_F=123497.7031
# mean_T=58.6932
# std_T=24.0967
# std_Ploss=1.8524
# std_B_max=0.1331
# #input
# B_sequence=0.025*torch.tensor([0,0.38,0.70,0.92,1,0.92,0.70,0.38,0,-0.38,-0.70,-0.92,-1,-0.92,-0.70,-0.38])
# frequency=torch.tensor(100e3)
# temperature=torch.tensor(25)
# B_max=torch.max(B_sequence)-torch.min(B_sequence)
# Ve=torch.tensor(201390e-9)
#
# B_seq_processed= B_sequence/std_B
# frequency_processed=frequency/std_F
# temperature_processed=(temperature-mean_T)/std_T
# B_max_processed=B_max/std_B_max
#
# sample=torch.cat((frequency_processed.unsqueeze(0),temperature_processed.unsqueeze(0),B_max_processed.unsqueeze(0),B_seq_processed))
# ploss_processed=net(sample)
#
# #output materials are:
# #3C90,3C94,3E6,3F6,77,78,N27,N30,N49,N87
# ploss=torch.exp(ploss_processed*std_Ploss)*Ve
# print(ploss)


#2. calculate winding loss

class Transformer(nn.Module):
    def __init__(self,
        input_size :int,
        output_size: int,
        dec_seq_len :int,
        max_seq_len :int,
        out_seq_len :int,
        dim_val :int,
        n_encoder_layers :int,
        n_decoder_layers :int,
        n_heads :int,
        dropout_encoder,
        dropout_decoder,
        dropout_pos_enc,
        dim_feedforward_encoder :int,
        dim_feedforward_decoder :int,
        dim_feedforward_projecter :int,
        num_var: int
        ):

        super().__init__()

        self.dec_seq_len = dec_seq_len
        self.n_heads = n_heads
        self.out_seq_len = out_seq_len
        self.dim_val = dim_val
        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))
        self.decoder_input_layer = nn.Sequential(
            nn.Linear(output_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))
        self.linear_mapping = nn.Sequential(
            nn.Linear(dim_val, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, output_size))
        self.linear_mapping_scalar = nn.Sequential(
            nn.Linear(out_seq_len, out_seq_len),
            nn.Tanh(),
            nn.Linear(out_seq_len, input_size))
        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc, max_len=max_seq_len)
        self.projector = nn.Sequential(
            nn.Linear(dim_val + num_var, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, dim_val))
        self.projector_directout = nn.Sequential(
            nn.Linear(dim_val + num_var, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, output_size))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="relu",
            batch_first=True
            )
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_encoder_layers, norm=None)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            activation="relu",
            batch_first=True
            )
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=n_decoder_layers, norm=None)

    def forward(self, src: Tensor, var: Tensor,src_mask, src_padding_mask) -> Tensor:
        ########################
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src,mask=src_mask,src_key_padding_mask=src_padding_mask)
        enc_seq_len = 53

        var = var.unsqueeze(1).repeat(1,enc_seq_len,1)
        src = self.projector(torch.cat((src,var),dim=2))

        output = self.projector_directout(torch.cat((src, var), dim=2))

        # tgt = self.decoder_input_layer(tgt.unsqueeze(2))
        # tgt = self.positional_encoding_layer(tgt)
        # output = self.decoder(
        #     tgt=tgt,
        #     memory=src,
        #     tgt_mask=tgt_mask,
        #     tgt_key_padding_mask=tgt_padding_mask,
        #     memory_mask=tgt_mask,
        #     )
        # output= self.linear_mapping(output)

        return output

class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz1: int, sz2: int) -> Tensor:
    #Generates an upper-triangular matrix of -inf, with zeros on diag.
    return torch.triu(torch.ones(sz1, sz2) * float('-inf'), diagonal=1)


# Select GPU as default device
device = torch.device("cpu")

# Setup network
net = Transformer(
    dim_val=64,
    input_size=2,
    output_size=1,
    dec_seq_len=53,
    max_seq_len=53,
    out_seq_len=53,
    n_decoder_layers=1,
    n_encoder_layers=1,
    n_heads=8,
    dropout_encoder=0.1,
    dropout_decoder=0.1,
    dropout_pos_enc=0.1,
    dim_feedforward_encoder=96,
    dim_feedforward_decoder=96,
    dim_feedforward_projecter=96,
    num_var=5)

#net.load_state_dict(torch.load('winding_loss_parameters_6_64dim2')) #64dim log no square
net.load_state_dict(torch.load('winding_loss_parameters_7_64dim_encoder_c6000_2'))

net.eval()
# net=torch.load('winding_loss_transformer.pt')

# parameters
std_gap_height=torch.tensor(0.0013842)
std_H_vector=torch.tensor(3.6154215)
std_position_x_shifted=torch.tensor(0.00388534)
std_position_y=torch.tensor(0.00731278)
std_window_height=torch.tensor(0.02307151)
std_window_width=torch.tensor(0.00606402)
std_diameter=torch.tensor(0.0022456)

#input
gap_height = torch.tensor(1.4e-3)
window_height = torch.tensor(93.7e-3)
window_width = torch.tensor(22.8e-3)
thickness_side = torch.tensor(0.7e-3)
thickness_top = torch.tensor(1e-3)
#diameter=torch.tensor(0.0082)
core_length=torch.tensor(0.0138)
center_core_width=torch.tensor(0.01375)
turns = torch.tensor(21)
f = torch.tensor(100e3)
T = torch.tensor(25)
current = 1.41 * torch.tensor([0, 0.38, 0.70, 0.92, 1, 0.92, 0.70, 0.38, 0, -0.38, -0.70, -0.92, -1, -0.92, -0.70, -0.38])
litz_diameter = torch.tensor(0.1e-3)
litz_diameter_table={'0.04':0.069,'0.07':0.104,'0.1':0.14,'0.2':0.259}
strand_filling_factor=torch.pi/(2*torch.sqrt(torch.tensor(3)))


#generate position
diameter_ini = torch.sqrt((window_height - thickness_top * 2) * (window_width - thickness_side) / turns / torch.pi * 4)
for i in range(100):
    n_l = torch.floor(((window_width - thickness_side) / diameter_ini - (-torch.sqrt(torch.tensor(3)) + 2) / 2) * 2 / torch.sqrt(torch.tensor(3)))
    n_t1 = torch.floor((window_height - thickness_top * 2) / diameter_ini)
    n_t2 = torch.floor((window_height - thickness_top * 2) / diameter_ini - 0.5)
    turn_tot = 0
    for j in range(int(n_l.item())):
        if torch.remainder(torch.tensor(j), 2) == 0:
            turn_tot = turn_tot + n_t1
        else:
            turn_tot = turn_tot + n_t2

    if turn_tot >= turns:
        break
    else:
        diameter_ini = diameter_ini * 0.95
diameter = diameter_ini
#diameter=torch.tensor(0.00801)

position_x_sample = torch.empty(0)
position_y_sample = torch.empty(0)
for i in range(int(n_l.item())):
    if torch.remainder(torch.tensor(i), 2) == 0:
        position_x_sample = torch.cat((position_x_sample,torch.ones(int(n_t1.item())) * ((thickness_side + diameter / 2) + (i ) * torch.sqrt(torch.tensor(3)) / 2 * diameter)))
        y_sample = (-window_height / 2 + thickness_top - diameter / 2) + torch.linspace(1, int(n_t1), int(n_t1)) * diameter
        position_y_sample = torch.cat((position_y_sample,y_sample))
    else:
        position_x_sample = torch.cat((position_x_sample,torch.ones(int(n_t2.item())) * ((thickness_side + diameter / 2) + (i ) * torch.sqrt(torch.tensor(3)) / 2 * diameter)))
        y_sample = (-window_height / 2 + thickness_top) + torch.linspace(1, int(n_t2), int(n_t2)) * diameter
        position_y_sample = torch.cat((position_y_sample,y_sample))

position_x_sample = position_x_sample[0:turns]
position_y_sample = position_y_sample[0:turns]

# strand number
formatted_value = "{:g}".format((litz_diameter*1e3).item())
litz_number = diameter ** 2 / (litz_diameter_table[str(formatted_value)] * 1e-3) **2

# FFT
Y = torch.fft.fft(current)
P2 = abs(Y / len(current))
P1 = P2[0:int(len(current) / 2 + 1)]
P1[1:  - 1] = 2 * P1[1:  - 1]
I_fft = P1

#test code


# #calculate output
# def predict (net,src,var):
#     # # x = [1, 50]
#     # net.eval()
#
#     # src mask[1, 1, 50, 50]
#     src_seq_len = src.shape[1]
#     src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
#     src_padding_mask = ((src[:,:,0]+src[:,:,1]) == 0)
#
#     src = net.encoder_input_layer(src)
#     src = net.positional_encoding_layer(src)
#     src = net.encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)
#     enc_seq_len = 53
#
#     var = var.unsqueeze(1).repeat(1, enc_seq_len, 1)
#     src = net.projector(torch.cat((src, var), dim=2))
#
#     # 初始化输出,这个是固定值
#     # [1, 50]
#     # [0.1,0,0,0,...0]
#     tgt=torch.tensor([torch.tensor(0.1)]+[torch.tensor(0.0)]*52).unsqueeze(0)
#     output_list=[]
#     # 遍历生成第1个词到第53个词
#     for i in range(52):
#         # [1, 50]
#         tgt_temp = tgt
#
#         # [1, 1, 50, 50]
#         #mask
#         tgt_seq_len = tgt_temp.shape[1]
#         tgt_mask = generate_square_subsequent_mask(tgt_seq_len, tgt_seq_len)
#         tgt_padding_mask = (tgt_temp == 0)
#         tgt_mask=tgt_mask.bool()
#
#         tgt_temp = net.decoder_input_layer(tgt_temp.unsqueeze(2))
#         tgt_temp = net.positional_encoding_layer(tgt_temp)
#         output = net.decoder(
#             tgt=tgt_temp,
#             memory=src,
#             tgt_mask=tgt_mask,
#             tgt_key_padding_mask=tgt_padding_mask,
#             memory_mask=tgt_mask,
#             )
#         output= net.linear_mapping(output)
#
#         # 取出当前词的输出
#         # [1, 50, 39] -> [1, 39]
#         output = output.squeeze(2)[:, i]
#
#         # 以当前词预测下一个词,填到结果中
#         tgt[:, i + 1] = output
#         output_list=output_list+[output]
#
#         if i==52:
#             i=i
#     output_list=torch.tensor(output_list)
#     return output_list

def create_mask(src):
    src_seq_len = src.shape[1]

    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = ((src[:,:,0]+src[:,:,1]) == 0)
    return src_mask,src_padding_mask

norm_length=53
pad_length=norm_length-turns
position_x_sample = torch.nn.functional.pad(position_x_sample, (0, pad_length), mode='constant', value=0)
position_y_sample = torch.nn.functional.pad(position_y_sample, (0, pad_length), mode='constant', value=0)
src=torch.stack(((position_x_sample/std_position_x_shifted).unsqueeze(0),(position_y_sample/std_position_y).unsqueeze(0)),dim=2)
var=torch.stack(((gap_height/std_gap_height).unsqueeze(0), (window_height/std_window_height).unsqueeze(0), (window_width/std_window_width).unsqueeze(0),(turns).unsqueeze(0),(diameter/std_diameter).unsqueeze(0)), dim=1)
src_mask,src_padding_mask = create_mask(src)
start=time.time()
tgt = net(src=src,var=var,src_mask=src_mask, src_padding_mask=src_padding_mask)
# tgt=predict(net,src,var)
end=time.time()
import cProfile
# cProfile.run('predict(net,src,var)')
#flops, params = profile(net, inputs=(src,var,src_mask, src_padding_mask))
H_vector_sample=(tgt.squeeze(0)*std_H_vector)[0:turns]
H_vector_sample=torch.exp(H_vector_sample)
print(H_vector_sample)
print(end-start)

#calculate loss
formatted_value = "{:g}".format((litz_diameter*1e3).item())
n_strand=diameter**2/(litz_diameter_table[formatted_value]**2/1e6/strand_filling_factor)
k = 5.8e7
sigma = 4 * torch.pi * 1e-7
f_vector=torch.tensor([f,2*f,3*f,4*f])
delta = torch.sqrt(1 / (torch.pi*k * f_vector * sigma))
alpha = (1 + torch.tensor(1j)) / delta
a=litz_diameter/2
Ds = 2 * torch.pi * torch.real(alpha * a * (iv(1,alpha * a) / iv(0,alpha * a)))
winding_length_vector=(core_length+(position_x_sample[0:turns])*torch.pi/2+center_core_width)*4
P_prox=H_vector_sample.squeeze(1).unsqueeze(0)**2*n_strand*(winding_length_vector/k)*(Ds.unsqueeze(1)*(I_fft[1:5]**2).unsqueeze(1)/2)
Fs= 0.5 * torch.real(alpha * a * iv(0, alpha * a) / iv(1, alpha * a))
Fs=torch.cat((torch.tensor(1.0).unsqueeze(0),Fs))
R0_vector=winding_length_vector/k/(n_strand*a**2*torch.pi)
P_skin=(I_fft[0:5]**2/2*Fs).unsqueeze(1)*R0_vector.unsqueeze(0)
P_hf=torch.clone(P_skin)
P_hf[1:5,:]=P_prox+P_hf[1:5,:]
P_hf_sum=torch.sum(P_hf,dim=0)
P_winding=torch.sum(P_hf_sum,dim=0)