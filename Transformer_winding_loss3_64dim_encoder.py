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

warnings.filterwarnings('ignore')

np.random.seed(1234)


# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')




# Load the dataset
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
def load_dataset(data_length=53,str='3F4'):
    # Load .csv Files
    gap_height= pd.read_csv('processed2d/gap_height_par_correct6000.csv',header=None)
    gap_height = np.array(gap_height)
    H_vector= pd.read_csv('processed2d/H_vector_correct6000.csv',header=None)
    H_vector = np.array(H_vector)
    position_x_shifted= pd.read_csv('processed2d/position_x_shifted_correct6000.csv',header=None)
    position_x_shifted = np.array(position_x_shifted)
    position_y= pd.read_csv('processed2d/position_y_correct6000.csv',header=None)
    position_y= np.array(position_y)
    window_height= pd.read_csv('processed2d/window_height_correct6000.csv',header=None)
    window_height = np.array(window_height)
    window_width= pd.read_csv('processed2d/window_width_correct6000.csv',header=None)
    window_width = np.array(window_width)
    core_height= pd.read_csv('processed2d/core_height_correct6000.csv',header=None)
    core_height = np.array(core_height)
    outer_limb_width= pd.read_csv('processed2d/outer_limb_width_correct6000.csv',header=None)
    outer_limb_width = np.array(outer_limb_width)

    # Format data into tensors
    mask = ~np.isnan(H_vector)
    turns=np.count_nonzero(mask,axis=1)
    turns=torch.from_numpy(turns).float().view(-1, 1)
    gap_height = torch.from_numpy(gap_height).float().view(-1, 1)
    H_vector = torch.from_numpy(H_vector).float().view(-1, data_length)
    #H_vector=torch.square(H_vector)
    H_vector = torch.log(H_vector)
    H_vector=torch.nan_to_num(H_vector,nan=0)
    position_x_shifted = torch.from_numpy(position_x_shifted).float().view(-1, data_length, 1)
    position_x_shifted = torch.nan_to_num(position_x_shifted, nan=0.0)
    position_y = torch.from_numpy(position_y).float().view(-1, data_length, 1)
    position_y = torch.nan_to_num(position_y, nan=0.0)
    window_height=torch.from_numpy(window_height).float().view(-1, 1)
    window_width=torch.from_numpy(window_width).float().view(-1, 1)
    core_height=torch.from_numpy(core_height).float().view(-1, 1)
    outer_limb_width=torch.from_numpy(outer_limb_width).float().view(-1, 1)
    diameter = (position_y[:,1]-position_y[:,0])

    # plt.scatter(np.linspace(1,data_length,num=data_length),in_B[300,:,0])

    # Normalize
    gap_height = (gap_height ) / torch.std(gap_height)
    H_vector = (H_vector) / torch.std(H_vector)
    position_x_shifted = (position_x_shifted ) / torch.std(position_x_shifted)
    position_y = (position_y) / torch.std(position_y)
    window_height= (window_height ) / torch.std(window_height)
    window_width= (window_width ) / torch.std(window_width)
    core_height=core_height/torch.std(core_height)
    outer_limb_width= outer_limb_width / torch.std(outer_limb_width)
    diameter=diameter/torch.std(diameter)

    #add head
    head = 0.1 * torch.ones(H_vector.size()[0], 1)
    H_vector_head = torch.cat((head, H_vector), dim=1)
    H_vector_head=H_vector_head[:,:-1]

    # Save the normalization coefficients for reproducing the output sequences
    # For model deployment, all the coefficients need to be saved.
    #normH = [torch.mean(out_H), torch.std(out_H)]

    print(gap_height.size())

    return torch.utils.data.TensorDataset(gap_height, window_height,  window_width,position_x_shifted,position_y,H_vector,H_vector_head,turns,core_height,outer_limb_width,diameter)


def init_weights(m):
    if isinstance(m, nn.Linear):
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)
def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

        #   Args:
        #    input_size: int, number of input variables. 1 if univariate.
        #    dec_seq_len: int, the length of the input sequence fed to the decoder
        #    max_seq_len: int, length of the longest sequence the model will receive. Used in positional encoding.
        #    out_seq_len: int, the length of the model's output (i.e. the target sequence length)
        #    dim_val: int, aka d_model. All sub-layers in the model produce outputs of dimension dim_val
        #    n_encoder_layers: int, number of stacked encoder layers in the encoder
        #    n_decoder_layers: int, number of stacked encoder layers in the decoder
        #    n_heads: int, the number of attention heads (aka parallel attention layers)
        #    dropout_encoder: float, the dropout rate of the encoder
        #    dropout_decoder: float, the dropout rate of the decoder
        #    dropout_pos_enc: float, the dropout rate of the positional encoder
        #    dim_feedforward_encoder: int, number of neurons in the linear layer of the encoder
        #    dim_feedforward_decoder: int, number of neurons in the linear layer of the decoder
        #    dim_feedforward_projecter :int, number of neurons in the linear layer of the projecter
        #    num_var: int, number of additional input variables of the projector

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

    def forward(self, src: Tensor, var: Tensor,src_mask,  src_padding_mask) -> Tensor:
        ########################
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src,mask=src_mask,src_key_padding_mask=src_padding_mask)
        enc_seq_len = 53

        var = var.unsqueeze(1).repeat(1,enc_seq_len,1)
        src = self.projector(torch.cat((src,var),dim=2))

        output= self.projector_directout(torch.cat((src, var), dim=2))
        #
        # tgt = self.decoder_input_layer(tgt.unsqueeze(2))
        # tgt = self.positional_encoding_layer(tgt)
        # batch_size = src.size()[0]
        # # tgt_mask = generate_square_subsequent_mask(sz1=self.out_seq_len, sz2=self.out_seq_len)
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

def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len,tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)


    src_padding_mask = ((src[:,:,0]+src[:,:,1]) == 0)
    tgt_padding_mask = (tgt == 0)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# Reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
NUM_EPOCH = 100000
BATCH_SIZE = 10000
DECAY_EPOCH = 300
DECAY_RATIO = 0.9
LR_INI = 0.001
SEQUENCE_LENGTH=53

# Select GPU as default device
device = torch.device("cpu")

# Load dataset
dataset = load_dataset()
sampling_size = len(dataset) // 10
random_sequence = torch.randperm(len(dataset))[:sampling_size]
#dataset = torch.utils.data.TensorDataset(*dataset[random_sequence])
#dataset = torch.utils.data.TensorDataset(*dataset[0:1440])

# Split the dataset
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
kwargs = {'num_workers': 0, 'pin_memory': False, 'pin_memory_device': "cuda"}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=30, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE+1, shuffle=False, **kwargs)

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

# Log the number of parameters
print("Number of parameters: ", count_parameters(net))
# Setup optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR_INI,betas=(0.9, 0.999))
# optimizer = optim.RMSprop(net.parameters(), lr=LR_INI)
#optimizer = torch.optim.SGD(net.parameters(), lr=LR_INI)

prev_loss=0
reset=0

# Train the network
for epoch_i in range(NUM_EPOCH):

    # Train for one epoch
    epoch_train_loss = 0
    net.train()
    optimizer.param_groups[0]['lr'] = LR_INI * (DECAY_RATIO ** (0 + epoch_i // DECAY_EPOCH))
    # material=torch.tensor([0.8,0.4])
    for gap_height, window_height,  window_width,position_x_shifted,position_y,H_vector,H_vector_head,turns,core_height,outer_limb_width,diameter in train_loader:
        optimizer.zero_grad()
        src=torch.cat((position_x_shifted.to(device),position_y),dim=2)
        tgt=H_vector_head.to(device)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)
        # net.zero_grad()
        output = net(src=src,
                     var=torch.cat((gap_height.to(device), window_height.to(device), window_width.to(device),turns,diameter), dim=1),
                     src_mask=src_mask,  src_padding_mask=src_padding_mask )
        loss = criterion(output.squeeze(2)[:,:],
                         H_vector.to(device))  # mind dimension.
        # select = H_vector != 0
        # pred = output.squeeze(2)[:, :-1][select]
        # loss = criterion(pred,
        #                  H_vector[select])  # mind dimension.
        #mind dimension.
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
        optimizer.step()
        epoch_train_loss += loss.item()

    # Compute validation
    with torch.no_grad():
        net.eval()
        epoch_valid_loss = 0
        for gap_height, window_height,  window_width,position_x_shifted,position_y,H_vector,H_vector_head,turns,core_height,outer_limb_width,diameter in valid_loader:
            src = torch.cat((position_x_shifted.to(device), position_y), dim=2)
            tgt = H_vector_head.to(device)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)

            output = net(src=src,
                         var=torch.cat(
                             (gap_height.to(device), window_height.to(device), window_width.to(device), turns,diameter), dim=1),
                         src_mask=src_mask, src_padding_mask=src_padding_mask)

            # test_src=torch.cat((position_x_shifted.to(device), position_y), dim=2)[0].unsqueeze(0)
            # test_tgt=torch.tensor([1.0,1.0]).unsqueeze(0)
            # test_var=torch.cat((gap_height.to(device), window_height.to(device), window_width.to(device)), dim=1)[0].unsqueeze(0)
            # output = net(src=test_src, tgt=test_tgt, var=test_var)

            # select = H_vector != 0
            # pred = output.squeeze(2)[:,:][select]
            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax.scatter(H_vector[select], pred-H_vector[select])
            # ax.set_yscale("log")
            # ax.set_xscale('log')

            # predict_H_tot=torch.sum((torch.mul(output.squeeze(2), select)*4),dim=1)
            # H_tot = torch.sum((torch.mul(H_vector, select) * 4), dim=1)
            # torch.mean((H_tot-predict_H_tot)/H_tot)
            # plt.scatter(H_tot,(H_tot-predict_H_tot)/H_tot)

            # loss = criterion(pred,
            #                  H_vector[select])  # mind dimension.


            loss = criterion(output.squeeze(2)[:,:],
                             H_vector.to(device))  # mind dimension.
            epoch_valid_loss += loss.item()
            #plt.scatter(H_vector,(output.squeeze(2)-H_vector))
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.scatter(in_F.squeeze(1), in_B_max.squeeze(1), Ploss_pre.squeeze(1))
    # ax.set_xlabel('f/std')
    # ax.set_ylabel('B/std')
    # ax.set_zlabel('log(P/std)')
    if (epoch_i + 1) % 10 == 0:
        print(f"Epoch {epoch_i + 1:2d} "
              f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
              f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")
        #torch.save('winding_loss.pt')
        # if reset==1:
        #     optimizer = optim.Adam(net.parameters(), lr=LR_INI, betas=(0.9, 0.999))
        #     reset=0
        if (abs(prev_loss-epoch_valid_loss)<0.0001)&(epoch_valid_loss / len(valid_dataset) * 1e5>10000):
            net.projector_material.apply(init_weights_xavier)
            net.projector.apply(init_weights_xavier)
            # optimizer = torch.optim.SGD(net.parameters(), lr=LR_INI*10)
            reset=1
        prev_loss = epoch_valid_loss
    if epoch_i==5800:
        print(f'ss')

    # from thop import profile
    #
    # src = torch.cat((position_x_shifted.to(device), position_y), dim=2)
    # tgt = H_vector
    # var = torch.cat((gap_height.to(device), window_height.to(device), window_width.to(device)), dim=1)
    # device = device
    # macs, params = profile(net, inputs=(src, tgt, var, device))
    # print(macs)

