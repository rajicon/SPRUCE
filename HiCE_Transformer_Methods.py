import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import numpy as np


#My methods ----------------------

class NoResCrossEncoderLayer(nn.Module):
    '''
        Cross Transformer Encoder, which will be used for both context encoder and aggregator.
    '''
    def __init__(self, n_head, n_hid, att_dropout = 0.1, ffn_dropout = 0.1, res_dropout = 0.3):
        super(NoResCrossEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, n_hid, att_dropout)
        self.feed_forward = PositionwiseFeedForward(n_hid, ffn_dropout)
        self.twosublayer = NoResTwoInputSublayerConnection(n_hid, res_dropout)
        self.onesublayer = SublayerConnection(n_hid, res_dropout)
    def forward(self, x, y, mask):
        x = self.twosublayer(x, y, lambda x, y: self.self_attn(x, y, y, mask))
        return self.onesublayer(x, self.feed_forward)

class NoResTwoInputSublayerConnection(nn.Module):
    '''
        A residual connection followed by a layer norm.
    '''
    def __init__(self, size, dropout = 0.3):
        super(NoResTwoInputSublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(self.dropout(sublayer(x, y)))
        
        

class CrossEncoderLayer(nn.Module):
    '''
        Cross Transformer Encoder, which will be used for both context encoder and aggregator.
    '''
    def __init__(self, n_head, n_hid, att_dropout = 0.1, ffn_dropout = 0.1, res_dropout = 0.3):
        super(CrossEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, n_hid, att_dropout)
        self.feed_forward = PositionwiseFeedForward(n_hid, ffn_dropout)
        self.twosublayer = TwoInputSublayerConnection(n_hid, res_dropout)
        self.onesublayer = SublayerConnection(n_hid, res_dropout)
    def forward(self, x, y, mask):
        x = self.twosublayer(x, y, lambda x, y: self.self_attn(x, y, y, mask))
        return self.onesublayer(x, self.feed_forward)
        
#inspired by DUMA
class Cross_MHA_Only(nn.Module):
    '''
        Cross Transformer Encoder, which will be used for both context encoder and aggregator.
    '''
    def __init__(self, n_head, n_hid, att_dropout = 0.1, ffn_dropout = 0.1, res_dropout = 0.3):
        super(Cross_MHA_Only, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, n_hid, att_dropout)

    def forward(self, x, y, mask):
        return self.self_attn(x, y, y, mask)
      

class TwoInputSublayerConnection(nn.Module):
    '''
        A residual connection followed by a layer norm.
    '''
    def __init__(self, size, dropout = 0.3):
        super(TwoInputSublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x, y)))
        


#from HiCE ----------------------

class EncoderLayer(nn.Module):
    '''
        Transformer Encoder, which will be used for both context encoder and aggregator.
    '''
    def __init__(self, n_head, n_hid, att_dropout = 0.1, ffn_dropout = 0.1, res_dropout = 0.3):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, n_hid, att_dropout)
        self.feed_forward = PositionwiseFeedForward(n_hid, ffn_dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(n_hid, res_dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class LayerNorm(nn.Module):
    '''
        Construct a layernorm module.
    '''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_hid, dropout=0.3):
        '''
            Multihead self-attention that can calcualte mutual attention table
            based on which to aggregate embedding at different position.
        '''
        super(MultiHeadedAttention, self).__init__()
        self.d_k = n_hid // n_head
        self.h = n_head
        self.linears = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(3)])
        self.out     = nn.Linear(self.d_k * n_head, n_hid)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from n_hid => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.out(x)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            #print('scores '+str(scores.size()))
            #print('mask '+str(mask))
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    
class PositionwiseFeedForward(nn.Module):
    '''
        Implements FFN equation (1-D convolution).
    '''
    def __init__(self, n_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(n_hid, n_hid * 2)
        self.w_2 = nn.Linear(n_hid * 2, n_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
   
    
class SublayerConnection(nn.Module):
    '''
        A residual connection followed by a layer norm.
    '''
    def __init__(self, size, dropout = 0.3):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))
    
class PositionalEncoding(nn.Module):
    '''
        Implement the Position Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 1000, dropout = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_hid)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0., n_hid, 2.)) / n_hid)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0) / np.sqrt(n_hid)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.dropout(x + Variable(self.pe[:, :, :x.shape[-2]], requires_grad=False))

class PositionalAttention(nn.Module):
    '''
        A simple positional attention layer that assigns different weights for word in different relative position.
    '''
    def __init__(self, n_seq):
        super(PositionalAttention, self).__init__()
        self.pos_att = nn.Parameter(torch.ones(n_seq))
    def forward(self, x):
        # x: L * d -> d * L
        return (x.transpose(-2, -1) * self.pos_att).transpose(-2, -1)
    
class CharCNN(nn.Module):
    '''
        A simple implementation of CharCNN (Kim et al. https://arxiv.org/abs/1508.06615)
    '''
    def __init__(self, n_hid, dropout = 0.3):
        super(CharCNN, self).__init__()
        self.char_emb = nn.Embedding(26+1, n_hid)
        self.filter_num_width = [2,4,6,8]
        self.convs    = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = filter_width),
                nn.ReLU()
            ) for filter_width in self.filter_num_width])
        self.linear = nn.Linear(n_hid * len(self.filter_num_width), n_hid)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(n_hid)
    def forward(self, x):
        x = self.char_emb(x).transpose(1,2)
        conv_out = [torch.max(conv(x), dim=-1)[0] for conv in self.convs]
        conv_out = self.dropout(torch.cat(conv_out, dim=1))
        return self.norm(self.linear(conv_out))
  
#My CharCNN
#outputs the pieces for further cross att      
class MyCharCNN(nn.Module):
    '''
        A simple implementation of CharCNN (Kim et al. https://arxiv.org/abs/1508.06615)
    '''
    def __init__(self, n_hid, dropout = 0.3):
        super(MyCharCNN, self).__init__()
        self.char_emb = nn.Embedding(26+1, n_hid)
        self.filter_num_width = [2,4,6,8]
        self.convs    = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = filter_width),
                nn.ReLU()
            ) for filter_width in self.filter_num_width])
        self.linear = nn.Linear(n_hid * len(self.filter_num_width), n_hid)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(n_hid)
    def forward(self, x):
        x = self.char_emb(x).transpose(1,2)
        conv_out = [conv(x) for conv in self.convs]
        
        #print(len(conv_out))
        
        #for a in range(len(conv_out)):     
        #    print(conv_out[a].size()) #[64, 400, 18], [64, 400, 12] ...
        #    print('--')

        
        new_conv_out = self.dropout(torch.cat(conv_out, dim=2))
        new_conv_out = new_conv_out.transpose(1,2)
        #print(new_conv_out.size())
        
        #new_conv_out = new_conv_out.view(new_conv_out.size()[0], conv_out, -1)
        return new_conv_out
        
def get_qkv_transformed_data(multihead_struct, query, key, value, mask=None):
	nbatches = query.size(0)
	# 1) Do all the linear projections in batch from n_hid => h x d_k 
	query, key, value = \
		[l(x).view(nbatches, -1, multihead_struct.h, multihead_struct.d_k).transpose(1, 2)
		 for l, x in zip(multihead_struct.linears, (query, key, value))]
		 
	return query, key, value