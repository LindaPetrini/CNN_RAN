import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from torch.nn._functions.rnn import Recurrent, StackedRNN
import os
import torchwordemb


class CNN(nn.Module):
    def __init__(self, emb_size, pad_len, classes):
        super(CNN, self).__init__()
        self.emb_size = emb_size
        self.pad_len = pad_len
        self.classes = classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(100, 3), padding=1)
            #nn.BatchNorm2d(16),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)
            )

        # self.conv = nn.Conv2d(1, 16, kernel_size=(300, 3), padding=1),
        # self.norm = nn.BatchNorm2d(16),
        # self.relu = nn.ReLU(),
        # self.max_pool = nn.MaxPool2d(2)


        #self.fc = nn.Linear(16*int(self.pad_len), self.classes)
        self.fc = nn.Linear(1632, self.classes)
        self.sm = nn.LogSoftmax()

    def forward(self, x):
        # out = self.conv(x)
        # print("after conv")
        # print(out)
        #
        # out = self.relu(out)
        # print("after relu")
        # print(out)
        # out = self.max_pool(out)
        # print("after max pool")
        # print(out)


        out = self.layer1(x)
        #print("layer1 ", out.data)

        out = out.view(1, -1)
        #print(out)
        out = self.fc(out)
        #print("after linear")
        #print(out)
        out = self.sm(out)
        return out



# class RAN(nn.Module):
#
#     def __init__(self, input_size, hidden_size, nlayers=1, dropout=0.5):
#         super().__init__()
#         if nlayers > 1:
#             raise NotImplementedError("TODO: nlayers > 1")
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.nlayers = nlayers
#         self.dropout = dropout
#
#         self.w_cx = nn.Parameter(torch.Tensor(hidden_size, input_size))
#         self.w_ic = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.w_ix = nn.Parameter(torch.Tensor(hidden_size, input_size))
#         self.w_fc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.w_fx = nn.Parameter(torch.Tensor(hidden_size, input_size))
#
#         self.b_cx = nn.Parameter(torch.Tensor(hidden_size))
#         self.b_ic = nn.Parameter(torch.Tensor(hidden_size))
#         self.b_ix = nn.Parameter(torch.Tensor(hidden_size))
#         self.b_fc = nn.Parameter(torch.Tensor(hidden_size))
#         self.b_fx = nn.Parameter(torch.Tensor(hidden_size))
#
#         self.weights = self.w_cx, self.w_ic, self.w_ix, self.w_fc, self.w_fx
#         for w in self.weights:
#             init.xavier_uniform(w)
#
#         self.biases = self.b_cx, self.b_ic, self.b_ix, self.b_fc, self.b_fx
#         for b in self.biases:
#             b.data.fill_(0)
#
#     def forward(self, input, hidden):
#         layer = (Recurrent(RANCell), )
#         func = StackedRNN(layer, self.nlayers, dropout=self.dropout)
#         hidden, output = func(input, hidden, ((self.weights, self.biases), ))
#         return output, hidden
#
#
# def RANCell(input, hidden, weights, biases):
#     w_cx, w_ic, w_ix, w_fc, w_fx = weights
#     b_cx, b_ic, b_ix, b_fc, b_fx = biases
#
#     ctilde_t = F.linear(input, w_cx, b_cx)
#     i_t = F.sigmoid(F.linear(hidden, w_ic, b_ic) + F.linear(input, w_ix, b_ix))
#     f_t = F.sigmoid(F.linear(hidden, w_fc, b_fc) + F.linear(input, w_fx, b_fx))
#     c_t = i_t * ctilde_t + f_t * hidden
#     h_t = F.tanh(c_t)
#
#     return h_t






















