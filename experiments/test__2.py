import os
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn import init
from torch.nn._functions.rnn import Recurrent, StackedRNN
import torchwordemb
from test_3 import CNN
import numpy as np
from sklearn.utils import shuffle


#vocab, vec = torchwordemb.load_word2vec_bin("./GoogleNews-vectors-negative300.bin")

train_data = []
train_targets = []

fname = './2017_English_final/GOLD/Subtask_A/twitter-2016train-A.txt'
something = True
for line in open(fname):
    if something:
        something = False
    else:
        splitted_line = line.split()
        train_targets.append(1 if splitted_line[1]=="positive" else 0 if splitted_line[1]=="neutral" else -1)
        train_data.append(splitted_line[2:])

punct = [".", ",", ";", ":", '"', "@", "#"]
for ind, sentence in enumerate(train_data):

    a = ["".join([letter if letter not in punct else "" for letter in word.lower()])
         if "http" not in word else None for word in sentence]
    if None in a:
        a.remove(None)
    train_data[ind] = a

pad_len = len(max(train_data, key=lambda x: len(x)))
print(pad_len)



for sentence in train_data:
    if len(sentence) < pad_len:
        tmp = pad_len - len(sentence)
        for i in range(tmp):
            sentence.append("<hey>")


vocab, vec = {}, {}


batch_size = 10
window = 3
emb_size = 100
classes = 3

for ind, sentence in enumerate(train_data):
    for k, word in enumerate(sentence):
        if word not in vocab:
            vocab[word] = len(vocab)
            vec[vocab[word]] = np.random.normal(0, 1, emb_size)


cnn = CNN(emb_size, pad_len, classes)

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
loss = nn.NLLLoss()
#loss = nn.CrossEntropyLoss()


for ind, sentence in enumerate(train_data):
    emb_mat = np.zeros((emb_size, pad_len))
    for k, word in enumerate(sentence):
        if word in vocab:
            emb_mat[:, k] = vec[vocab[word]]
        else:
            emb_mat[:, k] = np.random.normal(0, 1, emb_size)

    inp = Variable(torch.Tensor(emb_mat).view(1, 1, emb_size, -1))
    out = cnn.forward(inp)
    expected_targ = Variable(torch.LongTensor([train_targets[ind]+1]))

    optimizer.zero_grad()
    error = loss(out, expected_targ)

    error.backward()
    optimizer.step()

    # for param in cnn.parameters():
    #     print("data: ", param.data, "\n grad: ", param.grad.data)



    print("\ntweet ", ind)
    print("expected targ: ", expected_targ.data)
    #print("output ", nn.functional.softmax(out.data))
    print("output ", out.data)
    print("error ", error)

for i in range(4):
    print("**************************************************************")
    train_data, train_targets= shuffle(train_data, train_targets, random_state=42)
    for ind, sentence in enumerate(train_data):
        emb_mat = np.zeros((emb_size, pad_len))
        for k, word in enumerate(sentence):
            if word in vocab:
                emb_mat[:, k] = vec[vocab[word]]

            else:
                emb_mat[:, k] = np.random.normal(0, 1, emb_size)

        inp = Variable(torch.Tensor(emb_mat).view(1, 1, emb_size, -1))
        out = cnn.forward(inp)

        expected_targ = Variable(torch.LongTensor([train_targets[ind] + 1]))

        optimizer.zero_grad()
        error = loss(out, expected_targ)

        error.backward()
        optimizer.step()

        if ind % 100 == 0:
            print("\ntweet ", ind)
            print("expected targ: ", expected_targ.data)
            print("output ", nn.functional.softmax(out.data))
            print("error ", error)



















