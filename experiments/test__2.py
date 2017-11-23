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
import re
import random
import gc



train_data = []
train_targets = []

fname = './2017_English_final/GOLD/Subtask_A/twitter-2016train-A.txt'

with open(fname) as f:
    f.readline()
    for line in f.readlines():
    
        train_data.append(line)
        
        line = line.strip().split()
        
        if line[1] == 'positive':
            train_targets.append(2)
        elif line[1] == 'negative':
            train_targets.append(0)
        else:
            train_targets.append(1)
        
        

punct = [".", ",", ";", ":", '"', "@", "#"]
pad_len = 0


for ind, sentence in enumerate(train_data):
    
    sentence = sentence.strip().split(None, 2)[2].lower()
    
    re.sub("https?://[\w/\.]+", "<URL>", sentence)
    
    for p in punct:
        sentence = sentence.replace(p, '')
    
    train_data[ind] = sentence.strip().split()

    pad_len = max(pad_len, len(train_data[ind]))


print(pad_len)
print(len(train_data))

for sentence in train_data:
    while len(sentence) < pad_len:
        sentence.append("<hey>")

batch_size = 10
window = 3
emb_size = 300
classes = 3



# vocab, vec = {}, {}
# for ind, sentence in enumerate(train_data):
#     for k, word in enumerate(sentence):
#         if word not in vocab:
#             vocab[word] = len(vocab)
#             vec[vocab[word]] = np.random.normal(0, 1, emb_size)

print('importing embeddings')
vocab, vec = torchwordemb.load_word2vec_bin("./GoogleNews-vectors-negative300.bin")
print('imported embeddings')


train_sentences = []

for ind, sentence in enumerate(train_data):
    
    emb_mat = np.zeros((emb_size, pad_len))
    
    for k, word in enumerate(sentence):
        if word in vocab:
            emb_mat[:, k] = vec[vocab[word]].numpy()
        else:
            emb_mat[:, k] = np.random.normal(0, 1, emb_size)
    
    train_sentences.append(emb_mat.reshape((1, 1, emb_size, pad_len)))

print('train matrices built')

del vec
del vocab
gc.collect()

print('garbage collected')



cnn = CNN(emb_size, pad_len, classes)

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

loss = nn.NLLLoss()



for ind, emb_mat in enumerate(train_sentences):
    inp = Variable(torch.Tensor(emb_mat))
    
    print("tweet ", ind)

    out = cnn.forward(inp)
    
    target = train_targets[ind]
    
    expected_targ = Variable(torch.LongTensor([target]))

    print('\tExpected - Predicted:', target, np.argmax(out.data.numpy().flatten()))

    optimizer.zero_grad()
    error = loss(out, expected_targ)

    error.backward()
    optimizer.step()
    print("\tError ", error.data.numpy().flatten())

















