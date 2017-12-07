import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from torch.nn._functions.rnn import Recurrent, StackedRNN
import torchwordemb
from test_3 import CNN
import numpy as np
import re
import random
import gc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
form twittToken import twokenize



# train_data = []
# train_targets = []
#
# fname = './2017_English_final/GOLD/Subtask_A/twitter-2016train-A.txt'
# fname2 = './2017_English_final/GOLD/Subtask_A/twitter-2016test-A.txt'
#
# with open(fname) as f:
#     f.readline()
#     for line in f.readlines():
#
#         train_data.append(line)
#
#         line = line.strip().split()
#
#         if line[1] == 'positive':
#             train_targets.append(2)
#         elif line[1] == 'negative':
#             train_targets.append(0)
#         else:
#             train_targets.append(1)



# punct = [".", ",", ";", ":", '"', "@", "#", "~", "(", ")", "?", "!", "'", "$"]
# smiles = [":(?=\))", ";(?=\))", "=(?=\))"]
# sad_faces = [":(?=\()", ":(?<=D)", ";(?=\()", "=(?=\()"]
# lols = [":(?=D)", "lol", "LOL"]
# pad_len = 0


# for ind, sentence in enumerate(train_data):
#
#     tmp = sentence.strip().split(None, 2)[2].lower()
#
#
#     tmp = re.sub("https?:?//[\w/.]+", "<URL>", tmp)
#     for smile in smiles:
#         tmp = re.sub(smile, "<smile>", tmp)
#     for sad in sad_faces:
#         tmp = re.sub(sad, "<sad>", tmp)
#     for lol in lols:
#         tmp = re.sub(lol, "<lol>", tmp)
#
#     tmp = re.sub("-", " ", tmp)
#     tmp = re.sub("/", " ", tmp)
#     tmp = re.sub("_", " ", tmp)
#
#     for p in punct:
#         tmp = tmp.replace(p, '')
#
#     train_data[ind] = tmp.strip().split()
#
#
#     pad_len = max(pad_len, len(train_data[ind]))


#
# print(pad_len)
# print(len(train_data))
#
# for sentence in train_data:
#     while len(sentence) < pad_len:
#         sentence.append("<hey>")

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

epochs = 10
errors = []
for it in range(epochs):

    train_sentences, train_targets = shuffle(train_sentences, train_targets)
    print("***********************************************************")
    print("iter = ", it + 1)
    print("***********************************************************")
    total_err = 0
    for ind, emb_mat in enumerate(train_sentences):
        inp = Variable(torch.Tensor(emb_mat))
        out = cnn.forward(inp)
        target = train_targets[ind]
        expected_targ = Variable(torch.LongTensor([target]))
        optimizer.zero_grad()
        error = loss(out, expected_targ)
        error.backward()
        optimizer.step()
        total_err += error
        if ind % 100 == 0:
            print('\tExpected - Predicted:', target, np.argmax(out.data.numpy().flatten()))
            print("tweet ", ind)
            print("\tError ", error.data.numpy().flatten())
    errors.append(total_err/len(train_data))
    print("Errors: ", errors)

print("errors:", errors)
plt.plot(range(epochs), errors)
plt.show()

















