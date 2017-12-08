import functions as fun
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

train_fname = './datasets/emb_preprocessed.txt'

print("Reading data...")
train_data, train_targets, pad_len = fun.read_data(train_fname)


print("\nnumber train data: ", len(train_data))
print("padding length: ", pad_len)

input("Enter to Sample Dataset...")

print("train ", train_data[:100])
print("targets ", train_targets[:100])

input("Enter to Continue initializing the model...")

emb_size = 300
classes = 2

unique_words = [word for sentence in train_data for word in sentence]
unique_words = list(set(unique_words))
vocab_size = len(unique_words)

word2ind = fun.create_vocab(unique_words)

cnn = fun.CNN(emb_size, pad_len, classes, vocab_size)

#cnn.init_emb(word2ind)
cnn.init_from_txt("trained_emb.txt", word2ind)

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

loss = nn.NLLLoss()

epochs = 100
errors = []

for it in range(epochs):
    train_data, train_targets = shuffle(train_data, train_targets)

    print("*"*100)
    print("iter = ", it + 1)
    print("*" * 100)
    total_err = 0

    for index, sentence in enumerate(train_data):
        inp = cnn.encode_words(sentence, word2ind)
        out = cnn.forward(inp)
        target = train_targets[index]
        expected_targ = Variable(torch.LongTensor([target]))

        optimizer.zero_grad()

        error = loss(out, expected_targ)
        error.backward()
        optimizer.step()
        total_err += error.data.numpy()

        if index % 20 == 0:
            print('\tExpected - Predicted:', target, np.argmax(out.data.numpy().flatten()))
            print("tweet ", index)
            print("\tError ", error.data.numpy().flatten())
    errors += [total_err / len(train_data)]
    print("Errors: ", errors)

print("errors:", errors)
plt.plot(range(epochs), errors)
plt.show()

cnn.emb_to_txt("trained_emb.txt", word2ind)
