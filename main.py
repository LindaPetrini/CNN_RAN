import functions as fun
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt







train_fname = './datasets/emb_preprocessed.txt'
#test_fname = './2017_English_final/GOLD/Subtask_A/twitter-2016test-A.txt'

print("reading data")
train_data, train_targets, pad_len = fun.read_data(train_fname)
#test_data, test_targets, test_pad_len = fun.read_data(test_fname)

#assert pad_len == test_pad_len, "tweet length not matching"

print("\nnumber train data: ", len(train_data))
#print("number test data: ", len(test_data))
print("padding length: ", pad_len)


input("Enter to Continue...")

print("train ", train_data[:100])
print("targets ", train_targets[:100])

input("Enter to Continue...")

emb_size = 300
classes = 3


unique_words = [word for sentence in train_data for word in sentence]
unique_words = list(set(unique_words))
vocab_size = len(unique_words)

word2ind = fun.create_vocab(unique_words)

cnn = fun.CNN(emb_size, pad_len, classes, vocab_size)

cnn.init_emb(word2ind)

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

loss = nn.NLLLoss()


epochs = 10
errors = []

for it in range(epochs):
    train_data, train_targets = shuffle(train_data, train_targets)

    print("****************************************************************************")
    print("iter = ", it + 1)
    print("****************************************************************************")
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
        total_err += error
        if index % 200 == 0:
            print('\tExpected - Predicted:', target, np.argmax(out.data.numpy().flatten()))
            print("tweet ", index)
            print("\tError ", error.data.numpy().flatten())
    errors.append(total_err / len(train_data))
    print("Errors: ", errors)

print("errors:", errors)
plt.plot(range(epochs), errors)
plt.show()

torch.save(cnn, 'cnn.pt')



