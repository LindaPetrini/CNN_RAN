import functions as fun
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--initial', type=str, choices=["google", "prev"], default="google",
                    help='choose initialisation(prev, google')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--plot', action='store_true',
                    help='plot loss')
parser.add_argument('--save', type=str, default='trained_emb.txt',
                    help='path to save the final model')

args = parser.parse_args()

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

if args.initial == "google":
    cnn.init_emb(word2ind)
else:
    cnn.init_from_txt("trained_emb.txt", word2ind)

optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)

loss = nn.NLLLoss()

errors = []

for it in range(args.epochs):
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
if args.plot:
    plt.plot(range(args.epochs), errors)
    plt.show()

if args.save:
    cnn.emb_to_txt(args.save, word2ind)
