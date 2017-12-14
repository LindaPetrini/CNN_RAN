import torch
import torch.nn as nn
from torch.autograd import Variable
import torchwordemb
import numpy as np
import gc


class CNN(nn.Module):
    def __init__(self, emb_size, pad_len, classes, vocab_size):
        super(CNN, self).__init__()

        self.emb_size = emb_size
        self.pad_len = pad_len
        self.classes = classes
        self.vocab_size = vocab_size
        self.window = 3
        self.channels = 16
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_size)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.channels, kernel_size=(self.emb_size, self.window), padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(self.channels*int(self.pad_len/2), self.classes)
        self.sm = nn.LogSoftmax()

    def forward(self, x):
        x = self.embeddings(x).view((1, 1, self.emb_size, -1))
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        out = self.sm(out)
        return out

    def init_emb(self, vocaboulary):
        emb_mat = self.create_emb_matrix(vocaboulary)
        self.embeddings.weight.data.copy_(torch.from_numpy(emb_mat))

    def init_from_txt(self, fname, vocaboulary):
        emb_mat =  np.zeros((self.vocab_size, self.emb_size))
        with open(fname, 'r', encoding='utf-8') as f:
            for l in f:
                word, vec = l.split("\t")
                vec = vec.split(",")
                emb_mat[vocaboulary[word]] = np.asarray(vec)
        self.embeddings.weight.data.copy_(torch.from_numpy(emb_mat))

    def create_emb_matrix(self, vocaboulary):
        print('importing embeddings')
        vocab, vec = torchwordemb.load_word2vec_bin("./datasets/GoogleNews-vectors-negative300.bin")
        print('imported embeddings')

        emb_mat = np.zeros((self.vocab_size, self.emb_size))

        for i, word in enumerate(vocaboulary.keys()):
            if i % 1000 == 0:
                print("Reading word ", i, "/", len(vocaboulary))
            if word in vocab:
                emb_mat[vocaboulary[word]] = vec[vocab[word]].numpy()
            else:
                emb_mat[vocaboulary[word]] = np.random.normal(0, 1, self.emb_size)

        print('train matrices built')

        del vec
        del vocab
        gc.collect()

        print('garbage collected')

        return emb_mat

    def encode_words(self, sentence, word2ind, is_cuda=False):
        inp = np.asarray([word2ind[word] for word in sentence])
        inp = Variable(torch.cuda.LongTensor(inp.tolist())) if is_cuda else Variable(torch.LongTensor(inp))
        return inp

    def emb_to_txt(self, oname: str, word2ind: dict):

        with open(oname, "w") as o:
            o.write("word\tembedding")
            for word in word2ind.keys():
                emb = self.embeddings.weight.data.numpy()[word2ind[word]]
                emb = [str(el) for el in emb]
                o.write(str(word) + "\t" + ",".join(emb) + "\n")


def read_data(fname):
    train_data = []
    train_targets = []

    with open(fname) as f:
        f.readline()
        for line in f.readlines():
            train_target, train_sentence = line.strip().split(None, 1)
            train_data.append(train_sentence.split(" "))
            train_targets.append(int(train_target))

    tweet_len = len(train_data[-1])
    
    print('Dataset size:', len(train_data))
    
    return train_data, train_targets, tweet_len


def create_vocab(vocab_list):
    vocab = {}
    for word in vocab_list:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab



