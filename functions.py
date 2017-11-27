from nltk.tokenize import TweetTokenizer
import re
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
        self.embeddings = nn.Embedding(self.vocab_size + 1, self.emb_size)

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


    def create_emb_matrix(self, vocaboulary):
        print('importing embeddings')
        vocab, vec = torchwordemb.load_word2vec_bin("./GoogleNews-vectors-negative300.bin")
        print('imported embeddings')

        emb_mat = np.zeros((self.vocab_size + 1, self.emb_size))

        for word in vocaboulary.keys():
            if word in vocab:
                emb_mat[vocaboulary[word]] = vec[vocab[word]].numpy()
            else:
                emb_mat[vocaboulary[word]] = np.random.normal(0, 1, self.emb_size)

        #hypotetically, the one for <unk>
        emb_mat[-1] = np.random.normal(0, 1, self.emb_size)

        print('train matrices built')

        del vec
        del vocab
        gc.collect()

        print('garbage collected')

        return emb_mat

    def encode_words(self, sentence, word2ind):
        inp = np.asarray([word2ind[word] for word in sentence])
        #inp.reshape((1, 1, self.pad_len))
        inp = Variable(torch.LongTensor(inp))
        return inp









def read_data(fname, pad_len=50, padding=True):
    '''
    ItemID,Sentiment,SentimentSource,SentimentText
    :param fname:
    :param pad_len:
    :param padding:
    :return:
    '''
    train_data = []
    train_targets = []

    with open(fname) as f:
        f.readline()
        for line in f.readlines():
            train_target, train_sentence = line.strip().split(None, 1)
            train_data.append(train_sentence)
            train_targets.append(train_target)
            # print(train_target, train_sentence)
    #         if line[1] == 'positive':
    #             train_targets.append(2)
    #         elif line[1] == 'negative':
    #             train_targets.append(0)
    #         else:
    #             train_targets.append(1)

    tknzr = TweetTokenizer()
    max_len = 0

    for ind, tmp in enumerate(train_data):
        #tmp = sentence.strip().split(None, 1)[1]
        print(ind)
        tmp = re.sub("https?:?//[\w/.]+", "<URL>", tmp)
        tmp = find_emoticons(tmp)
        tmp = re.sub("[\-_#@('s ):.]", " ", tmp)
        train_data[ind] = tknzr.tokenize(tmp.lower())
        max_len = max(max_len, len(train_data[ind]))

    actual_pad = max(max_len, pad_len)
    if padding:
        for sentence in train_data:
            assert len(sentence) <= actual_pad, "tweet longer than padding"

            while len(sentence) < actual_pad:
                sentence.append("<PAD>")

    return train_data, train_targets, actual_pad


def find_emoticons(sentence):
    smile_emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose (optional)
            [D\)\]] # Mouth
        )"""

    sad_emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-']? # Nose (optional)
            [\(\[\|] # Mouth
        )"""
    funny_emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose (optional)
            [OpP] # Mouth
        )"""
    sentence = re.sub(smile_emoticons_str, "<smile>", sentence)
    sentence = re.sub(sad_emoticons_str, "<sad>", sentence)
    sentence = re.sub(funny_emoticons_str, "<funny>", sentence)

    return sentence

def create_vocab(vocab_list):
    vocab = {}
    for word in vocab_list:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

