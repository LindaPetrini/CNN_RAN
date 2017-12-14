import os
import torch
import numpy as np
import random


def targetToFloat(target, n_classes):
    
    cl = int(target)
    
    enc = [0.0]*n_classes
    enc[cl] = 1.0
    
    return enc, cl


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, n_classes=2, lim=float('+inf')):
        self.dictionary = Dictionary()
        self.dictionary.add_word("<pad>")
        self.n_classes = n_classes
        self.data, self.target, self.length, self.tweet_len, self.train_weights = self.tokenize_single(path, lim)
    
    def tokenize_single(self, path, lim=float('+inf')):
        assert os.path.exists(path)
        
        random.seed(1234)
        
        max_length = 0
        with open(path, 'r') as f:
            tokens = 0
            tweet_amount = 0
            for i, line in enumerate(f):
                if i+1 > lim:
                    break
                    
                target, sentence = line.strip().split(None, 1)
                words = sentence.split()
                tokens += len(words)
                if len(words) > max_length:
                    max_length = len(words)
                    
                tweet_amount += 1
                for word in words:
                    self.dictionary.add_word(word)
        
        
        tweet_len = max_length
        
        with open(path, 'r') as f:
            ids = torch.cuda.LongTensor(tokens)
            targets = torch.cuda.FloatTensor(tokens, self.n_classes)
            token = 0
            for i, line in enumerate(f):
                if i+1 > lim:
                    break
                    
                target, sentence = line.strip().split(None, 1)
                words = sentence.split()
                this_target, _ = targetToFloat(target, self.n_classes)
                
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    
                    for j in range(self.n_classes):
                        targets[token][j] = this_target[j]
                    token += 1
        
        tot = targets.sum(0)
        
        weights = tot.sum() / tot
        print(path, "tweet_len:", tweet_len, "tweet_amount", tweet_amount, "classes", tot)  # , "weightsNLL ",weights)
        
        return ids, targets, tweet_amount, tweet_len, weights
    
    def shuffle_content(self, epoch):
        l = self.tweet_len
        idx = np.arange(self.length)
        np.random.seed(epoch)
        np.random.shuffle(idx)
        new_train = torch.cuda.LongTensor(self.length * self.tweet_len)
        new_train_t = torch.cuda.FloatTensor(self.length * self.tweet_len, self.n_classes)
        for i in range(self.length):
            for n in range(l):
                new_train[l * i + n] = self.data[l * idx[i] + n]
                new_train_t[l * i + n] = self.target[l * idx[i] + n]
                # print("original ", self.train[l*idx[i]:l*idx[i]+l])
                # print("shuffled ", new_train[l*i:l*i+l])
                # input("Continue")
        self.data = new_train
        self.target = new_train_t