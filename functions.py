import torch
import torch.nn as nn
from torch.autograd import Variable
import torchwordemb
import numpy as np
import gc


class CNN(nn.Module):
    def __init__(self, emb_size, pad_len, batch_size, classes, vocab_size):
        super(CNN, self).__init__()
        
        self.filters_sizes = [2, 4, 6]
        self.channels = 10
        
        
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.pad_len = pad_len
        self.classes = classes
        self.vocab_size = vocab_size
        
        
        self.encoder = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)
        
        self.conv_layers = []
        
        for i, filter in enumerate(self.filters_sizes):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(1, self.channels, kernel_size=(self.emb_size, filter), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((1, self.pad_len -filter +1))
                ).cuda()
            )

        self.do = nn.Dropout(0.5)
        
        self.fc = nn.Linear(self.channels*len(self.filters_sizes), self.classes)
        
        self.sm = nn.LogSoftmax()

    def forward(self, x):
        x = self.encoder(x).view((self.batch_size, 1, self.emb_size, -1))
        
        features = []
        
        for i, conv in enumerate(self.conv_layers):
            c = conv(x)
            features.append(c.view(c.size(0), -1))
        
            
        out = torch.cat(features, dim=1)
        
        #out = out.view(out.size(0), -1)
        
        out = self.do(out)
        
        out = self.fc(out)

        out = self.sm(out)
        
        return out


    
    def init_emb(self, vocaboulary):
        emb_mat = self.create_emb_matrix(vocaboulary)
        self.encoder.weight.data.copy_(torch.from_numpy(emb_mat))

    def init_from_txt(self, fname, vocaboulary):
        emb_mat =  np.zeros((self.vocab_size, self.emb_size))
        with open(fname, 'r', encoding='utf-8') as f:
            for l in f:
                word, vec = l.split("\t")
                vec = vec.split(",")
                emb_mat[vocaboulary[word]] = np.asarray(vec)
        self.encoder.weight.data.copy_(torch.from_numpy(emb_mat))

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
    

    def encode_words(self, sentence, word2ind, is_cuda=True):
        inp = np.asarray([word2ind[word] for word in sentence])
        inp = Variable(torch.cuda.LongTensor(inp.tolist())) if is_cuda else Variable(torch.LongTensor(inp))
        return inp

    def emb_to_txt(self, oname: str, word2ind: dict):
    
        embeddings = self.encoder.weight.data.cpu().numpy()
        with open(oname, "w") as o:
            o.write("#word\tembedding\n")
            for word in word2ind.keys():
                emb = embeddings[word2ind[word]]
                emb = [str(el) for el in emb]
                o.write(str(word) + "\t" + ", ".join(emb) + "\n")






