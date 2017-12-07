import torchwordemb
import re
from nltk.tokenize import TweetTokenizer

vocab, vec = torchwordemb.load_word2vec_bin("./GoogleNews-vectors-negative300.bin")
from gensim.models import KeyedVectors

# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
# print(model.most_similar('and', topn=3))


# print(vec[vocab["end"]])
# print(vec[vocab["a"]])
# print(vec[vocab["is"]])

train_data = []
train_targets = []

fname = './2017_English_final/GOLD/Subtask_A/twitter-2016train-A.txt'
tknzr = TweetTokenizer()

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


for ind, sentence in enumerate(train_data):

    tmp = sentence.strip().split(None, 2)[2].lower()
    #tmp = tknzr.tokenize(tmp)
    tmp = re.sub("https?:?//[\w/.]+", "<URL>", tmp)
    tmp = re.sub("-|#|@", " ", tmp)
    train_data[ind] = tknzr.tokenize(tmp)


all_words = []
unseen = []
for sentence in train_data:
    for word in sentence:
        all_words.append(word)
        if word not in vocab:

            unseen.append(word)

for el in set(unseen):
    print(el)
print("number of unseen words: ", len(set(unseen)), " numb of total words ", len(set(all_words)))