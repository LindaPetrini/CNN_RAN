
from nltk.tokenize import TweetTokenizer

fname = './2017_English_final/GOLD/Subtask_A/twitter-2016train-A.txt'
#fname2 = './2017_English_final/GOLD/Subtask_A/twitter-2016test-A.txt'

tknzr = TweetTokenizer()

train_data = []
train_targets = []

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

for ind, sentence in enumerate(train_data[:100]):
    tmp = sentence.strip().split(None, 2)[2].lower()
    print("BEFORE: ", tmp)
    train_data[ind] = tknzr.tokenize(tmp)
    print("AFTER: ", train_data[ind])
