import re
import string
from nltk.tokenize import TweetTokenizer


def read_data(fname, oname, padding=True):
    '''
    ItemID,Sentiment,SentimentSource,SentimentText
    :param fname:
    :param pad_len:
    :param padding:
    :return:
    '''
    MAX_SIZE = 40
    train_data = []
    train_targets = []
    printable = set(string.printable)

    with open(fname, encoding='utf-8') as f:
        f.readline()
        for line in f.readlines():
            train_target, train_sentence = line.strip().split(None, 1)
            train_sentence = "".join(filter(lambda x: x in printable, train_sentence))
            train_data.append(train_sentence)
            train_targets.append(train_target)

            
            # print(train_target, train_sentence)
    #         if line[1] == 'positive':
    #             train_targets.append(2)
    #         elif line[1] == 'negative':
    #             train_targets.append(0)
    #         else:
    #             train_targets.append(1)

    tknzr = TweetTokenizer(reduce_len=True)
    max_len = 0
    longest_sent = ""
    longest_ind = -1
    train_data_filter = []
    train_targets_filter = []
    
    for ind, tmp in enumerate(train_data):
        #tmp = sentence.strip().split(None, 1)[1]
        #print(ind)

        tmp = re.sub("https?:?//[\w/.]+", "<URL>", tmp)
        tmp = find_emoticons(tmp)
        tmp = re.sub('[\-_"#@(),!*;:.\[\]~]', " ", tmp)
        tmp = tmp.replace("?", "")
        tmp = tmp.replace("&", "")
        tmp = tmp.replace("'s", "")
        tmp = tmp.replace("&quot", "")
        tmp = tmp.replace("&amp", "")
        tmp = tmp.replace("'", " ")
        tmp = tmp.replace("\t", " ")
        tkn = tknzr.tokenize(tmp.lower())
        # print(tkn)
        if len(tkn) <= MAX_SIZE:
            train_data_filter.append(tkn)
            train_targets_filter.append(train_targets[ind])
            if len(tkn) > max_len:
                print(tkn)
                longest_sent = tmp
                longest_ind = len(train_data_filter)-1
            max_len = max(max_len, len(train_data_filter[-1]))

    actual_pad = max_len #max(max_len, pad_len)
    print("actual pad", actual_pad)
    print("longest sent ", longest_sent)
    print("longest ind", longest_ind)
    if padding:
        for sentence in train_data_filter:
            assert len(sentence) <= actual_pad, "tweet longer than padding"

            while len(sentence) < actual_pad:
                sentence.append("<pad>")

    #assert len(train_data_filter) == len(train_targets), "number of targets differ number of tweets"
    #assert len(train_data_filter[0]) == actual_pad, "actual_pad isn't respected on first element"
    print("num data ", len(train_data_filter))
    print("num targets ", len(train_targets_filter))
    with open(oname, "w") as o:
        for i in range(len(train_data_filter)):
            o.write(train_targets_filter[i]+"\t"+" ".join(train_data_filter[i])+"\n")

    return train_data_filter, train_targets_filter, actual_pad


def find_emoticons(sentence):
    smile_emoticons_str = r"""(?:[:=;][oO\-]?[D\)\]])"""
    sad_emoticons_str = r"""(?:[:=;][oO\-']?[\(\[\|])"""
    funny_emoticons_str = r"""(?:[:=;@][oO\-]?[Oo0pP])"""
    sentence = re.sub(smile_emoticons_str, "<smile>", sentence)
    sentence = re.sub(sad_emoticons_str, "<sad>", sentence)
    sentence = re.sub(funny_emoticons_str, "<funny>", sentence)

    return sentence

#read_data("./emb_dataset.txt","./emb_preprocessed.txt")
read_data("./dataset.txt","preocessed.txt")