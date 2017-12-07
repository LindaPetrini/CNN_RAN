import os
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


sentences = MySentences('./2017_English_final/GOLD/Subtask_A/twitter-2016train-A.txt')  # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences)