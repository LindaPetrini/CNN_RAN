import torchwordemb

vocab, vec = torchwordemb.load_word2vec_bin("./GoogleNews-vectors-negative300.bin")

print(vec[vocab["a"]])