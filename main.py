import functions as fun
import batchifier
import data
import argparse
import time
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
import os.path

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--initial', type=str, default=None,
                    help='choose initialisation(google, path_to_file, None = random)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--plot', action='store_true',
                    help='plot loss')
parser.add_argument('--save', type=str, default='./embeddings/'+str(time.time())+'/trained_emb.txt',
                    help='path to save the final model')
parser.add_argument('--dataset', type=str, default='./datasets/preprocessed.txt',
                    help='path to the dataset')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--shuffle', action='store_true',
                    help='shuffle train data every epoch')
parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                    help='batch size')
parser.add_argument('--pause', action='store_true',
                    help='not optimise embeddings for the first 5 epochs')
parser.add_argument('--pause_value', type=int, default=0,
                    help='not optimise embeddings for the first 5 epochs')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--lim', type=int, default=float('+inf'), help='set maximum number of datapoints to use')


args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


# def read_data(fname):
#     train_data = []
#     train_targets = []
#
#     with open(fname) as f:
#         f.readline()
#         for line in f.readlines():
#             train_target, train_sentence = line.strip().split(None, 1)
#             train_data.append(train_sentence.split(" "))
#             train_targets.append(int(train_target))
#
#     tweet_len = len(max(train_data, key=lambda x: len(x)))
#
#     print('Dataset size:', len(train_data))
#
#     return train_data, train_targets, tweet_len

# def create_vocab(vocab_list):
#     vocab = {}
#     for word in vocab_list:
#         if word not in vocab:
#             vocab[word] = len(vocab)
#     return vocab



print("Reading data...")
#train_data, train_targets, pad_len = read_data(args.dataset)

corpus = data.Corpus(args.dataset, cuda=args.cuda, lim=args.lim)

#train_data, train_targets, pad_len = corpus.data, corpus.target, corpus.tweet_len
# print("\nnumber train data: ", len(train_data))
# print("padding length: ", pad_len)

print("\nnumber train data: ", len(corpus.data))
print("padding length: ", corpus.tweet_len)


eval_batch_size = 10
emb_size = 300
classes = 2

train_data = batchifier.batchify(corpus.data, args.batch_size, args.cuda)
train_data_t = batchifier.batchify_target(corpus.target, args.batch_size, classes, args.cuda)

train_confusion = np.reshape([[0 for i in range(classes)] for j in range(classes)], (classes, classes))


cnn = fun.CNN(emb_size, corpus.tweet_len, args.batch_size, classes, len(corpus.dictionary.word2idx))

lossCriterion = nn.NLLLoss()

if args.cuda:
    cnn.cuda()

if args.save:
    dir = os.path.dirname(args.save)
    if not os.path.exists(dir):
        os.makedirs(dir)

if args.initial == "google":
    cnn.init_emb(corpus.dictionary.word2idx)
elif args.initial is not None:
    if os.path.exists(args.initial):
        cnn.init_from_txt(args.initial, corpus.dictionary.word2idx)
    else:
        raise FileNotFoundError("File {} doesn't exist".format(args.initial))





lr = args.lr
best_val_loss = None
best_epoch = -1
best_recall_epoch = -1
best_fitness = 0

if args.pause:
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, cnn.parameters()), lr=args.lr)
else:
    optimizer = torch.optim.Adagrad(cnn.parameters(), lr=args.lr)

def confusion_matrix(output, target, matrix, n_classes):
    _, y = torch.max(output.view(-1, n_classes), 1)
    _, t = torch.max(target.view(-1, n_classes), 1)
    t = t.data.cpu().numpy()
    y = y.data.cpu().numpy()
    assert len(t) == len(y), "target and output have different sizes"
    for i in range(len(t)):
        matrix[t[i], y[i]] += 1
    return

def recallFitness(conf_arr, n_classes):
    recall = np.zeros(n_classes)
    for i in range(len(conf_arr[0])):
        recall[i] = conf_arr[i, i] / (np.sum(conf_arr[i]))
    average_recall = np.sum(recall) / n_classes
    return average_recall

def train():
    # Turn on training mode which enables dropout.
    cnn.train()
    
    epoch_loss = 0
    total_loss = 0
    
    start_time = time.time()
    
    for batch, i in enumerate(range(0, train_data.size(0)-1, corpus.tweet_len)):
        # print("training........... ", train_data.size(0)," ", corpus.tweet_len)
        optimizer.zero_grad()
        data, targets = batchifier.get_batch(train_data, train_data_t, i, corpus.tweet_len, corpus.n_classes)
        
        #print('batch {}: {}'.format(batch+1, data.size()))
        targets = targets[-1]
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        
        cnn.zero_grad()
        
        output = cnn(data)
        
        _, index_target = torch.max(targets, 1)
        loss = lossCriterion(output.view(-1, corpus.n_classes), index_target.view(-1))
        loss.backward()
        
        optimizer.step()

        total_loss += loss.data[0]
        epoch_loss += loss.data[0]

        confusion_matrix(output, targets, train_confusion, corpus.n_classes)
        
        if batch % args.log_interval == 0:
            cur_loss = total_loss# / args.log_interval
            cur_recall = recallFitness(train_confusion, corpus.n_classes) #/ args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:2d}| {:3d}/{:3d}| ms/btc {:4.2f}| '
                  'loss {:5.2f} | Rec {:3.4f} '.format(
                epoch, batch+1, len(train_data) // corpus.tweet_len,
                              elapsed * 1000 / args.log_interval, cur_loss, cur_recall))
            total_loss = 0
            start_time = time.time()
        
        
    
    return epoch_loss * args.batch_size / train_data.size(0)

def plotter(conf_arr, epoch=0):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(conf_arr), cmap=plt.cm.jet,
                    interpolation='nearest')
    
    # print(conf_arr)
    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = ["negative", "positive"]
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + "confusion_matrix_" + str(epoch) + '.png', format='png')
    plt.close()
    return

print("START TRAINING")
# At any point you can hit Ctrl + C to break out of training early.
try:
    exec_time = time.time()
    path = "./confusion_matrixes/" +\
           "_lr" + str(args.lr) +\
           "_batchsize_" + str(args.batch_size) +\
           "_" + str(exec_time)[-3:] +\
           ("_pause" if args.pause else "") + \
           ("_google" if args.initial == 'google' else "prev" if args.initial else "") + \
           ("_shuffle/" if args.shuffle else "/")
    
    begin_time = time.time()
    
    losses = []
    
    for epoch in range(1, args.epochs + 1):
        if args.pause:
            if epoch > args.pause_value:
                cnn.encoder.weight.requires_grad = True
                optimizer = torch.optim.Adagrad(cnn.parameters(), lr=args.lr)
        epoch_start_time = time.time()
        
        #if args.shuffle:
            # print("...shuffling")
            #train_data, train_data_t = batchifier.shuffle_data(corpus, epoch, args.batch_size, corpus.n_classes, args.cuda)
            # print("...shuffled!")
        
        train_confusion = np.reshape([[0 for i in range(classes)] for j in range(classes)], (classes, classes))
        loss = train()
        
        losses.append(loss)

        elapsed = time.time() - epoch_start_time
        
        print('| epoch {:2d}| train loss: {:4f} | s {:3.3f}| '.format(epoch, loss, elapsed))
        if args.plot:
            plotter(train_confusion, epoch)

        if args.save:
            name, ext = os.path.splitext(args.save)
            cnn.emb_to_txt(name+str(epoch)+ext, corpus.dictionary.word2idx)
        

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

end_time = time.time()

print('Total Execution Time:', end_time - begin_time)

print(train_confusion)
if args.plot:
    plt.plot(range(len(losses)), losses)
    plt.savefig(
                os.path.dirname(args.save)+
                "_losstrend" +
                "_lr" + str(args.lr) +
                "_btchsize_" + str(args.batch_size) +
                "_" + str(exec_time)[-3:] +
                ("_pause" if args.pause else "") +
                ("_google" if args.initial == 'google' else "prev" if args.initial else "") + \
                ("_shuffle" if args.shuffle else "") +
                '.png'
                )

if args.save:
    cnn.emb_to_txt(args.save, corpus.dictionary.word2idx)
