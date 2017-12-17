# coding: utf-8

from torch.autograd import Variable



# print("len of train corpus  ",len(corpus.train))
# print(corpus.train[:20])
# print(corpus.train_t[:20])
# input("Press Enter to continue with batching...")

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    # print("batchified dims ",data.size(), " num batch ",nbatch)
    
    
    return data


def batchify_target(data, bsz, n_classes, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1, n_classes).transpose(0, 1).contiguous()
    if cuda:
        data = data.cuda()
    # print("batchified dims ",data.size(), " num batch ",nbatch)
    return data


def shuffle_data(corpus, epoch, batch_size, n_classes, cuda):
    corpus.shuffle_content(epoch)

    train_data = batchify(corpus.data, batch_size, cuda)
    train_data_t = batchify_target(corpus.target, batch_size, n_classes, cuda)
    return train_data, train_data_t



# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, targets, i, bptt, n_class, evaluation=False):
    seq_len = min(bptt, len(source) - i)
    data = Variable(source[i:i + seq_len], volatile=evaluation)
    target = Variable(targets[i:i + seq_len, :].view(seq_len, -1, n_class))
    return data, target

