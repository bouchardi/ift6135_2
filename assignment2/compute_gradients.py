#!/bin/python
# coding: utf-8

import collections
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from GRU_RNN import GRU
from simple_RNN import RNN


# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


###############################################################################
#
# DATA LOADING & PROCESSING
#
###############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# LOAD DATA
print('Loading data from '+ 'data')
raw_data = ptb_raw_data(data_path='data')
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))


###############################################################################
#
# MODEL SETUP
#
###############################################################################


###############################################################################
#
# DEFINE COMPUTATIONS FOR PROCESSING ONE EPOCH
#
###############################################################################

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    This prevents Pytorch from trying to backpropagate into previous input
    sequences when we use the final hidden states from one mini-batch as the
    initial hidden states for the next mini-batch.

    Using the final hidden states in this way makes sense when the elements of
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


def compute_average_grad(model, data, model_name):
    model.eval()

    if model_name != 'TRANSFORMER':
        hidden = model.init_hidden()
        hidden = hidden.to(device)

    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        # Prepare data
        inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        # Forward pass
        outputs, hidden = model(inputs, hidden, keep_hiddens=True)
        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)
        # Resize
        tt = torch.squeeze(targets.view(-1, model.batch_size * model.seq_len))
        out = outputs.contiguous().view(-1, model.vocab_size)
        # Compute the loss
        loss = loss_fn(out, tt)
        # Get the euclidian norm of the gradients with respect to each hidden layer
        grads = [float(torch.autograd.grad(loss, hidden, retain_graph=True)[0].norm(2).cpu().numpy()) for hidden in model.hiddens]
        # Return the result for the first and second hidden layer separately
        return grads[0::2], grads[1::2]


###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################

print("\n########## Running Main Loop ##########################")


# MAIN LOOP

RNN_PATH = 'RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.6_save_best_0'
GRU_PATH = 'GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0'

EMB_SIZE_RNN = 200
HIDDEN_SIZE_RNN = 1500
SEQ_LEN_RNN = 35
BATCH_SIZE_RNN = 20
VOCAB_SIZE_RNN = 10000
NUM_LAYERS_RNN = 2
DP_KEEP_PROB_RNN = 0.6

EMB_SIZE_GRU = 200
HIDDEN_SIZE_GRU = 1500
SEQ_LEN_GRU = 35
BATCH_SIZE_GRU = 20
VOCAB_SIZE_GRU = 10000
NUM_LAYERS_GRU = 2
DP_KEEP_PROB_GRU = 0.35

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

word_to_id, id_2_word = _build_vocab(os.path.join('data', 'ptb' + ".train.txt"))

# LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()



model_classes = [RNN, GRU]
for model_class in model_classes:
    if model_class == RNN:
        MODEL_PATH = RNN_PATH
        model = model_class(emb_size=EMB_SIZE_RNN,
                    hidden_size=HIDDEN_SIZE_RNN,
                    seq_len=SEQ_LEN_RNN,
                    batch_size=BATCH_SIZE_RNN,
                    vocab_size=VOCAB_SIZE_RNN,
                    num_layers=NUM_LAYERS_RNN,
                    dp_keep_prob=DP_KEEP_PROB_RNN)
        model_name = 'RNN'

    if model_class == GRU:
        MODEL_PATH = GRU_PATH
        model = model_class(emb_size=EMB_SIZE_GRU,
                    hidden_size=HIDDEN_SIZE_GRU,
                    seq_len=SEQ_LEN_GRU,
                    batch_size=BATCH_SIZE_GRU,
                    vocab_size=VOCAB_SIZE_GRU,
                    num_layers=NUM_LAYERS_GRU,
                    dp_keep_prob=DP_KEEP_PROB_GRU)
        model_name = 'GRU'

    load_path = os.path.join(MODEL_PATH, 'best_params.pt')
    model.load_state_dict(torch.load(load_path))

    model = model.to(device)
    first_layer_grads, second_layer_grads = compute_average_grad(model, train_data, model_name)

    print(model_name)
    print('First layer: {}'.format(first_layer_grads))
    print('Len first layer: {}'.format(len(first_layer_grads)))
    print('Second layer: {}'.format(second_layer_grads))
    print('Len second layer: {}'.format(len(second_layer_grads)))


#plt.plot(loss_array, '-o', label=model_name)

#plt.title("Norm of Gradients Vs. Timesteps")
#plt.ylabel("Norm of Gradients")
#plt.xlabel("Timestep")
#plt.legend()
#plt.savefig("Q5.2_PLOT.jpg")
