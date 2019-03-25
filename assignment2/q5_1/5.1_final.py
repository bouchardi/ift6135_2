#!/bin/python
# coding: utf-8

# Code outline/scaffold for
# ASSIGNMENT 2: RNNs, Attention, and Optimization
# By Tegan Maharaj, David Krueger, and Chin-Wei Huang
# IFT6135 at University of Montreal
# Winter 2019
#
# based on code from:
#    https://github.com/deeplearningathome/pytorch-language-model/blob/master/reader.py
#    https://github.com/ceshine/examples/blob/master/word_language_model/main.py
#    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_word_ptb.py
#    https://github.com/harvardnlp/annotated-transformer

# GENERAL INSTRUCTIONS:
#    - ! IMPORTANT!
#      Unless we're otherwise notified we will run exactly this code, importing
#      your models from models.py to test them. If you find it necessary to
#      modify or replace this script (e.g. if you are using TensorFlow), you
#      must justify this decision in your report, and contact the TAs as soon as
#      possible to let them know. You are free to modify/add to this script for
#      your own purposes (e.g. monitoring, plotting, further hyperparameter
#      tuning than what is required), but remember that unless we're otherwise
#      notified we will run this code as it is given to you, NOT with your
#      modifications.
#    - We encourage you to read and understand this code; there are some notes
#      and comments to help you.
#    - Typically, all of your code to submit should be written in models.py;
#      see further instructions at the top of that file / in TODOs.
#          - RNN recurrent unit
#          - GRU recurrent unit
#          - Multi-head attention for the Transformer
#    - Other than this file and models.py, you will probably also write two
#      scripts. Include these and any other code you write in your git repo for
#      submission:
#          - Plotting (learning curves, loss w.r.t. time, gradients w.r.t. hiddens)
#          - Loading and running a saved model (computing gradients w.r.t. hiddens,
#            and for sampling from the model)

# PROBLEM-SPECIFIC INSTRUCTIONS:
#    - For Problems 1-3, paste the code for the RNN, GRU, and Multi-Head attention
#      respectively in your report, in a monospace font.
#    - For Problem 4.1 (model comparison), the hyperparameter settings you should run are as follows:
#          --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
#          --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
#          --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
#    - In those experiments, you should expect to see approximately the following
#      perplexities:
#                  RNN: train:  120  val: 157
#                  GRU: train:   65  val: 104
#          TRANSFORMER:  train:  67  val: 146
#    - For Problem 4.2 (exploration of optimizers), you will make use of the
#      experiments from 4.1, and should additionally run the following experiments:
#          --model=RNN --optimizer=SGD --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9
#          --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35
#          --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9
#    - For Problem 4.3 (exloration of hyperparameters), do your best to get
#      better validation perplexities than the settings given for 4.1. You may
#      try any combination of the hyperparameters included as arguments in this
#      script's ArgumentParser, but do not implement any additional
#      regularizers/features. You may (and will probably want to) run a lot of
#      different things for just 1-5 epochs when you are trying things out, but
#      you must report at least 3 experiments on each architecture that have run
#      for at least 40 epochs.
#    - For Problem 5, perform all computations / plots based on saved models
#      from Problem 4.1. NOTE this means you don't have to save the models for
#      your exploration, which can make things go faster. (Of course
#      you can still save them if you like; just add the flag --save_best).
#    - For Problem 5.1, you can modify the loss computation in this script
#      (search for "LOSS COMPUTATION" to find the appropriate line. Remember to
#      submit your code.
#    - For Problem 5.3, you must implement the generate method of the RNN and
#      GRU.  Implementing this method is not considered part of problems 1/2
#      respectively, and will be graded as part of Problem 5.3


import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# NOTE ==============================================
# This is where your models are imported
from GRU_RNN import GRU
from simple_RNN import RNN
from attention import make_model as TRANSFORMER


##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

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
print('Loading data')
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


def avg_loss(model, data, model_name):
    model.eval()
    if model_name != 'TRANSFORMER':
        hidden = model.init_hidden()
        hidden = hidden.to(device)
        BATCH_SIZE = model.batch_size
        SEQ_LEN = model.seq_len
    else:
        BATCH_SIZE = BATCH_SIZE_TRANSFORMER
        SEQ_LEN = SEQ_LEN_TRANSFORMER
        
    losses = np.empty((0, 35))
    
    
    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptb_iterator(data, BATCH_SIZE, SEQ_LEN)):
        if model_name == 'TRANSFORMER':
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1,0)
        else:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
        # LOSS COMPUTATION
        # This line currently averages across all the sequences in a mini-batch
        # and all time-steps of the sequences.
        # For problem 5.3, you will (instead) need to compute the average loss
        #at each time-step separately.
        losses_batch = []
        for (output, label) in zip(outputs, targets):
            l = loss_fn(output, label)
            losses_batch.append(l.data.item())
        losses = np.vstack((losses, losses_batch))
    return np.mean(losses, axis=0)



###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################

print("\n########## Running Main Loop ##########################")


# MAIN LOOP

RNN_PATH = 'RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.6_save_best_0'
GRU_PATH = 'GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0'
TRANSFORMER_PATH = 'TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=.9_save_best_0'

EMB_SIZE_RNN = 200
HIDDEN_SIZE_RNN = 1500
SEQ_LEN_RNN = 35
BATCH_SIZE_RNN = 20
VOCAB_SIZE_RNN = 10000
NUM_LAYERS_RNN = 2
DP_KEEP_PROB_RNN = 0.9

EMB_SIZE_GRU = 200
HIDDEN_SIZE_GRU = 1500
SEQ_LEN_GRU = 35
BATCH_SIZE_GRU = 20
VOCAB_SIZE_GRU = 10000
NUM_LAYERS_GRU = 2
DP_KEEP_PROB_GRU = 0.35


HIDDEN_SIZE_TRANSFORMER = 512
VOCAB_SIZE_TRANSFORMER = 10000
NUM_LAYERS_TRANSFORMER = 2
DP_KEEP_PROB_TRANSFORMER = 0.9
BATCH_SIZE_TRANSFORMER = 128
SEQ_LEN_TRANSFORMER = 35


# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

word_to_id, id_2_word = _build_vocab(os.path.join('data', 'ptb' + ".train.txt"))

# LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()



model_classes = [RNN, GRU, TRANSFORMER]
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

    if model_class == TRANSFORMER:
        MODEL_PATH = TRANSFORMER_PATH
        model = model_class(vocab_size=VOCAB_SIZE_TRANSFORMER, n_units=HIDDEN_SIZE_TRANSFORMER,
                            n_blocks=NUM_LAYERS_TRANSFORMER, dropout=1.-DP_KEEP_PROB_TRANSFORMER)
        model_name = 'TRANSFORMER'
    load_path = os.path.join(MODEL_PATH, 'best_params.pt')
    model.load_state_dict(torch.load(load_path))

    if model_class != TRANSFORMER:
        hidden = model.init_hidden()
        hidden = hidden.to(device)
    model = model.to(device)
    loss_array = avg_loss(model, valid_data, model_name)
    plt.plot(loss_array, '-o', label=model_name)
plt.title("Validation Loss Vs. Timesteps")
plt.ylabel("Validation Loss")
plt.xlabel("Timestep")
plt.legend()
plt.savefig("Q5.1_PLOT.jpg")

