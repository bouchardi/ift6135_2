import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is
# what the main script expects. If you modify the contract,
# you must justify that choice, note it in your report, and notify the TAs
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention.


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).

    inputs:
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
# Implement a stacked vanilla RNN with Tanh nonlinearities.
class RNN(nn.Module):
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size,
               num_layers=2, dp_keep_prob=0.5):

    """
    emb_size:     The numvwe of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.num_layers = num_layers
    self.batch_size = batch_size
    self.vocab_size = vocab_size

    # To compute the average gradient
    self.hiddens = []

    # Embedding encoder
    self.embedding = nn.Embedding(vocab_size, emb_size)

    # N stacked recurrent layers (first layer has different input size)
    k = math.sqrt(1.0/self.hidden_size)
    linear_W0 = nn.Linear(emb_size, hidden_size)
    self.init_weights_uniform(nn.Linear(hidden_size, hidden_size), low=-k, high=k)
    linear_W = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]
    for layer in linear_W:
        self.init_weights_uniform(layer, low=-k, high=k)
    self.linear_W = nn.ModuleList([linear_W0] + linear_W)
    linear_U = nn.Linear(hidden_size, hidden_size)
    self.init_weights_uniform(linear_U, low=-k, high=k)
    self.linear_U = clones(linear_U, num_layers)

    # Embedding decoder
    self.decode = nn.Linear(hidden_size, vocab_size)

    self.dropout = nn.Dropout(1 - dp_keep_prob)
    self.activation = nn.Tanh()
    self.softmax = nn.Softmax(dim=2)

    # Weight initialization (Embedding has no bias)
    self.init_weights_uniform(self.embedding, low=-0.1, high=0.1, init_bias=False)
    self.init_weights_uniform(self.decode, low=-0.1, high=0.1, init_bias=True)

  def init_weights_uniform(self, layer, low, high, init_bias=False):
    """
    Initialize all the weights uniformly in the range [low, high]
    and all the biases to 0 (in place)
    """
    torch.nn.init.uniform_(layer.weight, a=low, b=high)
    if init_bias:
        torch.nn.init.constant_(layer.bias, 0)

  def init_hidden(self):
    """
    This is used for the first mini-batch in an epoch, only.
    """
    return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden, keep_hiddens=False):
    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)

    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the
              mini-batches in an epoch, except for the first, where the return
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details,
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """
    if keep_hiddens:
        self.hiddens = []
    h_previous_ts = hidden
    seq_logits = []
    emb = self.embedding(inputs)
    for i in range(self.seq_len):
        logits, h_previous_ts = self._forward_single_token_embedding(emb[i],
                                                                     h_previous_ts,
                                                                     keep_hiddens)
        seq_logits.append(logits)
    return torch.stack(seq_logits), h_previous_ts

  def _forward_single_token_embedding(self, embedding, h_previous_ts, keep_hiddens=False):
    """
    Forward pass for a single token embedding given the
    hidden state at the previous time step
    """
    h_next_ts = []
    h_previous_layer = self.dropout(embedding)
    for l in range(self.num_layers):
        # Recurrent layer
        a_W = self.linear_W[l](h_previous_layer)
        a_U = self.linear_U[l](h_previous_ts[l])
        h_recurrent = self.activation(a_U + a_W)
        if keep_hiddens:
            self.hiddens.append(h_recurrent)
        # Fully connected layer
        h_previous_layer = self.dropout(h_recurrent)
        # Keep the ref for next ts
        h_next_ts.append(h_recurrent)
    h_previous_ts = torch.stack(h_next_ts)
    logits = self.decode(h_previous_layer)
    return logits, h_previous_ts

  def generate(self, input, hidden, generated_seq_len, device):
    """
    Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used
                       for training (self.seq_len)
    Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """
    # Model in eval mode
    self.eval()

    samples = []
    h_previous_ts = hidden
    new_input = input
    for i in range(generated_seq_len):
        new_input = new_input.to(device)
        emb = self.embedding(new_input)
        logits, h_previous_ts = self._forward_single_token_embedding(emb, h_previous_ts)
        sample = self.softmax(logits)
        sample_index = int(np.argmax(sample.cpu().detach().numpy()))
        samples.append(sample_index)
        new_input[0, 0] = sample_index
    return samples


class GRU_cell(nn.Module):

  def __init__(self, emb_size, hidden_size):
    super(GRU_cell, self).__init__()
    self.emb_size = emb_size
    self.hidden_size = hidden_size

    self.W_x = nn.Parameter(torch.Tensor(emb_size,3*hidden_size))
    self.U_h = nn.Parameter(torch.Tensor(hidden_size,2*hidden_size))
    self.U_h_tilde = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
    self.bias_rzh = nn.Parameter(torch.Tensor(3*hidden_size))

    self.init_weights_uniform()
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()

  def init_weights_uniform(self):
    # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
    # in the range [-k, k] where k is the square root of 1/hidden_size
    k = math.sqrt(1.0/self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-k, k)

    torch.nn.init.zeros_(self.bias_rzh)

  def forward(self, inputs, hidden):
    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that
                    represent the index of the current token(s) in the vocabulary.
                        shape: (batch_size, embedding)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (batch_size, hidden_size)
    """
    batch_size = hidden.size(0)

    bias_rzh_batch = self.bias_rzh.unsqueeze(0).expand(batch_size, self.bias_rzh.size(0))
    W_x = torch.addmm(bias_rzh_batch,inputs,self.W_x)
    U_h_prev = torch.mm(hidden,self.U_h)
    W_rx, W_zx, W_hx = torch.split(W_x,self.hidden_size, dim=1)
    U_rh, U_zh = torch.split(U_h_prev,self.hidden_size, dim=1)

    r = self.sigmoid(W_rx + U_rh)
    z = self.sigmoid(W_zx + U_zh)
    h_tilde = self.tanh(W_hx + torch.mm(r * hidden,self.U_h_tilde))
    h = ((1-z) * hidden) + (z * h_tilde)

    return h

# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for
  GRU, not Vanilla RNN.
  """
  def __init__(self, emb_size, hidden_size, seq_len, batch_size,
               vocab_size, num_layers, dp_keep_prob):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(GRU, self).__init__()
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers

    # Keep hidden layers result when we want to compute the avg gradients
    self.hiddens = []

    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.decode = nn.Linear(hidden_size, vocab_size)

    self.dropout = nn.Dropout(1 - dp_keep_prob)
    self.softmax = nn.Softmax(dim=2)
    self.gen_softmax = nn.Softmax(dim=1)

    # Weight initialization (Embedding has no bias)
    self.init_weights_uniform(self.embedding, init_bias=False)
    self.init_weights_uniform(self.decode, init_bias=True)

    self.GRU_cells = nn.ModuleList([GRU_cell(emb_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

  def init_weights_uniform(self, layer, init_bias=False):
    """
    # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
    # and output biases to 0 (in place). The embeddings should not use a bias vector.
    """
    torch.nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
    if init_bias:
        torch.nn.init.constant_(layer.bias, 0)

  def init_hidden(self):
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    """
    return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden, keep_hiddens=False):
    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)

    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the
              mini-batches in an epoch, except for the first, where the return
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details,
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """
    self.hiddens = []
    h_previous_ts = hidden
    logits = []
    embeddings = self.embedding(inputs)
    for t in range(self.seq_len):
      h_next_ts = []
      input = self.dropout(embeddings[t])
      for h_index in range(self.num_layers):
        # Recurrent GRU cell
        h_recurrent = self.GRU_cells[h_index].forward(input, h_previous_ts[h_index])
        if keep_hiddens:
            self.hiddens.append(h_recurrent)
        # Fully connected layer with dropout
        h_previous_layer = self.dropout(h_recurrent)
        input = h_previous_layer # used vertically up the layers
        # Keep the ref for next ts
        h_next_ts.append(h_recurrent) # used horizontally across timesteps
      h_previous_ts = torch.stack(h_next_ts)
      logits.append(self.decode(h_previous_layer))
    return torch.stack(logits), h_next_ts

  def generate(self, input, hidden, generated_seq_len, device):  # generate next word using the GRU
    """
    Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used
                       for training (self.seq_len)
    Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """
    self.eval()
    input = input[0]
    samples = []
    h_previous_ts = hidden
    new_input = input

    for t in range(generated_seq_len):
      h_next_ts = []
      new_input = new_input.to(device)
      embedding = self.embedding(new_input)
      input = embedding
      for h_index in range(self.num_layers):
        # Recurrent GRU cell
        h_recurrent = self.GRU_cells[h_index].forward(input, h_previous_ts[h_index])
        # Fully connected layer with dropout
        h_previous_layer = self.dropout(h_recurrent)
        input = h_previous_layer  # used vertically up the layers
        # Keep the ref for next ts
        h_next_ts.append(h_recurrent)  # used horizontally across timesteps

      h_previous_ts = torch.stack(h_next_ts)

      sample = h_previous_layer
      sample = self.gen_softmax(self.decode(sample))
      sample_index = int(np.argmax(sample.cpu().detach().numpy()))
      samples.append(sample_index)
      new_input[0] = sample_index

    return samples

# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and
applying it to sequential language modelling. We use a binary "mask" to specify
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections,
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks,
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#----------------------------------------------------------------------------------

class MultiHeadedAttention(nn.Module):
  def __init__(self, n_heads, n_units, dropout=0.1):
    """
    n_heads: the number of attention heads
    n_units: the number of output units
    dropout: probability of DROPPING units
    """
    super(MultiHeadedAttention, self).__init__()
    # This sets the size of the keys, values, and queries (self.d_k) to all
    # be equal to the number of output units divided by the number of heads.
    self.d_k = n_units // n_heads

    # This requires the number of n_heads to evenly divide n_units.
    assert n_units % n_heads == 0, '{} heads is not evenly divisible by {} units'.format(n_heads, n_units)

    self.n_units = n_units
    self.n_heads = n_heads

    self.query_layer = nn.Linear(n_units, n_units)
    self.key_layer = nn.Linear(n_units, n_units)
    self.value_layer = nn.Linear(n_units, n_units)

    self.fc = nn.Linear(n_units, n_units)
    self.dropout = dropout

  def forward(self, query, key, value, mask=None):
    batch_size = query.size(0)

    # query, key, and value all have size: (batch_size, seq_len, self.n_units)
    Q = self.query_layer(query)
    K = self.key_layer(key)
    V = self.value_layer(value)

    # split each Q, K and V in self.n_heads
    Q = torch.cat(Q.split(split_size=self.d_k, dim=2), dim=0)
    K = torch.cat(K.split(split_size=self.d_k, dim=2), dim=0)
    V = torch.cat(V.split(split_size=self.d_k, dim=2), dim=0)

    A = torch.matmul(Q, K.transpose(1, 2))
    A = A / np.sqrt(self.d_k)

    if mask is not None:
      mask = mask.repeat(self.n_heads, 1, 1).float()
    A = (A * mask) - 1.e10 * (1 - mask)

    A = F.softmax(A, dim=-1)
    A = F.dropout(A, self.dropout)
    A = torch.matmul(A, V)

    # convert attention back to its input original size
    A = torch.cat(A.split(split_size=batch_size, dim=0), dim=2)
    return self.fc(A)


#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6,
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

