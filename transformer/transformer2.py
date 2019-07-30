from __future__ import absolute_import, division, print_function, unicode_literals
from ncon import ncon

import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from POVM import POVM
import itertools as it
import slicetf
from MPS import MPS
import sys
import os



def get_angles(pos, i, d_model):
  #TODO: why i//2
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  sines = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  cosines = np.cos(angle_rads[:, 1::2])

  pos_encoding = np.concatenate([sines, cosines], axis=-1)

  pos_encoding = pos_encoding[np.newaxis, ...] #(1, 2, #d_model)

  return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, -1), tf.float32)

  # add extra dimensions so that we can add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  ## return strictly upper triangular matrix
  return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)

  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.

  if mask is not None:
    #print(scaled_attention_logits.shape,mask.shape)
    scaled_attention_logits += (mask * -1e9) ## penalize upper triangular part, which corresponds to the later config

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  #print(attention_weights.shape,attention_weights)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  #print("output word 1",output[:,:,0,:])
  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    ## first map input (batch, d_model) to (batch, dk)
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)
    #print(self.depth,"depth")

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    #print(q.shape,k.shape,v.shape,"shapes??")

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    #print("scaled_attention w1",scaled_attention[:,:,0,:])
    #print("scaled_attention w2",scaled_attention[:,:,1,:])
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    #print("outputMHA1",output.shape, "w1", output[:,0,:],"w2", output[:,1,:])

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    #self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    #self.layernorm1 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6) ## uncomment
    #self.layernorm2 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6)
    #self.layernorm3 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6) ## uncomment

    self.dropout1 = tf.keras.layers.Dropout(rate)
    #self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)


  def call(self, x, training,
           look_ahead_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    #print(attn1.shape,attn1[:,0,:],"attn1 first word")
    #print(attn1.shape,attn1[:,1,:],"attn1 second word")
    attn1 = self.dropout1(attn1, training=training)
    #print(attn1.shape,attn1[:,0,:],"attn1 first word dropout")
    #print(attn1.shape,attn1[:,1,:],"attn1 second word dropout")
    #out1 = self.layernorm1(attn1 + x)
    out1 = attn1 + x
    #s = tf.shape(out1)
    #out1 = tf.reshape(out1,[s[0]*s[1],s[2]])
    out1 = self.layernorm1(out1) ## uncomment
    #out1 = tf.reshape(out1,[s[0],s[1],s[2]])
    #print("out1 w1",out1[:,0,:],"out1 w2",out1[:,1,:])
    #attn2, attn_weights_block2 = self.mha2(
    #    enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    #attn2 = self.dropout2(attn2, training=training)
    #out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
    #print("ffn_output word 1",ffn_output[:,0,:])
    #print("ffn_output word 2",ffn_output[:,1,:])
    ffn_output = self.dropout3(ffn_output, training=training)
    #out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)
    out3 = ffn_output + out1
    #s = tf.shape(out3)
    #out3 = tf.reshape(out3,[s[0]*s[1],s[2]])
    out3 = self.layernorm1(out3) ## uncomment
    #out3 = tf.reshape(out3,[s[0],s[1],s[2]])
    #out3 = self.layernorm3(ffn_output)

    return out3, attn_weights_block1


class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, max_length, num_heads, dff, target_vocab_size,
               rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.max_length = max_length
    self.target_vocab_size = target_vocab_size

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(max_length, self.d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training,
           look_ahead_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1 = self.dec_layers[i](x,  training,
                                             look_ahead_mask)

      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      #attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, max_length, num_heads, dff, input_vocab_size,
               target_vocab_size, rate=0.1,bias='zeros'):
    super(Transformer, self).__init__()

    #self.encoder = Encoder(num_layers, d_model, num_heads, dff,
    #                       input_vocab_size, rate)

    self.decoder = Decoder(num_layers, d_model, max_length, num_heads, dff,
                           target_vocab_size, rate)

    #self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    bi = tf.constant_initializer(bias)
    self.final_layer = tf.keras.layers.Dense(target_vocab_size,kernel_initializer='zeros',bias_initializer=bi)

  def call(self, tar, training,
           look_ahead_mask):

    #enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, training, look_ahead_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights


#this is from the original Transformer paper
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask



### Utilz (this part could be separted later)

def sample(ansatz,Nsamples=1000):

  MAX_LENGTH = ansatz.decoder.max_length
  d_model = ansatz.decoder.d_model
  target_vocab_size = ansatz.decoder.target_vocab_size

  encoder_input = tf.ones([Nsamples,MAX_LENGTH,d_model]) #(inp should be? bsize, sequence_length, d_model)
  output = tf.zeros([Nsamples,1], dtype=tf.uint8)
  logP = tf.zeros([Nsamples,1])

  for i in range(MAX_LENGTH):
    #print("conditional sampling at site", i)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = ansatz(output, # self, tar, training,look_ahead_mask
                                                 False,
                                                 None)
    #if i == MAX_LENGTH-1:
    #    logP = tf.math.log(tf.nn.softmax(predictions,axis=2)+1e-10) # to compute the logP of the sampled config after sampling

    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size) # select  # select the last word from the     seq_len dimension

    predictions = tf.reshape(predictions,[-1,target_vocab_size])  # (batch_size, 1, vocab_size)

    predicted_id = tf.random.categorical(predictions,1, dtype=tf.int32) # sample the conditional distribution

    lp = tf.math.log(tf.nn.softmax(predictions,axis=1)+1e-10)

    ohot = tf.reshape(tf.one_hot(predicted_id,target_vocab_size),[-1,target_vocab_size])

    preclp = tf.reshape(tf.reduce_sum(ohot*lp,[1]),[-1,1])

    logP = logP + preclp

    output = tf.concat([output, tf.cast(predicted_id,dtype=tf.uint8)], axis=1)

  output = tf.slice(output, [0, 1], [-1, -1]) # Cut the input of the initial call (zeros)

  #oh = tf.one_hot(tf.cast(output,dtype=tf.int32),target_vocab_size) # one hot vector of the sample
  #logP = tf.reduce_sum(logP*oh,[1,2]) # the log probability of the configuration
  #print(logP)

  return output,logP #, attention_weights

def logP(config,ansatz, training=False):

  MAX_LENGTH = ansatz.decoder.max_length
  d_model = ansatz.decoder.d_model
  target_vocab_size = ansatz.decoder.target_vocab_size

  Nsamples =  tf.shape(config)[0]
  encoder_input = tf.ones([Nsamples,MAX_LENGTH,d_model]) #(inp should be? bsize, sequence_length, d_model)
  init  = tf.zeros([Nsamples,1])
  output = tf.concat([init,tf.cast(config,dtype=tf.float32)],axis=1)
  output = output[:,0:MAX_LENGTH]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

  # predictions.shape == (batch_size, seq_len, vocab_size) # self, tar, training,look_ahead_mask
  predictions, attention_weights = ansatz(output,training,combined_mask)

  # predictions (Nsamples/b_size, MAX_LENGTH,vocab_size)
  # print(predictions)
  logP = tf.math.log(tf.nn.softmax(predictions,axis=2)+1e-10)
  #print(logP[:,0,:],logP.shape,"config+0",output)
  oh = tf.one_hot(config,target_vocab_size)
  logP = tf.reduce_sum(logP*oh,[1,2])

  return logP #, attention_weights


def flip2_tf(S,O,K,site,mask=False):
  ## S: batch, O: gate, K: number of measurement outcomes, sites: [j,j+1]
  ## S is not one-hot form
    Ns = tf.shape(S)[0] ## batch size
    N  = tf.shape(S)[1] ## Nqubit
    flipped = tf.reshape(tf.keras.backend.repeat(S, K**2),(Ns*K**2,N)) ## repeat is to prepare K**2 outcome after O adds on, after reshape it has shape (batchsize * 16, Nqubit)
    a = tf.constant(np.array(list(it.product(range(K), repeat = 2)),dtype=np.uint8)) # possible combinations of outcomes on 2 qubits ## it generates (0,0),(0,1),...,(3,3)
    s0 = flipped[:,site[0]]
    s1 = flipped[:,site[1]]
    a0 = tf.reshape(tf.tile(a[:,0],[Ns]),[-1]) ## tf.tile repeasts tensor Ns times
    a1 = tf.reshape(tf.tile(a[:,1],[Ns]),[-1])
    ## flipped has shape (batch, Nqubit), this function is to replace flipped[batch, site0], flipped[batch, site1] with a0, a1 values
    flipped = slicetf.replace_slice_in(flipped)[:,site[0]].with_value(tf.reshape( a0 ,[K**2*Ns,1]))
    flipped = slicetf.replace_slice_in(flipped)[:,site[1]].with_value(tf.reshape( a1 ,[K**2*Ns,1]))
    a = tf.tile(a,[Ns,1])
    indices_ = tf.cast(tf.concat([a,tf.reshape(s0,[tf.shape(s0)[0],1]),tf.reshape(s1,[tf.shape(s1)[0],1])],1),tf.int32)
    ##getting the coefficients of the p-gates that accompany the flipped samples ## (Nq,Nq,Nq,Nq) shape for index
    Coef = tf.gather_nd(O,indices_) ## O has to be tensor form
    ## transform samples to one hot vector
    #flipped = tf.one_hot(tf.cast(flipped,tf.int32),depth=K)
    #flipped = tf.reshape(flipped,[tf.shape(flipped)[0],tf.shape(flipped)[1]*tf.shape(flipped)[2]])
    if mask:
      ind = Coef > 1e-13
      Coef = tf.gather(Coef, tf.where(ind))
      flipped = tf.gather_nd(flipped, tf.where(ind))
    return flipped,Coef #,indices


def flip2_tf2(S,O,K,site,mask=False):
  ## S: batch, O: gate, K: number of measurement outcomes, sites: [j,j+1]
  ## S is not one-hot form
    Ns = tf.shape(S)[0] ## batch size
    N  = tf.shape(S)[1] ## Nqubit

    flipped = tf.reshape(tf.keras.backend.repeat(S, K**2),(Ns*K**2,N)) ## repeat is to prepare K**2 outcome after O adds on, after reshape it has shape (batchsize * 16, Nqubit)
    s0 = flipped[:,site[0]]
    s1 = flipped[:,site[1]]
    a = tf.constant(np.array(list(it.product(range(K), repeat = 2)),dtype=np.uint8)) # possible combinations of outcomes on 2 qubits ## it generates (0,0),(0,1),...,(3,3)
    a = tf.tile(a,[Ns,1])
    indices_ = tf.cast(tf.concat([a,tf.reshape(s0,[tf.shape(s0)[0],1]),tf.reshape(s1,[tf.shape(s1)[0],1])],1),tf.int32)

    a = tf.transpose(a, perm=[1,0])
    flipped = tf.transpose(flipped,perm=[1,0])
    ind = tf.constant([[site[0]], [site[1]]])
    flipped = tf.tensor_scatter_nd_update(flipped, ind, a)
    flipped = tf.transpose(flipped,perm=[1,0])

    ##getting the coefficients of the p-gates that accompany the flipped samples ## (Nq,Nq,Nq,Nq) shape for index
    Coef = tf.gather_nd(O,indices_) ## O has to be tensor form
    if mask:
      ind = Coef > 1e-13
      Coef = tf.gather(Coef, tf.where(ind))
      flipped = tf.gather_nd(flipped, tf.where(ind))

    return flipped,Coef #,indices


def flip2_reverse_tf(S,O,K,site):
  ## S: batch, O: gate, K: number of measurement outcomes, sites: [j,j+1]
  ## S is not one-hot form
    Ns = tf.shape(S)[0] ## batch size
    N  = tf.shape(S)[1] ## Nqubit

    flipped = tf.reshape(tf.keras.backend.repeat(S, K**2),(Ns*K**2,N)) ## repeat is to prepare K**2 outcome after O adds on, after reshape it has shape (batchsize * 16, Nqubit)
    s0 = flipped[:,site[0]]
    s1 = flipped[:,site[1]]
    a = tf.constant(np.array(list(it.product(range(K), repeat = 2)),dtype=np.float32)) # possible combinations of outcomes on 2 qubits ## it generates (0,0),(0,1),...,(3,3)
    a = tf.tile(a,[Ns,1])
    indices_ = tf.cast(tf.concat([tf.reshape(s0,[tf.shape(s0)[0],1]),tf.reshape(s1,[tf.shape(s1)[0],1]), a],1),tf.int32)

    a = tf.transpose(a, perm=[1,0])
    flipped = tf.transpose(flipped,perm=[1,0])
    ind = tf.constant([[site[0]], [site[1]]])
    flipped = tf.tensor_scatter_nd_update(flipped, ind, a)
    flipped = tf.transpose(flipped,perm=[1,0])

    ##getting the coefficients of the p-gates that accompany the flipped samples ## (Nq,Nq,Nq,Nq) shape for index
    Coef = tf.gather_nd(O,indices_) ## O has to be tensor form

    return flipped,Coef #,indices


def loss_function(flip,co,gtype,batch_size,ansatz):

    target_vocab_size = ansatz.decoder.target_vocab_size

    f = tf.cond(tf.equal(gtype,1), lambda: target_vocab_size, lambda: target_vocab_size**2)
    c = tf.cast(flip, dtype=tf.uint8) # c are configurations
    lnP = logP(c,ansatz, training=True)
    #oh =  tf.one_hot(tf.cast(flip,tf.int32),depth=target_vocab_size)
    #co = tf.cast(co,dtype = tf.float32)
    loss = -tf.reduce_sum(co * lnP) / tf.cast(batch_size,tf.float32)
    #loss2 = -tf.reduce_mean(co * lnP) * tf.cast(f,tf.float32)
    return loss #, loss2


def loss_function2(flip,co,gtype,batch,ansatz):

    target_vocab_size = ansatz.decoder.target_vocab_size
    batch_size = batch.shape[0]

    f = tf.cond(tf.equal(gtype,1), lambda: target_vocab_size, lambda: target_vocab_size**2)
    samples = tf.cast(batch[:,:2], dtype=tf.uint8) # c are configurations
    batch_lP = logP(samples,ansatz, training=True)
    c = tf.cast(flip, dtype=tf.uint8) # c are configurations
    Pj = tf.exp(logP(c,ansatz))
    co_Pj = tf.reshape(co*Pj,(batch_size, f))
    co_Pj_sum = tf.reduce_sum(co_Pj, axis=1)
    co_Pj_sum = tf.stop_gradient(co_Pj_sum)
    batch_prob = tf.stop_gradient(tf.exp(batch[:, 2]))

    loss = -tf.reduce_sum( co_Pj_sum * batch_lP / batch_prob) / tf.cast(batch_size,tf.float32)
    #loss2 = -tf.reduce_mean(co * lnP) * tf.cast(f,tf.float32)
    return loss #, loss2


def vectorize(num_sites, K, ansatz):
    l_basis = []
    for i in range(K**num_sites):
      basis_str = np.base_repr(i, base=K, padding=num_sites)[-num_sites:]
      l_basis.append(np.array(list(basis_str), dtype=int))
      #l_basis.append(basis_str)
    l_basis = np.array(l_basis)
    l_basis = tf.cast(l_basis, dtype=tf.uint8)
    lnP = logP(l_basis, ansatz, training=False)
    return lnP


