from __future__ import absolute_import, division, print_function, unicode_literals
from ncon import ncon

import tensorflow as tf
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from POVM import POVM
import itertools as it
import slicetf
from MPS import MPS
import sys
import os


def index(one_basis, base=4):
  return int(''.join(map(lambda x: str(int(x)), one_basis)), base)


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



def flip2_tf(S,O,K,site,mask=False):
  ## S: batch, O: gate, K: number of measurement outcomes, sites: [j,j+1]
  ## S is not one-hot form
    Ns = tf.shape(S)[0] ## batch size
    N  = tf.shape(S)[1] ## Nqubit
    flipped = tf.reshape(tf.keras.backend.repeat(S, K**2),(Ns*K**2,N)) ## repeat is to prepare K**2 outcome after O adds on, after reshape it has shape (batchsize * 16, Nqubit)
    a = tf.constant(np.array(list(it.product(range(K), repeat = 2)),dtype=np.float32)) # possible combinations of outcomes on 2 qubits ## it generates (0,0),(0,1),...,(3,3)
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


def flip1_tf(S,O,K,site,mask=False):
  ## S: batch, O: gate, K: number of measurement outcomes, sites: [j,j+1]
  ## S is not one-hot form
    Ns = tf.shape(S)[0] ## batch size
    N  = tf.shape(S)[1] ## Nqubit
    flipped = tf.reshape(tf.keras.backend.repeat(S, K),(Ns*K,N)) ## repeat is to prepare K**2 outcome after O adds on, after reshape it has shape (batchsize * 16, Nqubit)
    a = tf.constant(np.array(list(it.product(range(K), repeat = 1)),dtype=np.float32)) # possible combinations of outcomes on 2 qubits ## it generates (0,0),(0,1),...,(3,3)
    s0 = flipped[:,site[0]]
    a0 = tf.reshape(tf.tile(a[:,0],[Ns]),[-1]) ## tf.tile repeasts tensor Ns times
    ## flipped has shape (batch, Nqubit), this function is to replace flipped[batch, site0], flipped[batch, site1] with a0, a1 values
    flipped = slicetf.replace_slice_in(flipped)[:,site[0]].with_value(tf.reshape( a0 ,[K*Ns,1]))
    a = tf.tile(a,[Ns,1])
    indices_ = tf.cast(tf.concat([a,tf.reshape(s0,[tf.shape(s0)[0],1])],1),tf.int32)
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



def flip2_reverse_swift(S,O,K,site):
  ## S: batch, O: gate, K: number of measurement outcomes, sites: [j,j+1]
  ## S is not one-hot form
    Ns = tf.shape(S)[0] ## batch size
    N  = tf.shape(S)[1] ## Nqubit
    flipped = tf.cast(tf.reshape(tf.keras.backend.repeat(S, K**2),(Ns*K**2,N)),tf.float32) ## repeat is to prepare K**2 outcome after O adds on, after reshape it has shape (batchsize * 16, Nqubit)
    a = tf.constant(np.array(list(it.product(range(K), repeat = 2)),dtype=np.float32)) # possible combinations of outcomes on 2 qubits ## it generates (0,0),(0,1),...,(3,3)
    s0 = flipped[:,site[0]]
    s1 = flipped[:,site[1]]
    a0 = tf.reshape(tf.tile(a[:,0],[Ns]),[-1]) ## tf.tile repeasts tensor Ns times
    a1 = tf.reshape(tf.tile(a[:,1],[Ns]),[-1])
    ## flipped has shape (batch, Nqubit), this function is to replace flipped[batch, site0], flipped[batch, site1] with a0, a1 values
    flipped = slicetf.replace_slice_in(flipped)[:,site[0]].with_value(tf.reshape( a0 ,[K**2*Ns,1]))
    flipped = slicetf.replace_slice_in(flipped)[:,site[1]].with_value(tf.reshape( a1 ,[K**2*Ns,1]))
    a = tf.tile(a,[Ns,1])
    indices_ = tf.cast(tf.concat([tf.reshape(s0,[tf.shape(s0)[0],1]),tf.reshape(s1,[tf.shape(s1)[0],1]),a],1),tf.int32)
    ##getting the coefficients of the p-gates that accompany the flipped samples ## (Nq,Nq,Nq,Nq) shape for index
    Coef = tf.gather_nd(O,indices_) ## O has to be tensor form
    return flipped,Coef #,indices


def flip1_reverse_swift(S,O,K,site):
  ## S: batch, O: gate, K: number of measurement outcomes, sites: [j,j+1]
  ## S is not one-hot form
    Ns = tf.shape(S)[0] ## batch size
    N  = tf.shape(S)[1] ## Nqubit
    flipped = tf.cast(tf.reshape(tf.keras.backend.repeat(S, K),(Ns*K,N)),tf.float32) ## repeat is to prepare K**2 outcome after O adds on, after reshape it has shape (batchsize * 16, Nqubit)
    a = tf.constant(np.array(list(it.product(range(K), repeat = 1)),dtype=np.float32)) # possible combinations of outcomes on 2 qubits ## it generates (0,0),(0,1),...,(3,3)
    s0 = flipped[:,site[0]]
    a0 = tf.reshape(tf.tile(a[:,0],[Ns]),[-1]) ## tf.tile repeasts tensor Ns times
    ## flipped has shape (batch, Nqubit), this function is to replace flipped[batch, site0], flipped[batch, site1] with a0, a1 values
    flipped = slicetf.replace_slice_in(flipped)[:,site[0]].with_value(tf.reshape( a0 ,[K*Ns,1]))
    a = tf.tile(a,[Ns,1])
    indices_ = tf.cast(tf.concat([tf.reshape(s0,[tf.shape(s0)[0],1]),a],1),tf.int32)
    ##getting the coefficients of the p-gates that accompany the flipped samples ## (Nq,Nq,Nq,Nq) shape for index
    Coef = tf.gather_nd(O,indices_) ## O has to be tensor form
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
    a = tf.constant(np.array(list(it.product(range(K), repeat = 2)),dtype=np.uint8)) # possible combinations of outcomes on 2 qubits ## it generates (0,0),(0,1),...,(3,3)
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


def flip1_reverse_tf(S,O,K,site):
  ## S: batch, O: gate, K: number of measurement outcomes, sites: [j,j+1]
  ## S is not one-hot form
    Ns = tf.shape(S)[0] ## batch size
    N  = tf.shape(S)[1] ## Nqubit

    flipped = tf.reshape(tf.keras.backend.repeat(S, K),(Ns*K,N)) ## repeat is to prepare K**2 outcome after O adds on, after reshape it has shape (batchsize * 16, Nqubit)
    s0 = flipped[:,site[0]]
    a = tf.constant(np.array(list(it.product(range(K), repeat = 1)),dtype=np.uint8)) # possible combinations of outcomes on 2 qubits ## it generates (0,0),(0,1),...,(3,3)
    a = tf.tile(a,[Ns,1])
    indices_ = tf.cast(tf.concat([tf.reshape(s0,[tf.shape(s0)[0],1]), a],1),tf.int32)

    a = tf.transpose(a, perm=[1,0])
    flipped = tf.transpose(flipped,perm=[1,0])
    ind = tf.constant([[site[0]]])
    flipped = tf.tensor_scatter_nd_update(flipped, ind, a)
    flipped = tf.transpose(flipped,perm=[1,0])

    ##getting the coefficients of the p-gates that accompany the flipped samples ## (Nq,Nq,Nq,Nq) shape for index
    Coef = tf.gather_nd(O,indices_) ## O has to be tensor form

    return flipped,Coef #,indices


def loss_function(flip,co,gtype,batch_size,ansatz):

    target_vocab_size = ansatz.decoder.target_vocab_size

    f = tf.cond(tf.equal(gtype,1), lambda: target_vocab_size, lambda: target_vocab_size**2)
    c = tf.cast(flip, dtype=tf.uint8) # c are configurations
    lnP = ansatz(c,training=True)
    #oh =  tf.one_hot(tf.cast(flip,tf.int32),depth=target_vocab_size)
    #co = tf.cast(co,dtype = tf.float32)
    loss = -tf.reduce_sum(co * lnP) / tf.cast(batch_size,tf.float32)
    #loss2 = -tf.reduce_mean(co * lnP) * tf.cast(f,tf.float32)
    return loss #, loss2


def loss_function2(batch,ansatz):

    ## batch = (samples, lP, co_Pj_sum)
    target_vocab_size = ansatz.decoder.target_vocab_size
    batch_size = batch.shape[0]

    samples = tf.cast(batch[:,:-2], dtype=tf.uint8) # c are configurations
    batch_lP = tf.squeeze(ansatz(samples,training=True))
    co_Pj_sum = tf.squeeze(batch[:, -1])
    batch_prob = tf.squeeze(tf.stop_gradient(tf.exp(batch[:, -2])))

    loss = -tf.reduce_sum( co_Pj_sum * batch_lP / batch_prob) / tf.cast(batch_size,tf.float32)
    #loss2 = -tf.reduce_mean(co * lnP) * tf.cast(f,tf.float32)
    return loss #, loss2


def loss_function3(batch,ansatz):

    ## batch = (samples, lP, co_Pj_sum)
    target_vocab_size = ansatz.decoder.target_vocab_size
    batch_size = batch.shape[0]

    samples = tf.cast(batch[:,:-2], dtype=tf.uint8) # c are configurations
    batch_lP = tf.squeeze(ansatz(samples,training=True))
    update_Pi = tf.squeeze(batch[:, -1])
    batch_prob = tf.squeeze(tf.stop_gradient(tf.exp(batch[:, -2])))
    batch_ones = tf.ones([batch_size,],dtype=tf.float32)

    #loss = -tf.reduce_sum( (batch_ones-update_Pi/batch_prob) * batch_lP) / tf.cast(batch_size,tf.float32)
    loss = -tf.reduce_sum( update_Pi * batch_lP) / tf.cast(batch_size,tf.float32)
    return loss


def vectorize(num_sites, K, ansatz):
    l_basis = []
    for i in range(K**num_sites):
      basis_str = np.base_repr(i, base=K, padding=num_sites)[-num_sites:]
      l_basis.append(np.array(list(basis_str), dtype=int))
      #l_basis.append(basis_str)
    l_basis = np.array(l_basis)
    l_basis = tf.cast(l_basis, dtype=tf.uint8)
    lnP = ansatz(l_basis, training=False)
    return lnP


def reverse_samples(Ndataset, batch_size, gate, target_vocab_size, sites, ansatz):

    gate_factor = int(gate.ndim**2)
    ## it ensures at least one batch size samples, since Ncall can be zero
    Ncalls = Ndataset /batch_size
    samples,lP = ansatz.sample(batch_size) # get samples from the model
    lP = np.reshape(lP,[-1,1]) ## necessary for concatenate
    if gate_factor == 16:
        flip,co = flip2_reverse_tf(samples,gate,target_vocab_size,sites)
    else:
        flip,co = flip1_reverse_tf(samples,gate,target_vocab_size,sites)

    flip = tf.cast(flip, dtype=tf.uint8) # c are configurations
    Pj = tf.exp(ansatz(flip))
    co_Pj = tf.reshape(co*Pj,(batch_size, gate_factor))
    co_Pj_sum = tf.reduce_sum(co_Pj, axis=1)
    co_Pj_sum = np.reshape(co_Pj_sum,[-1,1])

    for k in range(int(Ncalls)):
        sa,llpp = ansatz.sample(batch_size)
        samples = np.vstack((samples,sa))
        llpp =np.reshape(llpp,[-1,1])
        lP =  np.vstack((lP,llpp))
        if gate_factor == 16:
            fp,coef = flip2_reverse_tf(sa,gate,target_vocab_size,sites)
        else:
            fp,coef = flip1_reverse_tf(sa,gate,target_vocab_size,sites)
        fp = tf.cast(fp, dtype=tf.uint8) # c are configurations
        pj = tf.exp(ansatz(fp))
        coef_pj = tf.reshape(coef*pj,(batch_size, gate_factor))
        coef_pj_sum = tf.reduce_sum(coef_pj, axis=1)
        coef_pj_sum = np.reshape(coef_pj_sum,[-1,1])
        co_Pj_sum =  np.vstack((co_Pj_sum,coef_pj_sum))

    samples = tf.stop_gradient(samples)
    co_Pj_sum = tf.stop_gradient(co_Pj_sum)
    lP = tf.stop_gradient(lP)

    return (samples, lP, co_Pj_sum)


# reverse samples for a sequence of gates ob TFIM ham
# TODO: change to append version and then reshape
def reverse_samples_ham(Ndataset, batch_size, Nqubit, target_vocab_size, hl, hlx, tau, ansatz):

    ## it ensures at least one batch size samples, since Ncall can be zero
    Ncalls = Ndataset /batch_size
    samples,lP = ansatz.sample(batch_size) # get samples from the model
    lP = np.reshape(lP,[-1,1]) ## necessary for concatenate
    update_Pi = tf.zeros([batch_size,], tf.float32)

    flip,co = flip2_reverse_tf(samples,hlx,target_vocab_size,site=[Nqubit-2, Nqubit-1])
    flip = tf.cast(flip, dtype=tf.uint8) # c are configurations
    Pj = tf.exp(ansatz(flip))
    co_Pj = tf.reshape(co*Pj,(batch_size, 16))
    co_Pj_sum = tf.reduce_sum(co_Pj, axis=1)
    update_Pi += co_Pj_sum
    for i in range(Nqubit-2):
        flip,co = flip2_reverse_tf(samples,hl,target_vocab_size,site=[i,i+1])
        flip = tf.cast(flip, dtype=tf.uint8) # c are configurations
        Pj = tf.exp(ansatz(flip))
        co_Pj = tf.reshape(co*Pj,(batch_size, 16))
        co_Pj_sum = tf.reduce_sum(co_Pj, axis=1)
        update_Pi += co_Pj_sum

    update_Pi = np.reshape(update_Pi,[-1,1])


    for k in range(int(Ncalls)):
        sa,llpp = ansatz.sample(batch_size)
        samples = np.vstack((samples,sa))
        llpp =np.reshape(llpp,[-1,1])
        lP =  np.vstack((lP,llpp))
        up_pi = tf.zeros([batch_size,], tf.float32)

        fp,coef = flip2_reverse_tf(sa,hlx,target_vocab_size,site=[Nqubit-2,Nqubit-1])
        fp = tf.cast(fp, dtype=tf.uint8) # c are configurations
        pj = tf.exp(ansatz(fp))
        coef_pj = tf.reshape(coef*pj,(batch_size, 16))
        coef_pj_sum = tf.reduce_sum(coef_pj, axis=1)
        up_pi += coef_pj_sum
        for i in range(Nqubit-2):
            fp,co = flip2_reverse_tf(sa,hl,target_vocab_size,site=[i,i+1])
            fp = tf.cast(fp, dtype=tf.uint8) # c are configurations
            pj = tf.exp(ansatz(fp))
            coef_pj = tf.reshape(coef*pj,(batch_size, 16))
            coef_pj_sum = tf.reduce_sum(coef_pj, axis=1)
            up_pi += coef_pj_sum

        up_pi = np.reshape(up_pi,[-1,1])
        update_Pi = np.vstack((update_Pi, up_pi))

    #update_Pi = tau * update_Pi
    update_Pi = (tf.exp(lP) - tau * update_Pi) / tf.exp(lP)
    samples = tf.stop_gradient(samples)
    update_Pi = tf.stop_gradient(update_Pi)
    lP = tf.stop_gradient(lP)

    return (samples, lP, update_Pi)


def forward_samples(Ndataset, batch_size, ansatz):
    if Ndataset != 0:
        ## it ensures at least one batch size samples, since Ncall can be zero
        Ncalls = Ndataset /batch_size
        samples,lP = ansatz.sample(batch_size) # get samples from the model
        lP = np.reshape(lP,[-1,1]) ## not necessary

        for k in range(int(Ncalls)):
            sa,llpp = ansatz.sample(batch_size)
            samples = np.vstack((samples,sa))
            llpp =np.reshape(llpp,[-1,1])
            lP =  np.vstack((lP,llpp))

    samples = tf.stop_gradient(samples)
    lP = tf.stop_gradient(lP)

    return (samples, lP)



def Fidelity_test(samp, llpp, MAX_LENGTH, target_vocab_size, mps, povm, prob, pho, ansatz):

    cFid, cFidError, KL, KLError = mps.cFidelity(tf.cast(samp,dtype=tf.int64),llpp)
    Fid, FidErrorr = mps.Fidelity(tf.cast(samp,dtype=tf.int64))
    print('cFid_mps: ', cFid, cFidError,Fid, FidErrorr)

    prob_povm = np.exp(vectorize(MAX_LENGTH, target_vocab_size, ansatz))
    pho_povm = ncon((prob_povm,povm.Ntn),([1],[1,-1,-2]))
    #Et = np.trace(pho_povm @ povm.ham)
    #print('exact E:', E, 'current E:', Et.real)
    #cFid2 = np.dot(np.sqrt(prob), np.sqrt(prob_povm))
    cFid2 = np.linalg.norm(prob-prob_povm,ord=1)
    #Fid2 = ncon((pho,pho_povm),([1,2],[2,1])) ## true for 2 qubit
    Fid2 = np.square(np.trace(sp.linalg.sqrtm(pho @ pho_povm @pho))) ## true for pure state pho
    print('cFid_ED: ', cFid2, Fid2)

    a = np.array(list(it.product(range(4),repeat = MAX_LENGTH)), dtype=np.uint8)
    l = np.sum(np.exp(ansatz(a)))
    print("prob", l)
    return cFid2, Fid2


def Fidelity_test_mps(samp, llpp, MAX_LENGTH, target_vocab_size, mps, ansatz):

    cFid, cFidError, KL, KLError = mps.cFidelity(tf.cast(samp,dtype=tf.int64),llpp)
    Fid, FidErrorr = mps.Fidelity(tf.cast(samp,dtype=tf.int64))
    print('cFid_mps: ', cFid, cFidError,Fid, FidErrorr)

    return cFid, Fid

def compute_observables(obs, site, samp):
    ndim = int(tf.size(tf.shape(obs)))
    if tf.math.equal(ndim, 1):
        indices = tf.cast(tf.reshape(samp[:,site[0]],[tf.shape(samp)[0],1]),tf.int32)
    else:
        indices = tf.cast(tf.concat([tf.reshape(samp[:,site[0]],[tf.shape(samp)[0],1]),tf.reshape(samp[:,site[1]],[tf.shape(samp)[0],1])],1),tf.int32)
    Coef = tf.gather_nd(obs, indices)

    return Coef


def compute_energy(hl_ob, hlx_ob, Nqubit, samp):
    Ns = samp.shape[0]
    Coef = compute_observables(hlx_ob, [Nqubit-2,Nqubit-1],samp)
    for i in range(Nqubit-2):
        Coef += compute_observables(hl_ob, [i,i+1], samp)
    Coef2 = tf.math.square(Coef, Coef)
    Coef_mean = tf.reduce_mean(Coef)
    Coef2_mean = tf.reduce_mean(Coef2)
    Err = tf.math.sqrt( (Coef2_mean- Coef_mean**2)/Ns)
    return Coef_mean, Err


def compute_energy_mpo(Hp, S):
    E = 0.0;
    E2 = 0.0;
    N = len(Hp)
    Ns = S.shape[0]
    for i in range(Ns):

        # contracting the entire TN for each sample S[i,:]
        eT = Hp[0][S[i,0],:];

        for j in range(1,N-1):
            eT = ncon((eT,Hp[j][:,S[i,j],:]),([1],[1,-1]));

        j = N-1
        eT = ncon((eT,Hp[j][:,S[i,j]]),([1],[1]));
        #print i, eT
        E = E + eT;
        E2 = E2 + eT**2;
        Fest=E/float(i+1);
        F2est=E2/float(i+1);
        Error = np.sqrt( np.abs( F2est-Fest**2 )/float(i+1));
        #print i,np.real(Fest),Error
        #disp([i,i/Ns, real(Fest), real(Error)])
        #fflush(stdout);

    E2 = E2/float(Ns);
    E = np.abs(E/float(Ns));
    Error = np.sqrt( np.abs( E2-E**2 )/float(Ns));
    return np.real(E), Error


@tf.function
def train_step(flip,co,gtype,batch_size,optimizer,ansatz):

    with tf.GradientTape() as tape:
        loss = loss_function(flip,co,gtype,batch_size,ansatz)

    gradients = tape.gradient(loss, ansatz.trainable_variables)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))
    return loss

@tf.function
def train_step2(batch,optimizer,ansatz):

    with tf.GradientTape() as tape:
        loss = loss_function2(batch,ansatz)

    gradients = tape.gradient(loss, ansatz.trainable_variables)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))
    return loss


@tf.function
def train_step3(batch,optimizer,ansatz):

    with tf.GradientTape() as tape:
        loss = loss_function3(batch,ansatz)

    gradients = tape.gradient(loss, ansatz.trainable_variables)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))
    return loss


