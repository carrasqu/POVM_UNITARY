from __future__ import absolute_import, division, print_function, unicode_literals
from ncon import ncon

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from POVM import POVM
import itertools as it
from MPS import MPS
import sys
import os
import time
import timeit
from transformer3 import *


if not os.path.exists("samples"):
    os.makedirs("samples")
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("data"):
    os.makedirs("data")

### Parameters setting
num_layers = 2 #4
d_model = 128 #128
dff = 128 # 512
num_heads = 4 # 8

target_vocab_size = 4 # number of measurement outcomes
input_vocab_size = target_vocab_size
dropout_rate = 0.0

MAX_LENGTH = int(sys.argv[1]) # number of qubits
T = int(sys.argv[2])
EPOCHS = int(sys.argv[3]) ##20
batch_size = int(sys.argv[4]) ##1000
Ndataset = int(sys.argv[5]) ## 100000
LOAD = int(sys.argv[6])
j_init = 0

povm_='Tetra'
initial_state='0'
povm = POVM(POVM=povm_, Number_qubits=MAX_LENGTH, initial_state=initial_state,Jz=1.0,hx=1.0,eps=10./float(T))
mps = MPS(POVM=povm_,Number_qubits=MAX_LENGTH,MPS="GHZ")
bias = povm.getinitialbias(initial_state)

# define target state
povm.construct_psi()
povm.construct_ham()
psi, E = povm.ham_eigh()
#psi = 1/2.0 * np.array([1.,1.,1.,-1], dtype=complex)
## GHZ state
psi = np.zeros(2**MAX_LENGTH)
psi[0] = 1.
psi[-1] = 1.
psi = psi/ np.sqrt(2**MAX_LENGTH)
pho = np.outer(psi, np.conjugate(psi))
prob = ncon((pho,povm.Mn),([1,2],[-1,2,1])).real


# define ansatz
ansatz = Transformer(num_layers, d_model, MAX_LENGTH, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate,bias)

if LOAD==1:
  print('load model')
  ansatz.load_weights('./models/transformer2')

# starting fidelity
print('starting fidelity')
prob_povm = np.exp(vectorize(MAX_LENGTH, target_vocab_size, ansatz))
pho_povm = ncon((prob_povm,povm.Ntn),([1],[1,-1,-2]))
cFid2 = np.dot(np.sqrt(prob), np.sqrt(prob_povm))
Fid2 = ncon((pho,pho_povm),([1,2],[2,1]))
print(cFid2, Fid2)

#plt.figure(1)
#plt.bar(np.arange(4**MAX_LENGTH),prob)
#plt.figure(2)
#plt.bar(np.arange(4**MAX_LENGTH),prob_povm)



# define training setting
learning_rate = CustomSchedule(d_model)
#optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                     epsilon=1e-9)
optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9) ## lr=1e-4

@tf.function
def train_step(flip,co,gtype,batch_size,ansatz):

    with tf.GradientTape() as tape:
        loss = loss_function(flip,co,gtype,batch_size,ansatz)

    gradients = tape.gradient(loss, ansatz.trainable_variables)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))

    return loss


@tf.function
def train_step2(batch,ansatz):

    with tf.GradientTape() as tape:
        loss = loss_function2(batch,ansatz)

    gradients = tape.gradient(loss, ansatz.trainable_variables)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))

    return loss

Fidelity=[]
for t in range(T):

  sites=[0]
  gate = povm.p_single_qubit[0] # 1 is CZ gate, 0 is CNOT
  samples, lP, co_Pj_sum = prepare_samples(Ndataset, batch_size, gate, target_vocab_size, sites, ansatz)
  ept = tf.random.shuffle(np.concatenate((samples,lP, co_Pj_sum),axis=1))

  nsteps = int(samples.shape[0] / batch_size) ## samples.shape[0]=Ndataset + batchsize
  bcount = 0
  counter=0

  for epoch in range(EPOCHS):

          for idx in range(nsteps):
              print("epoch", epoch,"out of ", EPOCHS,"site", sites[0], 'nsteps', idx)
              if bcount*batch_size + batch_size>=Ndataset:
                  bcount=0
                  ept = tf.random.shuffle(np.concatenate((samples,lP, co_Pj_sum),axis=1)) ## first two columns are configurations and the last column is the corresponding probability
              batch = ept[ bcount*batch_size: bcount*batch_size+batch_size,:]
              bcount=bcount+1

              loss = train_step2(batch,ansatz)
              print('loss:', loss)

              samp,llpp = ansatz.sample(1000) # get samples from the mode
              cFid, Fid = Fidelity_test(samp, llpp, MAX_LENGTH, target_vocab_size, mps, povm, prob, pho, ansatz)
              Fidelity.append(np.array([cFid, Fid]))

  ansatz.save_weights('./models/transformer2', save_format='tf')


  ## site [j,j+1] > epoch > nsteps = Ndataset/batchsize
  for j in range(j_init,MAX_LENGTH-1):

      sites=[j,j+1] # on which sites to apply the gate
      gate = povm.p_two_qubit[0] # 1 is CZ gate, 0 is CNOT

      samples, lP, co_Pj_sum = prepare_samples(Ndataset, batch_size, gate, target_vocab_size, sites, ansatz)
      ept = tf.random.shuffle(np.concatenate((samples,lP, co_Pj_sum),axis=1))

      nsteps = int(samples.shape[0] / batch_size) ## samples.shape[0]=Ndataset + batchsize
      bcount = 0
      counter=0

      for epoch in range(EPOCHS):
              for idx in range(nsteps):
                  print("epoch", epoch,"out of ", EPOCHS,"two site", j, 'nsteps', idx)
                  if bcount*batch_size + batch_size>=Ndataset:
                      bcount=0
                      ept = tf.random.shuffle(np.concatenate((samples,lP, co_Pj_sum),axis=1)) ## first two columns are configurations and the last column is the corresponding probability
                  batch = ept[ bcount*batch_size: bcount*batch_size+batch_size,:]
                  bcount=bcount+1

                  loss = train_step2(batch,ansatz)
                  print('loss:', loss)

                  samp,llpp = ansatz.sample(1000) # get samples from the mode
                  cFid, Fid = Fidelity_test(samp, llpp, MAX_LENGTH, target_vocab_size, mps, povm, prob, pho, ansatz)
                  Fidelity.append(np.array([cFid, Fid]))

      ansatz.save_weights('./models/transformer2', save_format='tf')


Fidelity = np.array(Fidelity)
np.savetxt('./data/Fidelity.txt',Fidelity)



#samples,lnP = sample(Ndataset)
if Ndataset != 0:
    Ncalls = Ndataset /batch_size
    samples,lP = ansatz.sample(batch_size) # get samples from the model
    lP = np.reshape(lP,[-1,1])

    for k in range(int(Ncalls)):
        sa,llpp = ansatz.sample(batch_size)
        samples = np.vstack((samples,sa))
        llpp =np.reshape(llpp,[-1,1])
        lP =  np.vstack((lP,llpp))


# classical fidelity from mps
#cFid, cFidError, KL, KLError = mps.cFidelity(tf.cast(samples,dtype=tf.int64),lP)
#Fid, FidErrorr = mps.Fidelity(tf.cast(samples,dtype=tf.int64))
#stabilizers,sError = mps.stabilizers_samples(tf.cast(samples,dtype=tf.int64))
#print(cFid, cFidError,Fid, FidErrorr)
#print(stabilizers,sError,np.mean(stabilizers),np.mean(sError))


prob = ncon((pho,povm.Mn),([1,2],[-1,2,1]))
prob_povm = np.exp(vectorize(MAX_LENGTH, target_vocab_size, ansatz))
pho_povm = ncon((prob_povm,povm.Ntn),([1],[1,-1,-2]))
cFid2 = np.dot(np.sqrt(prob), np.sqrt(prob_povm))
Fid2 = ncon((pho,pho_povm),([1,2],[2,1]))
print(cFid2, Fid2)
