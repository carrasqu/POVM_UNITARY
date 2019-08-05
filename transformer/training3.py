from __future__ import absolute_import, division, print_function, unicode_literals
from ncon import ncon

import tensorflow as tf
import numpy as np
import scipy as sp
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
num_heads = 2 # 8

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

povm_='Tetra_pos'
initial_state='0'
povm = POVM(POVM=povm_, Number_qubits=MAX_LENGTH, initial_state=initial_state,Jz=1.0,hx=1.0,eps=10./float(T))
mps = MPS(POVM=povm_,Number_qubits=MAX_LENGTH,MPS="GHZ")
bias = povm.getinitialbias(initial_state)

# define target state
povm.construct_psi()
povm.construct_Nframes()
povm.construct_ham()
psi, E = povm.ham_eigh()
## GHZ state
#psi = np.zeros(2**MAX_LENGTH)
#psi[0] = 1.
#psi[-1] = 1.
#psi = psi/ np.sqrt(2)
psi_t = povm.psi
psi_t = psi_t / np.linalg.norm(psi_t)
pho = np.outer(psi, np.conjugate(psi))
pho_t = np.outer(psi_t, np.conjugate(psi_t))
prob = ncon((pho,povm.Mn),([1,2],[-1,2,1])).real
prob_t = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real


# define ansatz
ansatz = Transformer(num_layers, d_model, MAX_LENGTH, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate,bias)

if LOAD==1:
  print('load model')
  ansatz.load_weights('./models/transformer2')


# energy test
Et = np.conjugate(psi_t.transpose()) @ povm.ham @ psi_t
samp,_ = ansatz.sample(10000) # get samples from the mode
E_samp,E2_samp = compute_energy(povm.hl_ob,povm.x_ob,MAX_LENGTH,samp)
E_mpo,E2_mpo = compute_energy_mpo(povm.Hp, samp)
print('correct E', Et)
print('samples E', E_samp, E2_samp)
print('mpo E', E_mpo, E2_mpo)
assert False, 'stop'



# starting fidelity
print('starting fidelity')
prob_povm = np.exp(vectorize(MAX_LENGTH, target_vocab_size, ansatz))
pho_povm = ncon((prob_povm,povm.Ntn),([1],[1,-1,-2]))
cFid = np.dot(np.sqrt(prob), np.sqrt(prob_povm))
Fid = ncon((pho,pho_povm),([1,2],[2,1]))
cFid_t = np.dot(np.sqrt(prob_t), np.sqrt(prob_povm))
Fid_t = ncon((pho_t,pho_povm),([1,2],[2,1]))
print('target fidelity:', cFid, Fid)
print('initial fidelity:', cFid_t, Fid_t)

plt.figure(1)
plt.bar(np.arange(4**MAX_LENGTH),prob)
plt.figure(2)
plt.bar(np.arange(4**MAX_LENGTH),prob_povm)
#assert False, 'stop'



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

  #SITE = [[0]]
  #GATE = [povm.H]
  #P_GATE = [povm.P_gate(povm.H)]
  #for i in range(MAX_LENGTH-1):
  #    SITE.append([i,i+1])
  #    GATE.append(povm.cnot)
  #    P_GATE.append(povm.P_gate(povm.cnot))
  SITE = [[0,1]]
  GATE = [povm.mat2]
  P_GATE = [povm.P_gate(povm.mat2)]

  for i in range(len(SITE)):

    sites=SITE[i]
    gate = P_GATE[i]
    gtype = int(gate.ndim/2)
    kron_gate = povm.kron_gate(GATE[i], sites[0], MAX_LENGTH)
    psi_t = psi_t @ kron_gate
    psi_t = psi_t / np.linalg.norm(psi_t)
    pho_t = np.outer(psi_t, np.conjugate(psi_t))
    prob_t = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real

    samples_lP_co = reverse_samples(Ndataset, batch_size, gate, target_vocab_size, sites, ansatz)
    #samples_lP_co = forward_samples(Ndataset, batch_size, ansatz)
    ept = tf.random.shuffle(np.concatenate(samples_lP_co,axis=1))

    nsteps = int(samples_lP_co[0].shape[0] / batch_size) ## samples.shape[0]=Ndataset + batchsize
    bcount = 0
    counter=0

    for epoch in range(EPOCHS):

            for idx in range(nsteps):
                print("gate", i, "site", sites[0], "epoch", epoch,"out of ", EPOCHS, 'nsteps', idx)
                if bcount*batch_size + batch_size>=Ndataset:
                    bcount=0
                    #ept = tf.random.shuffle(np.concatenate((samples,lP, co_Pj_sum),axis=1))
                    ept = tf.random.shuffle(np.concatenate((samples_lP_co),axis=1))
                batch = ept[ bcount*batch_size: bcount*batch_size+batch_size,:]
                bcount=bcount+1

                ## forward sampling training
                #if gtype == 1:
                #    ## batch[:,:-1] = samples in batch
                #    flip,co = flip1_tf(batch[:,:-1],gate,target_vocab_size,sites)
                #else:
                #    flip,co = flip2_tf(batch[:,:-1],gate,target_vocab_size,sites)
                #loss = train_step(flip,co,gtype,batch_size,ansatz)
                ## reverse sampling training
                loss = train_step2(batch,ansatz)

            print('loss:', loss)
            samp,llpp = ansatz.sample(10) # get samples from the mode
            #cFid, Fid = Fidelity_test(samp, llpp, MAX_LENGTH, target_vocab_size, mps, povm, prob, pho, ansatz)
            cFid_t, Fid_t = Fidelity_test(samp, llpp, MAX_LENGTH, target_vocab_size, mps, povm, prob_t, pho_t, ansatz)
            #cFid_t, Fid_t = Fidelity_test_mps(samp, llpp, MAX_LENGTH, target_vocab_size, mps, ansatz)
            #Fidelity.append(np.array([cFid, Fid]))
            Fidelity.append(np.array([cFid_t, Fid_t]))

    ansatz.save_weights('./models/transformer2', save_format='tf')


Fidelity = np.array(Fidelity)
np.savetxt('./data/Fidelity.txt',Fidelity)


## final fidelity
cFid, Fid = Fidelity_test(samp, llpp, MAX_LENGTH, target_vocab_size, mps, povm, prob, pho, ansatz)
#cFid, Fid = Fidelity_test_mps(samp, llpp, MAX_LENGTH, target_vocab_size, mps, ansatz)
