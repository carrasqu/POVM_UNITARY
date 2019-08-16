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
from utils import *


if not os.path.exists("samples"):
    os.makedirs("samples")
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("data"):
    os.makedirs("data")

### Parameters setting
num_layers = 1 #4  #1
d_model = 16 #128 #16
dff = 16 # 512     #16
num_heads = 1 # 8 #1

target_vocab_size = 4 # number of measurement outcomes
input_vocab_size = target_vocab_size
dropout_rate = 0.0

Nqubit = int(sys.argv[1]) # number of qubits
T = int(sys.argv[2])
EPOCHS = int(sys.argv[3]) ##20
batch_size = int(sys.argv[4]) ##1000
Ndataset = int(sys.argv[5]) ## 100000
LOAD = int(sys.argv[6])
j_init = 0

povm_='Tetra_pos'
initial_state='0'
tau = 10/float(T)
povm = POVM(POVM=povm_, Number_qubits=Nqubit, initial_state=initial_state,Jz=1.0,hx=1.0,eps=tau)
mps = MPS(POVM=povm_,Number_qubits=Nqubit,MPS="GHZ")
bias = povm.getinitialbias(initial_state)

# define ansatz
ansatz = Transformer(num_layers, d_model, Nqubit, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate,bias)
if LOAD==1:
  print('load model')
  ansatz.load_weights('./models/transformer2')


# define target state
povm.construct_psi()
#povm.construct_Nframes()
povm.construct_ham()
psi, E = povm.ham_eigh()
## GHZ state
#psi = np.zeros(2**Nqubit)
#psi[0] = 1.
#psi[-1] = 1.
#psi = psi/ np.sqrt(2)

# construct density matrix
pho = np.outer(psi, np.conjugate(psi))
prob = ncon((pho,povm.Mn),([1,2],[-1,2,1])).real
psi_t = povm.psi.copy()
psi_t = psi_t / np.linalg.norm(psi_t)
pho_t = np.outer(psi_t, np.conjugate(psi_t))
prob_t = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real
pho_t0 = pho_t.copy()
prob_t0 = prob_t.copy()

# starting fidelity
print('starting fidelity')
prob_povm = np.exp(vectorize(Nqubit, target_vocab_size, ansatz))
pho_povm = ncon((prob_povm,povm.Ntn),([1],[1,-1,-2]))
cFid = np.dot(np.sqrt(prob), np.sqrt(prob_povm))
Fid = ncon((pho,pho_povm),([1,2],[2,1]))
cFid_t = np.dot(np.sqrt(prob_t), np.sqrt(prob_povm))
Fid_t = ncon((pho_t,pho_povm),([1,2],[2,1]))
print('target fidelity:', cFid, Fid)
print('initial fidelity:', cFid_t, Fid_t)

#plt.figure(1)
#plt.bar(np.arange(4**Nqubit),prob_t)
#plt.figure(2)
#plt.bar(np.arange(4**Nqubit),prob_povm)


# starting energy
#samp,_ = ansatz.sample(10000) # get samples from the mode
#E_samp,E2_samp = compute_energy(povm.hl_ob,povm.hlx_ob,Nqubit,samp)
#print('Beginning energy:', E_samp, E2_samp)
print('Exact beginning energy:', np.trace(povm.ham @ pho_povm))


# define training setting
#learning_rate = CustomSchedule(d_model)
#optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9) ## lr=1e-4


Fidelity=[]
for t in range(T):

    #psi_t = (np.eye(2**Nqubit,2**Nqubit)-tau*povm.ham) @ psi_t
    #psi_t = psi_t / np.linalg.norm(psi_t)
    #pho_t = np.outer(psi_t, np.conjugate(psi_t))
    pho_t = pho_t - tau *( povm.ham @ pho_t + pho_t @ povm.ham)
    pho_t = pho_t / np.trace(pho_t)
    prob_t = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real
    #plt.figure(3)
    #plt.bar(np.arange(4**Nqubit),prob_t)

    print('prob diff at time '+str(t), np.linalg.norm(prob_t-prob_povm,ord=1))
    samples_lP_co = reverse_samples_ham2(Ndataset, batch_size, Nqubit, target_vocab_size, povm.hl_com, povm.hlx_com, tau, ansatz)


    #sa = samples_lP_co[0]
    #lp = samples_lP_co[1]
    #up_pi = samples_lP_co[2]
    #up_pi2 = up_pi * tf.exp(lp)
    ##freq = (tf.ones([sa.shape[0],]) - up_pi / tf.exp(lp)) / tf.cast(sa.shape[0],tf.float32)
    #freq = up_pi / tf.cast(sa.shape[0],tf.float32)
    #config = tf.map_fn(lambda x: index(x), sa)
    #plt.figure(4)
    #plt.hist(config, bins=4**Nqubit, density=True)
    #u, ind = np.unique(config, return_index=True)
    #up_pi2 = up_pi2.numpy()[ind].reshape(-1)
    #plt.figure(5)
    #plt.bar(np.arange(len(u)),up_pi2)

    #hist = np.zeros(len(u))
    #for i in range(len(u)):
    #    id = np.where(config.numpy() == u[i])
    #    hist[i] = np.sum(freq.numpy()[id])
    #plt.figure(6)
    #plt.bar(np.arange(len(u)),hist)

    ept = tf.random.shuffle(np.concatenate(samples_lP_co,axis=1))
    #ept = tf.constant(np.concatenate(samples_lP_co,axis=1))
    nsteps = int(samples_lP_co[0].shape[0] / batch_size) ## samples.shape[0]=Ndataset + batchsize
    bcount = 0

    for epoch in range(EPOCHS):
        for idx in range(nsteps):
            print("time step", t, "epoch", epoch,"out of ", EPOCHS, 'nsteps', idx)
            if bcount*batch_size + batch_size>=Ndataset:
                bcount=0
                ept = tf.random.shuffle(np.concatenate((samples_lP_co),axis=1))
                #ept = tf.constant(np.concatenate((samples_lP_co),axis=1))
            batch = ept[ bcount*batch_size: bcount*batch_size+batch_size,:]
            bcount=bcount+1

            ## forward sampling training
            #if gtype == 1:
            #    ## batch[:,:-1] = samples in batch
            #    flip,co = flip1_tf(batch[:,:-1],gate,target_vocab_size,sites)
            #else:
            #    flip,co = flip2_tf(batch[:,:-1],gate,target_vocab_size,sites)
            #loss = train_step(flip,co,gtype,batch_size,optimizer,ansatz)
            ## reverse sampling training
            #loss = train_step2(batch,optmizer,ansatz)
            loss = train_step3(batch,optimizer,ansatz)

        print('loss:', loss)
        prob_povm_t = np.exp(vectorize(Nqubit, target_vocab_size, ansatz))
        pho_povm_t = ncon((prob_povm_t,povm.Ntn),([1],[1,-1,-2]))

        samp,llpp = ansatz.sample(10) # get samples from the mode
        print('Exact energy at time t:', np.trace(povm.ham @ pho_t))
        print('NN energy at time t:', np.trace(povm.ham @ pho_povm_t))
        #cFid, Fid = Fidelity_test(samp, llpp, Nqubit, target_vocab_size, mps, povm, prob, pho, ansatz)
        cFid_t, Fid_t = Fidelity_test(samp, llpp, Nqubit, target_vocab_size, mps, povm, prob_t, pho_t, ansatz)
        #cFid_t, Fid_t = Fidelity_test_mps(samp, llpp, Nqubit, target_vocab_size, mps, ansatz)
        #Fidelity.append(np.array([cFid, Fid]))
        Fidelity.append(np.array([cFid_t, Fid_t]))

    ansatz.save_weights('./models/transformer2', save_format='tf')


Fidelity = np.array(Fidelity)
np.savetxt('./data/Fidelity.txt',Fidelity)


## final fidelity
#cFid, Fid = Fidelity_test(samp, llpp, Nqubit, target_vocab_size, mps, povm, prob, pho, ansatz)
#cFid, Fid = Fidelity_test_mps(samp, llpp, Nqubit, target_vocab_size, mps, ansatz)
