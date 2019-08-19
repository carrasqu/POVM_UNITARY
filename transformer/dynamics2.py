from __future__ import absolute_import, division, print_function, unicode_literals
from ncon import ncon

import tensorflow as tf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from POVM import POVM
import itertools as it
from MPS import MPS
import functools
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
num_batch = int(Ndataset/batch_size)
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
ansatz_copy = Transformer(num_layers, d_model, Nqubit, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate,bias)
if LOAD==1:
  print('load model')
  ansatz.load_weights('./models/transformer2')

ansatz.sample(10)
ansatz_copy.sample(10)
ansatz_copy.set_weights(ansatz.get_weights())


# define target state
povm.construct_psi()
povm.construct_Nframes()
povm.construct_ham()
psi, E = povm.ham_eigh()
## GHZ state
psi = np.zeros(2**Nqubit)
psi[0] = 1.
psi[-1] = 1.
psi = psi/ np.sqrt(2)

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
print('initial fidelity consistent:', cFid_t, Fid_t)

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


@tf.function
def batch_training_ham(num_batch, batch_size, Nqubit, target_vocab_size, hl, hlx, tau, optimizer, ansatz, ansatz_copy):

  for _ in tf.range(num_batch):
    samples_lP_co = reverse_samples_ham_tf(batch_size, Nqubit, target_vocab_size, hl, hlx, tau, ansatz, ansatz_copy)
    loss_fn = functools.partial(loss_function3,samples_lP_co,ansatz)
    optimizer.minimize(loss=loss_fn, var_list=ansatz.trainable_variables)
    #with tf.GradientTape() as tape:
    #    loss = loss_function3(samples_lP_co,ansatz)

    #gradients = tape.gradient(loss, ansatz.trainable_variables)
    #optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))


@tf.function
def batch_training_gate(num_batch, batch_size, Nqubit, target_vocab_size, gate, sites, gtype, optimizer, ansatz, ansatz_copy):

  for _ in tf.range(num_batch):
    samples_lP_co = reverse_samples_tf(batch_size, Nqubit, target_vocab_size, gate, sites, gtype, ansatz, ansatz_copy)
    loss_fn = functools.partial(loss_function2,samples_lP_co,ansatz)
    optimizer.minimize(loss=loss_fn, var_list=ansatz.trainable_variables)
    #with tf.GradientTape() as tape:
    #    loss = loss_function3(samples_lP_co,ansatz)

    #gradients = tape.gradient(loss, ansatz.trainable_variables)
    #optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))




#start = time.time()
#num_batch = int(5)
#batch_size = int(10)
#sites=[1,3]
#gate = povm.P_gate(povm.cnot)
#gtype = int(gate.ndim/2)
#K = 4
#S = ansatz.sample(2)[0]
#fp,co = flip_reverse_tf(S,gate,K,sites,gtype)
#optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9) ## lr=1e-4
##batch_training_ham(num_batch, batch_size, Nqubit, target_vocab_size, povm.hl_com, povm.hlx_com, tau, optimizer, ansatz, ansatz_copy)
#batch_training_gate(num_batch, batch_size, Nqubit, target_vocab_size, gate, sites, gtype, optimizer, ansatz, ansatz_copy)
#ansatz_copy.set_weights(ansatz.get_weights())
#end = time.time()
#print('time', start-end)

optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9) ## lr=1e-4


Fidelity=[]
for t in range(T):

  SITE = [[0]]
  GATE = [povm.H]
  P_GATE = [povm.P_gate(povm.H)]
  for i in range(Nqubit-1):
      SITE.append([i,i+1])
      GATE.append(povm.cnot)
      P_GATE.append(povm.P_gate(povm.cnot))
  #SITE = [[0,1]]
  #GATE = [povm.mat2]
  #P_GATE = [povm.P_gate(povm.mat2)]

  for i in range(len(SITE)):

    sites=SITE[i]
    gate = P_GATE[i]
    gtype = int(gate.ndim/2)
    kron_gate = povm.kron_gate(GATE[i], sites[0], Nqubit)
    psi_t = psi_t @ kron_gate
    psi_t = psi_t / np.linalg.norm(psi_t)
    pho_t = np.outer(psi_t, np.conjugate(psi_t))
    prob_t = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real

    #optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9) ## lr=1e-4
    batch_training_gate(num_batch, batch_size, Nqubit, target_vocab_size, gate, sites, gtype, optimizer, ansatz, ansatz_copy)
    #batch_training_ham(num_batch, batch_size, Nqubit, target_vocab_size, povm.hl_com, povm.hlx_com, tau, optimizer, ansatz, ansatz_copy)
    ansatz_copy.set_weights(ansatz.get_weights())

    samp,llpp = ansatz.sample(10) # get samples from the mode
    #cFid, Fid = Fidelity_test(samp, llpp, MAX_LENGTH, target_vocab_size, mps, povm, prob, pho, ansatz)
    cFid_t, Fid_t = Fidelity_test(samp, llpp, Nqubit, target_vocab_size, mps, povm, prob_t, pho_t, ansatz)
    #cFid_t, Fid_t = Fidelity_test_mps(samp, llpp, MAX_LENGTH, target_vocab_size, mps, ansatz)
    #Fidelity.append(np.array([cFid, Fid]))
    Fidelity.append(np.array([cFid_t, Fid_t]))

    ansatz.save_weights('./models/transformer2', save_format='tf')


assert False, 'stop'
Fidelity = np.array(Fidelity)
np.savetxt('./data/Fidelity.txt',Fidelity)


## final fidelity
#cFid, Fid = Fidelity_test(samp, llpp, Nqubit, target_vocab_size, mps, povm, prob, pho, ansatz)
#cFid, Fid = Fidelity_test_mps(samp, llpp, Nqubit, target_vocab_size, mps, ansatz)
