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
from transformer2 import *


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
povm = POVM(POVM=povm_, Number_qubits=MAX_LENGTH, initial_state='+',Jz=1.0,hx=1.0,eps=10./float(T))

mps = MPS(POVM=povm_,Number_qubits=MAX_LENGTH,MPS="Graph")
bias = povm.getinitialbias("+")

# define target state
povm.construct_psi()
povm.construct_ham()
psi, E = povm.ham_eigh()
psi = 1/2.0 * np.array([1.,1.,1.,-1], dtype=complex)
pho = np.outer(psi, np.conjugate(psi))
prob = ncon((pho,povm.Mn),([1,2],[-1,2,1])).real


# define ansatz
ansatz = Transformer(num_layers, d_model, MAX_LENGTH, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate,bias)

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

Fidelity=[]
for t in range(T):
  ## site [j,j+1] > epoch > nsteps = Ndataset/batchsize
  for j in range(j_init,MAX_LENGTH-1):

      sites=[j,j+1] # on which sites to apply the gate
      gate = povm.p_two_qubit[1] # CZ gate
      #gate = povm.Up # imaginary time evolution
      #gate = povm.Up2 # imaginary time evolution

      if Ndataset != 0:
          ## it ensures at least one batch size samples, since Ncall can be zero
          Ncalls = Ndataset /batch_size
          samples,lP = sample(ansatz, batch_size) # get samples from the model
          lP = np.reshape(lP,[-1,1]) ## not necessary

          for k in range(int(Ncalls)):
              sa,llpp = sample(ansatz, batch_size)
              samples = np.vstack((samples,sa))
              llpp =np.reshape(llpp,[-1,1])
              lP =  np.vstack((lP,llpp))

      gtype = 2 # 2-qubit gate

      nsteps = int(samples.shape[0] / batch_size) ## samples.shape[0]=Ndataset + batchsize
      bcount = 0
      counter=0
      samples = tf.stop_gradient(samples)

      ept = tf.random.shuffle(samples)


      for epoch in range(EPOCHS):

              print("epoch", epoch,"out of ", EPOCHS,"site", j)
              for idx in range(nsteps):

                  if bcount*batch_size + batch_size>=Ndataset:
                      bcount=0
                      ept = tf.random.shuffle(samples)

                  batch = ept[ bcount*batch_size: bcount*batch_size+batch_size,:]
                  bcount=bcount+1


                  flip,co = flip2_tf(batch,gate,target_vocab_size,sites)


                  loss = train_step(flip,co,gtype,batch_size,ansatz)

                  #samp,llpp = sample(100000) # get samples from the mode
                  samp,llpp = sample(ansatz,1000) # get samples from the mode

                  #np.savetxt('./samples/samplex_'+str(epoch)+'_iteration_'+str(idx)+'.txt',samp+1,fmt='%i')
                  #np.savetxt('./samples/logP_'+str(epoch)+'_iteration_'+str(idx)+'.txt',llpp)
                  cFid, cFidError, KL, KLError = mps.cFidelity(tf.cast(samp,dtype=tf.int64),llpp)
                  Fid, FidErrorr = mps.Fidelity(tf.cast(samp,dtype=tf.int64))
                  print('cFid: ', cFid, cFidError,Fid, FidErrorr)

                  prob_povm = np.exp(vectorize(MAX_LENGTH, target_vocab_size, ansatz))
                  pho_povm = ncon((prob_povm,povm.Ntn),([1],[1,-1,-2]))
                  #Et = np.trace(pho_povm @ povm.ham)
                  #print('exact E:', E, 'current E:', Et.real)
                  cFid2 = np.dot(np.sqrt(prob), np.sqrt(prob_povm))
                  Fid2 = ncon((pho,pho_povm),([1,2],[2,1]))
                  print('cFid2: ', cFid2, Fid2)
                  Fidelity.append(np.array([cFid2, Fid2]))

                  print('time:', t, 'epoch:', epoch, 'step:', idx)
                  print('loss', loss)
                  a = np.array(list(it.product(range(4),repeat = MAX_LENGTH)), dtype=np.uint8)
                  l = np.sum(np.exp(logP(a, ansatz)))
                  print("prob", l)


  ansatz.save_weights('./models/transformer2', save_format='tf')

Fidelity = np.array(Fidelity)
np.savetxt('./data/Fidelity.txt',Fidelity)



#samples,lnP = sample(Ndataset)
if Ndataset != 0:
    Ncalls = Ndataset /batch_size
    samples,lP = sample(ansatz,batch_size) # get samples from the model
    lP = np.reshape(lP,[-1,1])

    for k in range(int(Ncalls)):
        sa,llpp = sample(ansatz,batch_size)
        samples = np.vstack((samples,sa))
        llpp =np.reshape(llpp,[-1,1])
        lP =  np.vstack((lP,llpp))


# classical fidelity from mps
#cFid, cFidError, KL, KLError = mps.cFidelity(tf.cast(samples,dtype=tf.int64),lP)
#Fid, FidErrorr = mps.Fidelity(tf.cast(samples,dtype=tf.int64))
#stabilizers,sError = mps.stabilizers_samples(tf.cast(samples,dtype=tf.int64))
#print(cFid, cFidError,Fid, FidErrorr)
#print(stabilizers,sError,np.mean(stabilizers),np.mean(sError))

# calssical fidelity in vector form
#prob = ncon((pho,povm.M),([1,2],[-1,2,1]))
#prob_povm = np.exp(povm.bias)
#pho_povm = ncon((prob_povm,povm.Nt),([1],[1,-1,-2]))

prob = ncon((pho,povm.Mn),([1,2],[-1,2,1]))
prob_povm = np.exp(vectorize(MAX_LENGTH, target_vocab_size, ansatz))
pho_povm = ncon((prob_povm,povm.Ntn),([1],[1,-1,-2]))
cFid2 = np.dot(np.sqrt(prob), np.sqrt(prob_povm))
Fid2 = ncon((pho,pho_povm),([1,2],[2,1]))
print(cFid2, Fid2)

'''
ff = flip + 0.
ff2 = tf.transpose(ff,perm=[1,0])
val = tf.constant(np.random.rand(16000,1),dtype=np.float32)
val2 = tf.transpose(val,perm=[1,0])
a1 = slicetf.replace_slice_in(ff)[:,1].with_value(val)
ind = tf.constant([[1]])
a2 = tf.tensor_scatter_nd_update(ff2, ind, val2)
a2 = tf.transpose(a2, perm=[1,0])

indices = tf.constant([[0], [2]])
updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                        [7, 7, 7, 7], [8, 8, 8, 8]],
                       [[15, 25, 25, 5], [26, 36, 46, 46],
                        [17, 27, 27, 7], [28, 38, 48, 48]]],dtype=tf.float32)
tensor = tf.ones([4, 4, 4])
updated = tf.tensor_scatter_nd_update(tensor, indices, updates)
'''

## flip test
t0=time.time()
flip,co = flip2_tf(batch,gate,target_vocab_size,sites)
t1=time.time()
print(t1-t0)
t0=time.time()
flip2,co2 = flip2_tf2(batch,gate,target_vocab_size,sites)
t1=time.time()
print(t1-t0)
print(np.linalg.norm(flip-flip2))
print(np.linalg.norm(co-co2))


