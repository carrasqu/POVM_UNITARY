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
from transformer2 import *



### Parameters setting
num_layers = 2 #4
d_model = 128 #128
dff = 128 # 512
num_heads = 4 # 8


batch_size = 1000

target_vocab_size = 4 # number of measurement outcomes
input_vocab_size = target_vocab_size
dropout_rate = 0.0

MAX_LENGTH = 2 # number of qubits
povm_='4Pauli'
povm = POVM(POVM=povm_, Number_qubits=MAX_LENGTH)
mps = MPS(POVM=povm_,Number_qubits=MAX_LENGTH,MPS="Graph")
bias = povm.getinitialbias("+")

# define target state
psi = 1/2.0 * np.array([1.,1.,1.,-1], dtype=complex)
#psi = 1/np.sqrt(2.0) * np.array([1.,1.], dtype=complex)
pho = np.outer(psi, np.conjugate(psi))
prob = ncon((pho,povm.Mn),([1,2],[-1,2,1]))


EPOCHS = 1 ##20
j_init = 0
Ndataset = 4000 ## 100000
ansatz = Transformer(num_layers, d_model, MAX_LENGTH, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate,bias)


learning_rate = CustomSchedule(d_model)
#optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                     epsilon=1e-9)

optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

@tf.function
def train_step(flip,co,gtype, ansatz):

    with tf.GradientTape() as tape:
        loss = loss_function(flip,co,gtype,ansatz)

    gradients = tape.gradient(loss, ansatz.trainable_variables)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))


    return loss



#sys.exit(0)


if not os.path.exists("samples"):
    os.makedirs("samples")

## site [j,j+1] > epoch > nsteps = Ndataset/batchsize
for j in range(j_init,MAX_LENGTH-1):

    sites=[j,j+1] # on which sites to apply the gate
    gate = povm.p_two_qubit[1] # CZ gate

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
    samples = tf.stop_gradient(samples) # ?

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


                l = train_step(flip,co,gtype, ansatz)

                #samp,llpp = sample(100000) # get samples from the mode
                samp,llpp = sample(ansatz,1000) # get samples from the mode

                #np.savetxt('./samples/samplex_'+str(epoch)+'_iteration_'+str(idx)+'.txt',samp+1,fmt='%i')
                #np.savetxt('./samples/logP_'+str(epoch)+'_iteration_'+str(idx)+'.txt',llpp)
                cFid, cFidError, KL, KLError = mps.cFidelity(tf.cast(samp,dtype=tf.int64),llpp)
                Fid, FidErrorr = mps.Fidelity(tf.cast(samp,dtype=tf.int64))
                print('cFid: ', cFid, cFidError,Fid, FidErrorr)

                prob_povm = np.exp(vectorize(MAX_LENGTH, target_vocab_size, ansatz))
                pho_povm = ncon((prob_povm,povm.Ntn),([1],[1,-1,-2]))
                cFid2 = np.dot(np.sqrt(prob), np.sqrt(prob_povm))
                Fid2 = ncon((pho,pho_povm),([1,2],[2,1]))
                print('cFid2: ', cFid2, Fid2)

                print(epoch,idx,l)
                a = (np.array(list(it.product(range(4), repeat = 2)),dtype=np.uint8))
                l = np.sum(np.exp(logP(a, ansatz)))
                print("prob",l)


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


# classical fidelity
cFid, cFidError, KL, KLError = mps.cFidelity(tf.cast(samples,dtype=tf.int64),lP)
Fid, FidErrorr = mps.Fidelity(tf.cast(samples,dtype=tf.int64))
stabilizers,sError = mps.stabilizers_samples(tf.cast(samples,dtype=tf.int64))
print(cFid, cFidError,Fid, FidErrorr)
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




