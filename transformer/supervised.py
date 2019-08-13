from __future__ import absolute_import, division, print_function, unicode_literals
from POVM import *
from matplotlib import pyplot as plt
from transformer3 import *
from utils import *
import itertools as it
import tensorflow as tf


np.random.seed(30)
tf.compat.v1.random.set_random_seed(1234)


#@tf.function
#TODO: tf.function version
def generate_samples(log_P, batch_size, size, Nqubit, base=4):
    steps = int(size / batch_size)

    CAT = []
    CONFIG = []
    CONFIG_P = []
    for i in range(steps):
        cat = tf.random.categorical(logits=log_P, num_samples=int(batch_size), dtype=tf.int32)
        config = tf.map_fn(lambda x: np.array(list(np.base_repr(int(x), base=4, padding=Nqubit)[-Nqubit:]),dtype=np.float32) , tf.transpose(cat))

        ## TODO: use gather
        config_prob = tf.exp(tf.map_fn(lambda x: log_P[0,x], cat[0], dtype=tf.float32))
        config_prob = tf.expand_dims(config_prob,1)

        CAT.append(cat[0])
        CONFIG.append(config)
        CONFIG_P.append(config_prob)

    CAT = tf.stop_gradient(tf.convert_to_tensor(CAT))
    CONFIG = tf.stop_gradient(tf.convert_to_tensor(CONFIG, dtype=np.float32))
    CONFIG_P = tf.stop_gradient(tf.convert_to_tensor(CONFIG_P))

    return (CAT, CONFIG, CONFIG_P)


def loss_fn(config, config_prob, ansatz):
    lnP = tf.squeeze(ansatz(config,training=True))
    config_prob = tf.squeeze(config_prob)
    #loss = -tf.reduce_mean(lnP)  # KL div
    loss = -tf.reduce_mean(config_prob * lnP)
    #loss = tf.reduce_mean(tf.abs(tf.exp(lnP)-config_prob))
    #loss_k = tf.keras.losses.KLDivergence()
    #print('loss:', loss)
    return loss


#@tf.function
def train_step(loss_fn, optimizer, samples, samples_p, ansatz):
    ept = tf.random.shuffle(tf.concat([samples,samples_p],axis=-1))

    nsteps = ept.shape[0]

    for idx in range(nsteps):
        #print('nsteps', idx)
        batch = ept[idx]
        config = tf.cast(batch[:,:-1], dtype=tf.int32)
        config_prob = batch[:,-1]
        with tf.GradientTape() as tape:
            loss = loss_fn(config,config_prob,ansatz)

        gradients = tape.gradient(loss, ansatz.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))

    return loss



## initial setup

N = int(4)
Nqubit = N
num_layers = 1 #4
d_model = 16 #128
dff = 16 # 512
num_heads = 1 # 8
target_vocab_size = 4 # number of measurement outcomes
input_vocab_size = target_vocab_size
dropout_rate = 0.0

povm_='Tetra_pos'
initial_state='0'
T=1
tau = 0.1/float(T)
povm = POVM(POVM=povm_, Number_qubits=Nqubit, initial_state=initial_state,Jz=1.0,hx=1.0,eps=tau)
mps = MPS(POVM=povm_,Number_qubits=Nqubit,MPS="GHZ")
bias = povm.getinitialbias(initial_state)

# define ansatz
ansatz = Transformer(num_layers, d_model, Nqubit, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate,bias)


# define target state
povm.construct_psi()
povm.construct_Nframes()
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
plt.figure(1)
plt.bar(np.arange(4**Nqubit),prob_t)
#plt.figure(2)
#plt.bar(np.arange(4**Nqubit),prob_povm)


pho_t = pho_t - tau *( povm.ham @ pho_t + pho_t @ povm.ham)
prob_t_raw = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real.astype(np.float32)
pho_t = pho_t / np.trace(pho_t)
prob_t = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real.astype(np.float32)
plt.figure(3)
plt.bar(np.arange(4**Nqubit),prob_t)


#samples_lP_co = reverse_samples_ham(256, 256, Nqubit, target_vocab_size, povm.hl_com, povm.hlx_com, tau, ansatz)
#sa = tf.cast(samples_lP_co[0], dtype=tf.float32)
#lp = samples_lP_co[1]
#up_pi = samples_lP_co[2]
#update_Pi = up_pi * tf.exp(lp)
#
#config = tf.map_fn(lambda x: index(x), sa)
#u, ind = np.unique(config, return_index=True)
#update_Pi = update_Pi.numpy()[ind,0].reshape(-1)
#
#
#plt.figure()
#plt.bar(np.arange(4**Nqubit),update_Pi)
##plt.bar(np.arange(4**Nqubit),update_Pi[:,0].numpy())




# generate samples
batch_size =int(5e4)
sample_size = int(1e1)
log_prob_t = tf.math.log([prob_t+1e-13])


#cat, samples, samples_p = generate_samples(log_prob_t, batch_size, size=sample_size, Nqubit=N, base=4)

#plt.figure(4)
#plt.hist(np.reshape(cat,-1), bins=4**N, density=True)


samples_lP_co = reverse_samples_ham(sample_size, batch_size, Nqubit, target_vocab_size, povm.hl_com, povm.hlx_com, tau, ansatz)

sa = tf.cast(samples_lP_co[0], dtype=tf.float32)
lp = samples_lP_co[1]
up_pi = samples_lP_co[2]
up_pi2 = up_pi * tf.exp(lp)
#freq = (tf.ones([sa.shape[0],]) - up_pi / tf.exp(lp)) / tf.cast(sa.shape[0],tf.float32)
freq = up_pi / tf.cast(sa.shape[0],tf.float32)
config = tf.map_fn(lambda x: index(x), sa)
#plt.figure(5)
#plt.hist(config, bins=4**Nqubit, density=True)
u, ind = np.unique(config, return_index=True)
up_pi3 = up_pi2.numpy()[ind,0]
plt.figure(6)
plt.bar(np.arange(len(u)),up_pi3)

hist = np.zeros(len(u))
count = 0.
for i in range(len(u)):
    id = np.where(config.numpy() == u[i])
    count += id[0].shape[0]
    hist[i] = np.sum(freq.numpy()[id,0])
plt.figure(7)
plt.bar(np.arange(len(u)),hist)
#sa = tf.reshape(sa,[-1,batch_size,int(N)])
#lp = tf.reshape(lp,[-1,batch_size,1])
#up_pi = tf.reshape(up_pi,[-1,batch_size,1])

##ept = tf.random.shuffle(np.concatenate(samples_lP_co,axis=1))
#ept = tf.constant(np.concatenate(samples_lP_co,axis=1))
print('diff', np.linalg.norm(prob_t[u.astype(np.int)] - up_pi3/np.sum(up_pi3)))
print('raw_diff', np.linalg.norm(prob_t_raw[u.astype(np.int)] - up_pi3))
assert False, 'stop'


# training
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

print('initial prob diff:', np.linalg.norm(prob_t-prob_povm, ord=1))
Fidelity=[]
for i in range(100):
    loss_tr = train_step(loss_fn, optimizer, samples=sa, samples_p=up_pi, ansatz=ansatz)
    #loss_tr = train_step(loss_fn, optimizer, samples, samples_p, ansatz=ansatz)
    print('epoch complete loss:', loss_tr, 'epoch:', i)
    prob_f = np.exp(vectorize(Nqubit, target_vocab_size, ansatz))
    prob_diff = np.linalg.norm(prob_t-prob_f, ord=1)
    print('prob', np.sum(prob_f))
    print('prob diff:', prob_diff)
    Fidelity.append(np.array([prob_diff, loss_tr]))

Fidelity = np.array(Fidelity)

plt.figure(8)
plt.bar(np.arange(4**Nqubit),prob_f)

s1 = -np.dot(prob_t, np.log(prob_t+1e-13))
s2 = -np.dot(prob_t, np.log(prob_f+1e-13))
print('self entropy:', s1)
print('cross entropy:', s2)

assert False, 'stop'
