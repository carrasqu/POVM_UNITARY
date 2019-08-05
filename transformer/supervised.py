from __future__ import absolute_import, division, print_function, unicode_literals
from POVM import *
from matplotlib import pyplot as plt
from transformer3 import *
import itertools as it
import tensorflow as tf

@tf.function
def generate_samples(log_P, batch_size, size, Nqubit, base=4):
    steps = int(size / batch_size)

    CAT = []
    CONFIG = []
    CONFIG_P = []
    for i in range(steps):
        cat = tf.random.categorical(logits=log_P, num_samples=int(batch_size), dtype=tf.int32)
        config = tf.math.floormod(tf.bitwise.right_shift(tf.expand_dims(cat[0],1), tf.range(Nqubit)), 2)
         #config = tf.map_fn(lambda x: np.array(list(np.base_repr(int(x), base=4, padding=Nqubit)[-Nqubit:]),dtype=np.float32) , tf.transpose(cat))

        ## TODO: use gather
        config_prob = tf.exp(tf.map_fn(lambda x: log_P[0,x], cat[0], dtype=tf.float32))
        config_prob = tf.expand_dims(config_prob,1)

        CAT.append(cat[0])
        CONFIG.append(config)
        CONFIG_P.append(config_prob)

    CAT = tf.convert_to_tensor(CAT)
    CONFIG = tf.convert_to_tensor(CONFIG, dtype=np.float32)
    CONFIG_P = tf.convert_to_tensor(CONFIG_P)

    return (CAT, CONFIG, CONFIG_P)


def loss_fn(config, config_prob, ansatz):
    lnP = ansatz(config,training=True)
    loss = -tf.reduce_mean(lnP) # KL div
    #loss = tf.reduce_mean(tf.abs(tf.exp(lnP)-config_prob))
    #loss_k = tf.keras.losses.KLDivergence()
    #print('loss:', loss)
    return loss


@tf.function
def train_step(loss_fn, optimizer, samples, samples_p, ansatz):
    ept = tf.random.shuffle(tf.concat([samples,samples_p],axis=-1))

    nsteps = ept.shape[0]
    bcount = 0
    loss = 0.

    for idx in range(nsteps):
        print('nsteps', idx)
        batch = ept[idx]
        config = tf.cast(batch[:,:-1], dtype=tf.int32)
        config_prob = batch[:,-1]
        bcount = bcount + 1
        with tf.GradientTape() as tape:
            loss = loss_fn(config,config_prob,ansatz)

        gradients = tape.gradient(loss, ansatz.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))

    return loss



## initial setup
N = int(4)
batch_size =1e2
sample_size = 2e2
a = POVM(POVM='Tetra',Number_qubits=N, eps=1e-2)
a.construct_Nframes()
a.construct_ham()
a.construct_psi()
psi_t = a.psi.copy()
psi_g, E = a.ham_eigh()


# convert to POVM probability
pho_g = np.outer(psi_g, np.conjugate(psi_g))
prob_g = ncon((pho_g,a.Mn),([1,2],[-1,2,1])).real.astype(np.float32)
log_prob = tf.math.log([prob_g+1e-13])

cat, samples, samples_p = generate_samples(log_prob, batch_size, size=sample_size, Nqubit=N, base=4)

plt.figure(1)
plt.hist(np.reshape(cat,-1), bins=4**N, density=True)
plt.figure(2)
plt.bar(np.arange(4**N),prob_g)
#plt.show()
assert False, 'stop'


# construct ansatz
num_layers = 2 #4
d_model = 128 #128
dff = 128 # 512
num_heads = 4 # 8
target_vocab_size = 4 # number of measurement outcomes
input_vocab_size = target_vocab_size
dropout_rate = 0.0
MAX_LENGTH = N
bias = a.getinitialbias("+")
ansatz = Transformer(num_layers, d_model, MAX_LENGTH, num_heads, dff,input_vocab_size, target_vocab_size, dropout_rate,bias)

# check ansatz initial POVM probality
config_b = np.array(list(it.product(range(4),repeat = MAX_LENGTH)), dtype=np.uint8)
prob_i = np.exp(ansatz(config_b))
l = np.sum(np.exp(ansatz(config_b)))
plt.figure(3)
plt.bar(np.arange(4**N),prob_i)
cFid2 = np.dot(np.sqrt(prob_i), np.sqrt(prob_g))
print('initial fidelity:', cFid2)




# training
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

for i in range(10):
    loss_tr = train_step(loss_fn, optimizer, samples, samples_p, ansatz)
    prob_f = np.exp(ansatz(config_b))
    print('epoch complete loss:', loss_tr)
    print('prob', np.sum(prob_f))


#ansatz.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
#ansatz.fit(config, tf.math.log(config_prob+1e-13), epochs=10, batch_size=None)


prob_f = np.exp(ansatz(config_b))
plt.figure(4)
plt.bar(np.arange(4**N),prob_f)
cFid2 = np.dot(np.sqrt(prob_f), np.sqrt(prob_g))
print('final fidelity:', cFid2)
s = np.dot(prob_g, np.log(prob_g))
print('entropy:', s)

