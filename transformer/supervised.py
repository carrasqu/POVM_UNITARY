from __future__ import absolute_import, division, print_function, unicode_literals
from POVM import *
from matplotlib import pyplot as plt
from transformer3 import *
import itertools as it
import tensorflow as tf


def generate_samples(logP_fn, batch_size, Nqubit, base=4):
    cat = tf.random.categorical(logits=logP_fn, num_samples=int(batch_size))
    config = tf.map_fn(lambda x: np.array(list(np.base_repr(int(x), base=4, padding=Nqubit)[-Nqubit:]),dtype=int) , tf.transpose(cat))
    config_prob = tf.exp(tf.map_fn(lambda x: logP_fn[0,x], cat[0], dtype=tf.float32))
    return (cat, config, config_prob)



N = int(2)
a = POVM(POVM='Tetra',Number_qubits=N, eps=1e-2)
a.construct_ham()
a.construct_psi()
psi_t = a.psi.copy()
psi_g, E = a.ham_eigh()


# convert to POVM probability
pho_g = np.outer(psi_g, np.conjugate(psi_g))
prob_g = ncon((pho_g,a.Mn),([1,2],[-1,2,1])).real
log_prob = tf.math.log([prob_g+1e-13], dtype=tf.float32)
#cat2 = tf.random.categorical(logits=log_prob, num_samples=int(1e4))
#config = tf.map_fn(lambda x: np.array(list(np.base_repr(int(x), base=4, padding=N)[-N:]),dtype=int) , tf.transpose(cat2))
#config_prob = tf.map_fn(lambda x: prob_g[x], cat2[0], dtype=tf.float32)
cat2, config, config_prob = generate

plt.figure(1)
plt.hist(cat2[0], bins=4**N, density=True)
plt.figure(2)
plt.bar(np.arange(4**N),prob_g)
#plt.show()


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
loss_k = tf.keras.losses.KLDivergence()


assert False, 'stop'


def loss_fn(config, config_prob, ansatz):

	lnP = ansatz(config,training=True)
	loss = -tf.reduce_mean(lnP) # KL div
	#loss = tf.reduce_mean(tf.abs(tf.exp(lnP)-config_prob))
	#loss = loss_k(config_prob, tf.exp(lnP))
	return loss #, loss2


@tf.function
def train_step(loss_fn, optimizer, epoch, samples, samples_p, ansatz):
	with tf.GradientTape() as tape:
		loss = loss_fn(samples,samples_p,ansatz)

	gradients = tape.gradient(loss, ansatz.trainable_variables)
	optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))
	return loss


for i in range(10):

	loss_tr = train_step(config, config_prob, ansatz)
	prob_f = np.exp(ansatz(config_b))
	print(loss_tr)
	print(np.sum(prob_f))


#ansatz.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
#ansatz.fit(config, tf.math.log(config_prob+1e-13), epochs=10, batch_size=None)


prob_f = np.exp(ansatz(config_b))
plt.figure(4)
plt.bar(np.arange(4**N),prob_f)
cFid2 = np.dot(np.sqrt(prob_f), np.sqrt(prob_g))
print('final fidelity:', cFid2)
s = np.dot(prob_g, np.log(prob_g))
print('entropy:', s)

