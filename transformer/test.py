from __future__ import absolute_import, division, print_function, unicode_literals
from POVM import *
from matplotlib import pyplot as plt
from transformer2 import *
import itertools as it
import tensorflow as tf


N = int(2)
a = POVM(POVM='Tetra',Number_qubits=N, eps=1e-2)
b = POVM(POVM='Tetra',Number_qubits=N, eps=1e2)

# check dual frame correctness
#for i in range(N):
#  b1 = a.p_single_qubit[i] - a.p_single_qubit2[i]
#  b2 = a.p_two_qubit[i] - a.p_two_qubit2[i]
#  print(np.linalg.norm(b1))
#  print(np.linalg.norm(b2))
#  if np.linalg.norm(b1) >= 1e-14 and np.linalg.norm(b2) >= 1e-14:
#    assert False, 'dual frame test fails'
#print('pass dual frame test')


# construct 4 qubit wave function and reshape its tensor form
np.random.seed(30)
psi = np.random.rand(2**N)*2 - 1 + 1j*(np.random.rand(2**N)*2 - 1)
psi = psi / np.linalg.norm(psi)
pho = np.outer(psi, np.conjugate(psi))
w, v = np.linalg.eigh(pho)
print(np.trace(pho))
print(np.trace(pho @ pho))
print(np.linalg.norm(pho-np.conjugate(pho.transpose())))
print(w)
pho1 = pho.copy()
size = np.ones(2*N, dtype=int) * 2
pho1 = np.reshape(pho1, size)


# convert POVM probability to density matrix in vector basis
Nt2 = ncon((a.Nt,a.Nt),([-1,-3,-5],[-2,-4,-6]))
N3 = np.reshape(Nt2, (4,4,4,4))
N4 = np.reshape(Nt2, (16,4,4))

M2 = ncon((a.M,a.M),([-1,-3,-5],[-2,-4,-6]))
M3 = np.reshape(M2, (4,4,4,4))
M4 = np.reshape(M2, (16,4,4))

prob = np.random.rand(4**2)
prob = prob / np.sum(prob)
prob2 = np.reshape(prob, (4,4))

pho3 = ncon((N3,prob2),([1,2,-1,-2],[1,2]))
pho4 = ncon((N4,prob), ([1,-1,-2],[1]))
print('glue index works:', np.linalg.norm(pho3-pho4)<1e-14)

pho5 = pho4.copy()
prob3 = ncon((pho5,M3),([1,2],[-1,-2,2,1]))
prob4 = ncon((pho5,M4),([1,2],[-1,2,1]))
print(np.linalg.norm(prob3-prob2)<1e-14)
print(np.linalg.norm(prob4-prob)<1e-14)

# produce Nqubit tensor product Nt and M
Ntn = a.Nt.copy()
Mn = a.M.copy()
for i in range(N-1):
  Ntn = ncon((Ntn, a.Nt),([-1,-3,-5],[-2,-4,-6]))
  Mn = ncon((Mn, a.M),([-1,-3,-5],[-2,-4,-6]))
  Ntn = np.reshape(Ntn, (4**(i+2),2**(i+2),2**(i+2)))
  Mn = np.reshape(Mn, (4**(i+2),2**(i+2),2**(i+2)))
probn = ncon((pho,a.Mn),([1,2],[-1,2,1]))
phon = ncon((probn,a.Ntn),([1],[1,-1,-2]))
print('Nqubit works:', np.linalg.norm(pho-phon)<1e-14)


# test two qubit operator
b1 = np.kron(a.Z,a.Z)
b2 = ncon((a.Z,a.Z),([-1,-3],[-2,-4]))
b2 = np.reshape(b2,(4,4))


# test imaginary time evolution
a.construct_ham()
a.construct_psi()
psi_t = a.psi.copy()
psi_g, E = a.ham_eigh()
diff = a.ham @ psi_g - E * psi_g
print('diagonalization works:', np.linalg.norm(diff)<1e-14)
tau = 0.1
expH = expm(-tau * a.ham)
for i in range(100):
  psi_t = expH @ psi_t
  psi_t = psi_t / np.linalg.norm(psi_t)


Et = np.conjugate(psi_t.transpose()) @ a.ham @ psi_t
fidelity = np.abs(np.dot(psi_t,psi_g))
print('full imaginary time fidelity:', fidelity)
print('full imaginary time energy diff:', Et-E)


# check dual frame correctness
Up2 = a.two_body_gate(a.mat2)
print('dual frame works:', np.linalg.norm(Up2-a.Up2)<1e-14)
p1=[]
p2=[]
for i in range(4):
  p1.append(a.one_body_gate(a.single_qubit[i]))
  p2.append(a.two_body_gate(a.two_qubit[i]))
  print('dual frame works:', np.linalg.norm(p1[i]-a.p_single_qubit[i])<1e-14)
  print('dual frame works:', np.linalg.norm(p2[i]-a.p_two_qubit[i])<1e-14)


# sampling function
psi_p = np.multiply(psi_g, np.conjugate(psi_g)).real
log_psi = tf.math.log([psi_p+1e-13])
cat = tf.random.categorical(logits=log_psi, num_samples=int(1e3))
#plt.figure(1)
#plt.hist(cat[0], bins=2**N, density=True)
#plt.figure(2)
#plt.bar(np.arange(2**N),psi_p)
#plt.show()

# convert to POVM probability
pho_g = np.outer(psi_g, np.conjugate(psi_g))
prob_g = ncon((pho_g,a.Mn),([1,2],[-1,2,1])).real
log_prob = tf.math.log([prob_g+1e-13])
cat2 = tf.random.categorical(logits=log_prob, num_samples=int(1e4))
plt.figure(1)
plt.hist(cat2[0], bins=4**N, density=True)
plt.figure(2)
plt.bar(np.arange(4**N),prob_g)
#plt.show()

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

config_b = np.array(list(it.product(range(4),repeat = MAX_LENGTH)), dtype=np.uint8)
prob_i = np.exp(logP(config_b, ansatz))
l = np.sum(np.exp(logP(config_b, ansatz)))
plt.figure(3)
plt.bar(np.arange(4**N),prob_i)
cFid2 = np.dot(np.sqrt(prob_i), np.sqrt(prob_g))
print('initial fidelity:', cFid2)


config = tf.map_fn(lambda x: np.array(list(np.base_repr(int(x), base=4, padding=N)[-N:]),dtype=int) , tf.transpose(cat2))
config_prob = tf.map_fn(lambda x: prob_g[x], cat2[0], dtype=tf.float32)
config_tmp = config + 0
#log_P = logP(config, ansatz)

# training
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_k = tf.keras.losses.KLDivergence()


def loss_test(config, config_prob,ansatz):

	lnP = logP(config,ansatz,training=True)
	loss = -tf.reduce_mean(lnP) # KL div
	#loss = tf.reduce_mean(tf.abs(tf.exp(lnP)-config_prob))
	#loss = loss_k(config_prob, tf.exp(lnP))
	return loss #, loss2



@tf.function
def train_step(samples,samples_p, ansatz):
	with tf.GradientTape() as tape:
		loss = loss_test(samples, samples_p,ansatz)

	gradients = tape.gradient(loss, ansatz.trainable_variables)
	optimizer.apply_gradients(zip(gradients, ansatz.trainable_variables))
	return loss


for i in range(10):
	cat2 = tf.random.categorical(logits=log_prob, num_samples=int(1e4))
	config = tf.map_fn(lambda x: np.array(list(np.base_repr(int(x), base=4, padding=N)[-N:]),dtype=int) , tf.transpose(cat2))
	config_prob = tf.map_fn(lambda x: prob_g[x], cat2[0], dtype=tf.float32)

	loss_tr = train_step(config, config_prob, ansatz)
	prob_f = np.exp(logP(config_b, ansatz))
	print(loss_tr)
	print(np.sum(prob_f))

#ansatz.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
#ansatz.fit(config, config_prob, epochs=10, batch_size=1)


prob_f = np.exp(logP(config_b, ansatz))
plt.figure(4)
plt.bar(np.arange(4**N),prob_f)
cFid2 = np.dot(np.sqrt(prob_f), np.sqrt(prob_g))
print('final fidelity:', cFid2)
s = np.dot(prob_g, np.log(prob_g))
print('entropy:', s)

