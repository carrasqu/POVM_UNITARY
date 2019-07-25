from POVM import *

N = 2
a = POVM(POVM='Tetra',Number_qubits=N)

# check dual frame correctness
for i in range(N):
  b1 = a.p_single_qubit[i] - a.p_single_qubit2[i]
  b2 = a.p_two_qubit[i] - a.p_two_qubit2[i]
  print(np.linalg.norm(b1))
  print(np.linalg.norm(b2))
  if np.linalg.norm(b1) >= 1e-14 and np.linalg.norm(b2) >= 1e-14:
    assert False, 'dual frame test fails'
print('pass dual frame test')


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
