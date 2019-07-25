import numpy as np
from ncon import ncon
#import tensorly as tl
#from tensorly.decomposition import matrix_product_state
from copy import deepcopy
from scipy.linalg import expm


def basis(num_sites, base=2):
  l_basis = []
  for i in range(base**num_sites):
    basis_str = np.base_repr(i, base=base, padding=num_sites)[-num_sites:]
    l_basis.append(np.array(list(basis_str), dtype=int))
    #l_basis.append(basis_str)
  l_basis = np.array(l_basis)
  return l_basis

def index(one_basis, base=2):
  return int(''.join(map(lambda x: str(int(x)), one_basis)), base)



class POVM():
    def __init__(self, POVM='4Pauli',Number_qubits=4,initial_state='+',Jz=1.0,hx=1.0,eps=0.0001):

        self.N = Number_qubits;
        # Hamiltonian for calculation of energy (TFIM in 1d)
        self.Jz = Jz
        self.hx = hx
        self.eps = eps

        # POVMs and other operators
        # Pauli matrices,gates,simple states
        self.I = np.array([[1, 0],[0, 1]]);
        self.X = np.array([[0, 1],[1, 0]]);    self.s1 = self.X;
        self.Z = np.array([[1, 0],[0, -1]]);   self.s3 = self.Z;
        self.Y = np.array([[0, -1j],[1j, 0]]); self.s2 = self.Y;
        self.H = 1.0/np.sqrt(2.0)*np.array( [[1, 1],[1, -1 ]] )
        self.Sp = np.array([[1.0, 0.0],[0.0, -1j]])
        self.oxo = np.array([[1.0, 0.0],[0.0, 0.0]])
        self.IxI = np.array([[0.0, 0.0],[0.0, 1.0]])
        self.Phase = np.array([[1.0, 0.0],[0.0, 1j]]) # =S = (Sp)^{\dag}
        self.T = np.array([[1.0,0],[0,np.exp(-1j*np.pi/4.0)]])
        self.U1 = np.array([[np.exp(-1j*np.pi/3.0),  0] ,[ 0 ,np.exp(1j*np.pi/3.0)]])

        #two-qubit gates
        self.cy = ncon((self.oxo,self.I),([-1,-3],[-2,-4]))+ ncon((self.IxI,self.Y),([-1,-3],[-2,-4]))
        self.cz = ncon((self.oxo,self.I),([-1,-3],[-2,-4]))+ ncon((self.IxI,self.Z),([-1,-3],[-2,-4]))
        self.cnot = ncon((self.oxo,self.I),([-1,-3],[-2,-4]))+ ncon((self.IxI,self.X),([-1,-3],[-2,-4]))
        self.cu1  = ncon((self.oxo,self.I),([-1,-3],[-2,-4]))+ ncon((self.IxI,self.U1),([-1,-3],[-2,-4]))

        self.single_qubit =[self.H, self.Phase, self.T, self.U1]
        self.two_qubit = [self.cnot, self.cz, self.cy, self.cu1]


        if POVM=='4Pauli':
            self.K = 4;

            self.M = np.zeros((self.K,2,2),dtype=complex);

            self.M[0,:,:] = 1.0/3.0*np.array([[1, 0],[0, 0]])
            self.M[1,:,:] = 1.0/6.0*np.array([[1, 1],[1, 1]])
            self.M[2,:,:] = 1.0/6.0*np.array([[1, -1j],[1j, 1]])
            self.M[3,:,:] = 1.0/3.0*(np.array([[0, 0],[0, 1]]) + \
                                     0.5*np.array([[1, -1],[-1, 1]]) \
                                   + 0.5*np.array([[1, 1j],[-1j, 1]]) )

        if POVM=='Tetra': ## symmetric
            self.K=4;

            self.M=np.zeros((self.K,2,2),dtype=complex);

            self.v1=np.array([0, 0, 1.0]);
            self.M[0,:,:]=1.0/4.0*( self.I + self.v1[0]*self.s1+self.v1[1]*self.s2+self.v1[2]*self.s3);

            self.v2=np.array([2.0*np.sqrt(2.0)/3.0, 0.0, -1.0/3.0 ]);
            self.M[1,:,:]=1.0/4.0*( self.I + self.v2[0]*self.s1+self.v2[1]*self.s2+self.v2[2]*self.s3);

            self.v3=np.array([-np.sqrt(2.0)/3.0 ,np.sqrt(2.0/3.0), -1.0/3.0 ]);
            self.M[2,:,:]=1.0/4.0*( self.I + self.v3[0]*self.s1+self.v3[1]*self.s2+self.v3[2]*self.s3);

            self.v4=np.array([-np.sqrt(2.0)/3.0, -np.sqrt(2.0/3.0), -1.0/3.0 ]);
            self.M[3,:,:]=1.0/4.0*( self.I + self.v4[0]*self.s1+self.v4[1]*self.s2+self.v4[2]*self.s3);

        elif POVM=='Trine':
            self.K=3;
            self.M=np.zeros((self.K,2,2),dtype=complex);
            phi0=0.0
            for k in range(self.K):
                phi =  phi0+ (k)*2*np.pi/3.0
                self.M[k,:,:]=0.5*( self.I + np.cos(phi)*self.Z + np.sin(phi)*self.X)*2/3.0



        #% T matrix and its inverse
        self.t = ncon((self.M,self.M),([-1,1,2],[ -2,2,1]));
        self.it = np.linalg.inv(self.t);
        # dual frame of M
        self.Nt = ncon((self.it,self.M),([-1,1],[1,-2,-3]))
        # Nqubit tensor product of frame Mn and dual frame Ntn
        self.Ntn = self.Nt.copy()
        self.Mn = self.M.copy()
        for i in range(self.N-1):
          self.Ntn = ncon((self.Ntn, self.Nt),([-1,-3,-5],[-2,-4,-6]))
          self.Mn = ncon((self.Mn, self.M),([-1,-3,-5],[-2,-4,-6]))
          self.Ntn = np.reshape(self.Ntn, (4**(i+2),2**(i+2),2**(i+2)))
          self.Mn = np.reshape(self.Mn, (4**(i+2),2**(i+2),2**(i+2)))


        # Tensor for expectation value
        self.Trsx  = np.zeros((self.N,self.K),dtype=complex);
        self.Trsy  = np.zeros((self.N,self.K),dtype=complex);
        self.Trsz  = np.zeros((self.N,self.K),dtype=complex);
        self.Trrho = np.zeros((self.N,self.K),dtype=complex);
        self.Trrho2 = np.zeros((self.N,self.K,self.K),dtype=complex);
        self.T2 = np.zeros((self.N,self.K,self.K),dtype=complex);

        # probability gate set single qubit
        self.p_single_qubit = []
        self.p_single_qubit2 = []
        for i in range(len(self.single_qubit)):
            mat = ncon((self.M,self.single_qubit[i],self.M,self.it,np.transpose(np.conj(self.single_qubit[i]))),([-1,4,1],[1,2],[3,2,5],[3,-2],[5,4]))
            mat2 = ncon((self.M,self.single_qubit[i],self.Nt,np.transpose(np.conj(self.single_qubit[i]))),([-1,4,1],[1,2],[-2,2,5],[5,4]))
            self.p_single_qubit.append(mat)
            self.p_single_qubit2.append(mat2)

        # probability gate set two qubit
        self.p_two_qubit = []
        self.p_two_qubit2 = []
        for i in range(len(self.two_qubit)):
            mat = ncon((self.M,self.M,self.two_qubit[i],self.M,self.M,self.it,self.it,np.conj(self.two_qubit[i])),([-1,9,1],[-2,10,2],[1,2,3,4],[5,3,7],[6,4,8],[5,-3],[6,-4],[9,10,7,8]))
            mat2 = ncon((self.M,self.M,self.two_qubit[i],self.Nt,self.Nt,np.conj(self.two_qubit[i])),([-1,9,1],[-2,10,2],[1,2,3,4],[-3,3,7],[-4,4,8],[9,10,7,8]))
            self.p_two_qubit.append(mat)
            self.p_two_qubit2.append(mat2)
            #print(np.real(np.sum(np.reshape(self.p_two_qubit[i],(16,16)),1)),np.real(np.sum(np.reshape(self.p_two_qubit[i],(16,16)),0)))

        # set initial wavefunction
        if initial_state=='0':
            self.s = np.array([1,0])
        elif initial_state=='1':
            self.s = np.array([0,1])
        elif initial_state=='+':
            self.s = (1.0/np.sqrt(2.0))*np.array([1,1])
        elif initial_state=='-':
            self.s = (1.0/np.sqrt(2.0))*np.array([1,-1])
        elif initial_state=='r':
            self.s = (1.0/np.sqrt(2.0))*np.array([1,1j])
        elif initial_state=='l':
            self.s = (1.0/np.sqrt(2.0))*np.array([1,-1j])

        self.psi = self.s.copy()
        for i in range(self.N-1):
          self.psi = np.kron(self.psi, self.s)

        # set Hamiltonian
        self.ham = np.zeros((2**self.N,2**self.N), dtype=complex)
        l_basis = basis(self.N)
        for i in range(2**self.N):
          for j in range(self.N-1):
            self.ham[i, i] += - self.Jz *l_basis[i, j] * l_basis[i, j+1]
            hop_basis = l_basis[i,:].copy()
            hop_basis[j] =  int(abs(1-hop_basis[j]))
            i_hop = index(hop_basis)
            self.ham[i, i_hop] = -self.hx * 0.5
          hop_basis = l_basis[i,:].copy()
          hop_basis[self.N-1] =  int(abs(1-hop_basis[self.N-1]))
          i_hop = index(hop_basis)
          self.ham[i, i_hop] = -self.hx * 0.5



        # time evolution gate
        self.hl = self.Jz*np.kron(self.Z,self.Z) #+ self.Jz*np.kron(self.Y,self.Y)

        self.hl = self.hl +  self.hx*0.5*(np.kron(self.X,self.I)+np.kron(self.I,self.X))
        self.hl = -np.reshape(self.hl,(2,2,2,2))

        self.sx = np.reshape(0.5*np.kron(self.X,self.I)+ 0.5*np.kron(self.I,self.X),(2,2,2,2))

        self.exp_hl = np.reshape(-self.eps*self.hl,(4,4))
        self.exp_hl = expm(self.exp_hl)

        self.mat = np.reshape(self.exp_hl,(2,2,2,2))


        self.Up = ncon((self.M,self.M,self.mat,self.M,self.M,self.it,self.it,np.conj(self.mat)),([-1,9,1],[-2,10,2],[1,2,3,4],[5,3,7],[6,4,8],[5,-3],[6,-4],[9,10,7,8]))

        #self.Up = np.reshape(np.transpose(np.reshape(self.Up,(self.K**2,self.K**2))),(self.K,self.K,self.K,self.K))

        self.hlp = ncon((self.it,self.M,self.hl,self.it,self.M),([1,-1],[1,3,2],[2,5,3,6],[4,-2],[4,6,5]))
        self.sxp = ncon((self.it,self.M,self.sx,self.it,self.M),([1,-1],[1,3,2],[2,5,3,6],[4,-2],[4,6,5]))


    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


    def getinitialbias(self,initial_state):

        self.P = np.real(ncon((self.M,self.s,np.conj(self.s)),([-1,1,2],[1],[2])))

        # solving for bias
        self.bias = np.zeros(self.K)
        self.bias = np.log(self.P)

        if np.sum(np.abs(self.softmax(self.bias)-self.P))>0.00000000001:
           print("initial bias not found")
        else:
           return self.bias


    def ham_eigh(self):
      w, v = np.linalg.eigh(self.ham)
      ind = np.argmin(w)
      E = w[ind]
      psi_g = v[:, ind]
      return psi_g, E




