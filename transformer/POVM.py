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
    def __init__(self, POVM='4Pauli',Number_qubits=4,initial_state='+',Jz=1.0,hx=1.0,eps=1e-4):

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

        if POVM=='Tetra_pos':
            self.K=4;
            self.M=np.zeros((self.K,2,2),dtype=complex);

            self.v1=np.array([1.0, 1.0, 1.0])/np.sqrt(3);
            self.M[0,:,:]=1.0/4.0*( self.I + self.v1[0]*self.s1+self.v1[1]*self.s2+self.v1[2]*self.s3);

            self.v2=np.array([1.0, -1.0, -1.0])/np.sqrt(3);
            self.M[1,:,:]=1.0/4.0*( self.I + self.v2[0]*self.s1+self.v2[1]*self.s2+self.v2[2]*self.s3);

            self.v3=np.array([-1.0, 1.0, -1.0])/np.sqrt(3);
            self.M[2,:,:]=1.0/4.0*( self.I + self.v3[0]*self.s1+self.v3[1]*self.s2+self.v3[2]*self.s3);

            self.v4=np.array([-1.0, -1.0, 1.0])/np.sqrt(3);
            self.M[3,:,:]=1.0/4.0*( self.I + self.v4[0]*self.s1+self.v4[1]*self.s2+self.v4[2]*self.s3);

        elif POVM=='Trine':
            self.K=3;
            self.M=np.zeros((self.K,2,2),dtype=complex);
            phi0=0.0
            for k in range(self.K):
                phi =  phi0+ (k)*2*np.pi/3.0
                self.M[k,:,:]=0.5*( self.I + np.cos(phi)*self.Z + np.sin(phi)*self.X)*2/3.0



        #% T matrix and its inverse
        self.t = ncon((self.M,self.M),([-1,1,2],[ -2,2,1])).real;
        self.it = np.linalg.inv(self.t);
        # dual frame of M
        self.Nt = ncon((self.it,self.M),([-1,1],[1,-2,-3]))


        # Tensor for expectation value
        self.Trsx  = np.zeros((self.N,self.K),dtype=complex);
        self.Trsy  = np.zeros((self.N,self.K),dtype=complex);
        self.Trsz  = np.zeros((self.N,self.K),dtype=complex);
        self.Trrho = np.zeros((self.N,self.K),dtype=complex);
        self.Trrho2 = np.zeros((self.N,self.K,self.K),dtype=complex);
        self.T2 = np.zeros((self.N,self.K,self.K),dtype=complex);

        # probability gate set single qubit
        self.p_single_qubit = []
        for i in range(len(self.single_qubit)):
          #mat = ncon((self.M,self.single_qubit[i],self.M,self.it,np.transpose(np.conj(self.single_qubit[i]))),([-1,4,1],[1,2],[3,2,5],[3,-2],[5,4]))
          mat = self.one_body_gate(self.single_qubit[i])
          self.p_single_qubit.append(mat)

        # probability gate set two qubit
        self.p_two_qubit = []
        for i in range(len(self.two_qubit)):
          #mat = ncon((self.M,self.M,self.two_qubit[i],self.M,self.M,self.it,self.it,np.conj(self.two_qubit[i])),([-1,9,1],[-2,10,2],[1,2,3,4],[5,3,7],[6,4,8],[5,-3],[6,-4],[9,10,7,8]))
          mat = self.two_body_gate(self.two_qubit[i])
          self.p_two_qubit.append(mat)
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



        # time evolution gate

        self.hl = self.Jz*np.kron(self.Z,self.Z) +  self.hx*np.kron(self.X,self.I)
        self.hl = -np.reshape(self.hl,(2,2,2,2))
        self.hlx = self.Jz*np.kron(self.Z,self.Z) +  self.hx*(np.kron(self.X,self.I)+np.kron(self.I,self.X))
        self.hlx = -np.reshape(self.hlx,(2,2,2,2))

        #self.sx = np.reshape(np.kron(self.X,self.I)+ np.kron(self.I,self.X),(2,2,2,2))
        self.sx = np.reshape(np.kron(self.X,self.I),(2,2,2,2))

        self.exp_hl = np.reshape(-self.eps*self.hl,(4,4))
        self.exp_hl = expm(self.exp_hl)
        self.exp_hl_norm = np.linalg.norm(self.exp_hl)
        self.exp_hl2 = self.exp_hl / self.exp_hl_norm

        self.mat = np.reshape(self.exp_hl,(2,2,2,2))
        self.mat2 = np.reshape(self.exp_hl2,(2,2,2,2))

        self.Up = self.two_body_gate(self.mat)
        self.Up2 = self.two_body_gate(self.mat2)

        #self.hlp = ncon((self.it,self.M,self.hl,self.it,self.M),([1,-1],[1,3,2],[2,5,3,6],[4,-2],[4,6,5])).real
        #self.sxp = ncon((self.it,self.M,self.sx,self.it,self.M),([1,-1],[1,3,2],[2,5,3,6],[4,-2],[4,6,5])).real

        # Hamiltonian observable list
        self.hl_ob = ncon((self.hl,self.Nt,self.Nt), ([1,2,3,4],[-1,3,1],[-2,4,2])).real.astype(np.float32)
        self.hlx_ob = ncon((self.hlx,self.Nt,self.Nt), ([1,2,3,4],[-1,3,1],[-2,4,2])).real.astype(np.float32)
        self.x_ob = ncon((-self.hx*self.X,self.Nt), ([1,2],[-1,2,1])).real.astype(np.float32)

        # commuting and anti_computing operator
        hl_Nt = ncon((self.hl,self.Nt,self.Nt),([-1,-2,1,2],[-3,1,-5],[-4,2,-6]))
        Nt_hl = ncon((self.Nt,self.Nt, self.hl),([-3,-1,1],[-4,-2,2],[1,2,-5,-6]))
        hlx_Nt = ncon((self.hlx,self.Nt,self.Nt),([-1,-2,1,2],[-3,1,-5],[-4,2,-6]))
        Nt_hlx = ncon((self.Nt,self.Nt, self.hlx),([-3,-1,1],[-4,-2,2],[1,2,-5,-6]))
        x_Nt = ncon((-self.hx*self.X,self.Nt),([-1,1],[-2,1,-3]))
        Nt_x = ncon((self.Nt, -self.hx*self.X),([-2,-1,1],[1,-3]))
        self.hl_com = ncon((hl_Nt+Nt_hl,self.M,self.M),([1,2,-3,-4,3,4],[-1,3,1],[-2,4,2])).real.astype(np.float32)
        self.hl_anti = ncon((hl_Nt-Nt_hl,self.M,self.M),([1,2,-3,-4,3,4],[-1,3,1],[-2,4,2])).imag.astype(np.float32)
        self.hlx_com = ncon((hlx_Nt+Nt_hlx,self.M,self.M),([1,2,-3,-4,3,4],[-1,3,1],[-2,4,2])).real.astype(np.float32)
        self.hlx_anti = ncon((hlx_Nt-Nt_hlx,self.M,self.M),([1,2,-3,-4,3,4],[-1,3,1],[-2,4,2])).imag.astype(np.float32)
        self.x_com = ncon((x_Nt+Nt_x,self.M),([1,-2,2],[-1,2,1])).real.astype(np.float32)
        self.x_anti = ncon((x_Nt-Nt_x,self.M),([1,-2,2],[-1,2,1])).imag.astype(np.float32)


        # MPO H
        self.Ham = []

        mat = np.zeros((3,3,2,2))
        mat[0,0] = self.I
        mat[1,0] = -self.Z
        mat[2,0] = -self.X*self.hx
        mat[2,1] = self.Z
        mat[2,2] = self.I

        self.Ham.append(mat[2])
        for i in range(1,self.N-1):
            self.Ham.append(mat)
        self.Ham.append(mat[:,0,:,:])

        # MPS for Hamiltonian in probability space
        self.Hp = []
        mat = ncon((self.Ham[0],self.M,self.it),([-2,3,1],[2,1,3],[2,-1]))
        self.Hp.append(mat)
        for i in range(1,self.N-1):
            mat = ncon((self.Ham[i],self.M,self.it),([-1,-3,3,1],[2,1,3],[2,-2]))
            self.Hp.append(mat)
        mat = ncon((self.Ham[self.N-1],self.M,self.it),([-1,3,1],[2,1,3],[2,-2]))
        self.Hp.append(mat)



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

    def construct_psi(self):
      # initial wavefunction
      self.psi = self.s.copy()
      for i in range(self.N-1):
        self.psi = np.kron(self.psi, self.s)


    def construct_ham(self):
      # set Hamiltonian
      self.ham = np.zeros((2**self.N,2**self.N), dtype=complex)
      l_basis = basis(self.N)
      for i in range(2**self.N):
        for j in range(self.N-1):
          self.ham[i, i] += - self.Jz *(4.0*l_basis[i, j] * l_basis[i, j+1] - 2.0*l_basis[i,j]- 2.0*l_basis[i,j+1] +1. )
          hop_basis = l_basis[i,:].copy()
          hop_basis[j] =  int(abs(1-hop_basis[j]))
          i_hop = index(hop_basis)
          self.ham[i, i_hop] = -self.hx
        hop_basis = l_basis[i,:].copy()
        hop_basis[self.N-1] =  int(abs(1-hop_basis[self.N-1]))
        i_hop = index(hop_basis)
        self.ham[i, i_hop] = -self.hx

    def ham_eigh(self):
      w, v = np.linalg.eigh(self.ham)
      ind = np.argmin(w)
      E = w[ind]
      psi_g = v[:, ind]
      return psi_g, E


    def one_body_gate(self, gate, mask=True):
      g1 = ncon((self.M, gate, self.Nt,np.transpose(np.conj(gate))),([-1,4,1],[1,2],[-2,2,5],[5,4]))
      if mask:
        g1_mask = np.abs(g1.real) > 1e-15
        g1 = np.multiply(g1, g1_mask)
      return g1.real.astype('float32')


    def two_body_gate(self, gate, mask=True):
      g2 = ncon((self.M,self.M, gate,self.Nt,self.Nt,np.conj(gate)),([-1,9,1],[-2,10,2],[1,2,3,4],[-3,3,7],[-4,4,8],[9,10,7,8]))
      if mask:
        g2_mask = np.abs(g2.real) > 1e-15
        g2 = np.multiply(g2, g2_mask)
      return g2.real.astype('float32')


    def P_gate(self, gate, mask=True):
      gate_factor = int(gate.ndim/2)
      if gate_factor == 1:
        g = ncon((self.M, gate, self.Nt,np.transpose(np.conj(gate))),([-1,4,1],[1,2],[-2,2,5],[5,4]))
      else:
        g = ncon((self.M,self.M, gate,self.Nt,self.Nt,np.conj(gate)),([-1,9,1],[-2,10,2],[1,2,3,4],[-3,3,7],[-4,4,8],[9,10,7,8]))
      if mask:
        g_mask = np.abs(g.real) > 1e-15
        g = np.multiply(g, g_mask)
      return g.real.astype('float32')


    def kron_gate(self, gate, site, Nqubit):

      gate_factor = int(gate.ndim /2)
      g = gate.copy()
      if gate_factor == 2:
        g = np.reshape(g, (4,4))

      if site != 0:
        I_L = np.eye(2)
        for i in range(site-1):
          I_L = np.kron(I_L, np.eye(2))
      else:
        I_L = 1.

      if site != Nqubit - gate_factor:
        I_R = np.eye(2)
        for i in range(Nqubit-site-gate_factor-1):
          I_R = np.kron(I_R, np.eye(2))
      else:
        I_R = 1.

      g = np.kron(I_L, g)
      g = np.kron(g, I_R)
      return g

    def construct_Nframes(self):
      # Nqubit tensor product of frame Mn and dual frame Ntn
      self.Ntn = self.Nt.copy()
      self.Mn = self.M.copy()
      for i in range(self.N-1):
        self.Ntn = ncon((self.Ntn, self.Nt),([-1,-3,-5],[-2,-4,-6]))
        self.Mn = ncon((self.Mn, self.M),([-1,-3,-5],[-2,-4,-6]))
        self.Ntn = np.reshape(self.Ntn, (4**(i+2),2**(i+2),2**(i+2)))
        self.Mn = np.reshape(self.Mn, (4**(i+2),2**(i+2),2**(i+2)))
