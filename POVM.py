import numpy as np
from ncon import ncon
#import tensorly as tl
#from tensorly.decomposition import matrix_product_state
from copy import deepcopy
from scipy.linalg import expm

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
        self.Sp = np.array([[1.0, 0.0],[0.0, -1j]]) # S-gate
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


        # Tetra POVM
                # Tetra POVM
        if POVM=='4Pauli':
            self.K = 4;

            self.M = np.zeros((self.K,2,2),dtype=complex);

            self.M[0,:,:] = 1.0/3.0*np.array([[1, 0],[0, 0]])
            self.M[1,:,:] = 1.0/6.0*np.array([[1, 1],[1, 1]])
            self.M[2,:,:] = 1.0/6.0*np.array([[1, -1j],[1j, 1]])
            self.M[3,:,:] = 1.0/3.0*(np.array([[0, 0],[0, 1]]) + \
                                     0.5*np.array([[1, -1],[-1, 1]]) \
                                   + 0.5*np.array([[1, 1j],[-1j, 1]]) )

        if POVM=='Tetra':
            # symmetric
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

        elif POVM=='z':
            self.K=2
            self.M=np.zeros((self.K,2,2),dtype=complex);
            self.M[0,0,0] = 1
            self.M[1,1,1] = 1



        #% T matrix and its inverse
        self.t = ncon((self.M,self.M),([-1,1,2],[ -2,2,1]));
        self.it = np.linalg.inv(self.t);
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
            mat = ncon((self.M,self.single_qubit[i],self.M,self.it,np.transpose(np.conj(self.single_qubit[i]))),([-1,4,1],[1,2],[3,2,5],[3,-2],[5,4]))
            self.p_single_qubit.append(mat)

        # probability gate set two qubit
        self.p_two_qubit = []
        for i in range(len(self.two_qubit)):
            mat = ncon((self.M,self.M,self.two_qubit[i],self.M,self.M,self.it,self.it,np.conj(self.two_qubit[i])),([-1,9,1],[-2,10,2],[1,2,3,4],[5,3,7],[6,4,8],[5,-3],[6,-4],[9,10,7,8]))
            self.p_two_qubit.append(mat)
            #print(np.real(np.sum(np.reshape(self.p_two_qubit[i],(16,16)),1)),np.real(np.sum(np.reshape(self.p_two_qubit[i],(16,16)),0)))


        # time evolution gate

        self.hl = self.Jz*np.kron(self.Z,self.Z) #+ self.Jz*np.kron(self.Y,self.Y)

        self.hl = self.hl +  self.hx*0.5*(np.kron(self.X,self.I)+np.kron(self.I,self.X))
        self.hl = -np.reshape(self.hl,(2,2,2,2))

        self.sx = np.reshape(0.5*np.kron(self.X,self.I)+ 0.5*np.kron(self.I,self.X),(2,2,2,2))

        self.mat = np.reshape(-self.eps*self.hl,(4,4))
        self.mat = expm(self.mat)

        self.mat = np.reshape(self.mat,(2,2,2,2))


        self.Up = ncon((self.M,self.M,self.mat,self.M,self.M,self.it,self.it,np.conj(self.mat)),([-1,9,1],[-2,10,2],[1,2,3,4],[5,3,7],[6,4,8],[5,-3],[6,-4],[9,10,7,8]))

        #self.Up = np.reshape(np.transpose(np.reshape(self.Up,(self.K**2,self.K**2))),(self.K,self.K,self.K,self.K))

        self.hlp = ncon((self.it,self.M,self.hl,self.it,self.M),([1,-1],[1,3,2],[2,5,3,6],[4,-2],[4,6,5]))
        self.sxp = ncon((self.it,self.M,self.sx,self.it,self.M),([1,-1],[1,3,2],[2,5,3,6],[4,-2],[4,6,5]))

        # time evolution MPO

        self.Ox = expm(self.eps*self.hx*self.X)
        self.z = np.zeros((2,2,2))
        self.z[0,:,:] = self.I
        self.z[1,:,:] = self.Z
        self.zz = ncon((self.z,self.z),([-1,-3,1],[-2,1,-4]))
        self.B = np.zeros((2,2),dtype=complex)
        self.B[:,0] = np.array([np.sqrt(np.cosh(self.eps*self.Jz)),0],dtype=complex)
        self.B[:,1] = np.array([0,np.sqrt(np.sinh(self.eps*self.Jz))],dtype=complex)

        self.expH = []
        temp = ncon((self.z,self.B),([1,-2,-3],[-1,1]))
        self.expH.append(temp)
        for i in range(1,self.N-1):
            temp = ncon((self.zz,self.B,self.B),([1,2,-3,-4],[-1,1],[-2,2]))
            self.expH.append(temp)

        temp = ncon((self.z,self.B),([1,-2,-3],[-1,1]))
        self.expH.append(temp)

        # multiply by operator self.Ox
        self.expH[0] = ncon((self.expH[0],self.Ox),([-1,1,-3],[-2,1]))
        self.expH[self.N-1] = ncon((self.expH[self.N-1],self.Ox),([-1,1,-3],[-2,1]))
        for i in range(1,self.N-1):
            self.expH[i] = ncon((self.expH[i],self.Ox),([-1,-2,1,-4],[-3,1]))



        #self.expH3 = ncon((self.expH[0],self.expH[1],self.expH[2]),([1,-1,-4],[1,2,-2,-5],[2,-3,-6]))
        #self.expH3 = np.reshape(self.expH3,[2**3,2**3])
        #print(op)

        # MPO for probability evolution
        self.evolP = []

        mat = ncon((self.M,self.expH[0],self.M,self.it,np.conj(self.expH[0])),([-3,4,1],[-1,1,2],[5,2,3],[5,-4],[-2,3,4]))
        mat = np.reshape(mat,(self.expH[0].shape[0]**2,self.K,self.K))
        self.evolP.append(mat)

        for i in range(1,self.N-1):
            mat = ncon((self.M,self.expH[i],self.M,self.it,np.conj(self.expH[i])),([-5,5,1],[-1,-3,1,2],[3,2,4],[3,-6],[-2,-4,4,5]))
            mat = np.reshape(mat,(self.expH[0].shape[0]**2,self.expH[0].shape[0]**2,self.K,self.K))
            self.evolP.append(mat)

        mat = ncon((self.M,self.expH[self.N-1],self.M,self.it,np.conj(self.expH[self.N-1])),([-3,4,1],[-1,1,2],[5,2,3],[5,-4],[-2,3,4]))
        mat = np.reshape(mat,(self.expH[0].shape[0]**2,self.K,self.K))
        self.evolP.append(mat)

        #self.op = ncon((self.evolP[0],self.evolP[1],self.evolP[2]),([1,-1,-4],[1,2,-2,-5],[2,-3,-6]))
        #self.op = np.reshape(self.op,[4**3,4**3])
        #print(np.sum(self.op))


        # MPO H
        self.H = []

        mat = np.zeros((3,3,2,2))
        mat[0,0] = self.I
        mat[1,0] = -self.Z
        mat[2,0] = -self.X*self.hx
        mat[2,1] = self.Z
        mat[2,2] = self.I


        self.H.append(mat[2])
        for i in range(1,self.N-1):
            self.H.append(mat)
        self.H.append(mat[:,0,:,:])

        #self.H3 = ncon((self.H[0],self.H[1],self.H[2]),([1,-1,-4],[1,2,-2,-5],[2,-3,-6]))
        #self.H3 = np.reshape(self.H3,(2**3,2**3))
        #self.H33 = -self.Jz*(np.kron(np.kron(self.Z,self.Z),self.I ) +np.kron(np.kron(self.I,self.Z),self.Z)) -self.hx*( np.kron(np.kron(self.X,self.I),self.I )+ np.kron(np.kron(self.I,self.X),self.I ) + np.kron(np.kron(self.I,self.I),self.X ))

        # MPS for Hamiltonian in probability space
        self.Hp = []
        mat = ncon((self.H[0],self.M,self.it),([-2,3,1],[2,1,3],[2,-1]))
        self.Hp.append(mat)
        for i in range(1,self.N-1):
            mat = ncon((self.H[i],self.M,self.it),([-1,-3,3,1],[2,1,3],[2,-2]))
            self.Hp.append(mat)
        mat = ncon((self.H[self.N-1],self.M,self.it),([-1,3,1],[2,1,3],[2,-2]))
        self.Hp.append(mat)



    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


    def getinitialbias(self,initial_state):
        # which initial product state?
        if initial_state=='0':
            s = np.array([1,0])
        elif initial_state=='1':
            s = np.array([0,1])
        elif initial_state=='+':
            s = (1.0/np.sqrt(2.0))*np.array([1,1])
        elif initial_state=='-':
            s = (1.0/np.sqrt(2.0))*np.array([1,-1])
        elif initial_state=='r':
            s = (1.0/np.sqrt(2.0))*np.array([1,1j])
        elif initial_state=='l':
            s = (1.0/np.sqrt(2.0))*np.array([1,-1j])
        elif initial_state=='ab':
            a = 1;b=2
            s = (1.0/np.sqrt(a**2+b**2))*np.array([a,b])

        self.P = np.real(ncon((self.M,s,np.conj(s)),([-1,1,2],[1],[2])))

        # solving for bias
        self.bias = np.zeros(self.K)
        self.bias = np.log(self.P)

        if np.sum(np.abs(self.softmax(self.bias)-self.P))>0.00000000001:
           print("initial bias not found")
        else:
           return self.bias



# TODO: two different time evolution gate



