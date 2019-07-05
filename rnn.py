
import tensorflow as tf
import numpy as np
import os
import itertools as it
import sys
from POVM import POVM
class Circuit():
    def __init__(self, povm='Tetra', Number_qubits=6 , latent_rep_size=100, gru_hidden=100,decoder='TimeDistributed_mol', Nsamples=0, init_state='0'):

        """ decoder='TimeDistributed'   is the simplest time distributed decoder feeding z z z as input to the decoder
        at every time step
        decoder='TimeDistributed_mol' during training, input of the decoder is concatenating [z,symbol_t]; during
        generation input should be [z,predicted_symbol_t]
        # where predicted molecule is a sample of the probability vector (or a greedy sample where we convert
        predicted_molecule_t to one hot vector of the most likely symbol )
        vae=0 autoencoder vae=1 variational autoencoder
        """
        if not os.path.exists("models"): # saving the models as for checkpointing
            os.makedirs("models")

        self.batchsize = Nsamples
        self.latent_rep_size = latent_rep_size
        self.gru_hidden = gru_hidden
        
        self.max_length = Number_qubits  # self.data_train.shape[1]  size of the molecule representation

        # initializing POVM
        self.povm = POVM(POVM=povm, Number_qubits=self.max_length) 
        self.charset_length = self.povm.K 
        self.K = self.povm.K

        self.Ngru = 2 # number of stacked GRU


        # initial state/probability distribution
        # initial Weight matrix soft max layer. Initialize to zero to decouple the rnn from the dense layer 
        self.initW = tf.constant_initializer(np.zeros((self.gru_hidden,self.charset_length)))

        # initial bias softmax layer. initialize such that we get the desired product probability
        self.bias = self.povm.getinitialbias(init_state) 
        self.initBias = tf.constant_initializer(self.bias)

        self.POVM_meas = tf.placeholder(tf.float32, [None, self.max_length*self.charset_length])
        self.molecules = tf.reshape(self.POVM_meas, [-1, self.max_length, self.charset_length])

        self.generated_molecules,_ = self.generation(decoder, self.molecules,reu=None)  # molecules
        #self.logP_a = tf.log(1e-10 + self.generated_molecules)

        self.y0 = tf.placeholder(tf.float32, [None, self.charset_length])
        self.sample_onehot, self.logP = self.generation(decoder, molecules=None,y0=self.y0) 
        self.sample_RNN = tf.argmax(self.sample_onehot, axis=2) 
        
      
#        self.flipped_1qubit = tf.placeholder(tf.float32, [self.batchsize*self.K**1, self.max_length*self.charset_length])


        #  update
        self.gtype = tf.placeholder(tf.int64) 
        f = tf.cond(tf.equal(self.gtype,1), lambda: self.K, lambda: self.K**2) 
        self.flip_2 = tf.placeholder(tf.float32, [None, self.max_length*self.charset_length])
        self.f2 = tf.reshape(self.flip_2,[-1, self.max_length, self.charset_length])
        self.co_2 = tf.placeholder(tf.float32, [None])
        self.probs,_ = self.generation(decoder, self.f2,y0=None, reu=True)
        self.V = tf.reduce_sum(self.f2 * tf.log(1e-10 + self.probs ), [1, 2])
        self.cost = -tf.cast(f,tf.float32)*tf.reduce_mean(self.co_2 * self.V)
 
        tf.summary.scalar('Cost_function', self.cost)
   
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            # self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost) # optimizer
            self.opt = tf.train.AdamOptimizer(0.001)
            self.gradients, self.variables = zip(*self.opt.compute_gradients(self.cost))
            for grad, var in zip(self.gradients,self.variables):
                tf.summary.histogram(var.name + '/gradient', grad)

            self.gradients, _ = tf.clip_by_global_norm(self.gradients, 1.0)
            self.optimizer = self.opt.apply_gradients(zip(self.gradients, self.variables))
   

    def generation(self,decoder, molecules=None,y0=None, reu=True):
                 
        if decoder == 'TimeDistributed_mol':
            if molecules == None:
                b_size = tf.shape(y0)[0] 
                with tf.variable_scope("generation", reuse=tf.AUTO_REUSE):
                    zt = tf.get_variable('z', shape=[1, self.latent_rep_size])
                    z = tf.tile(zt, [b_size, 1])
                    # fully connected layer applied to latent vector z
                    z_matrix = tf.layers.dense(inputs=z, units=self.latent_rep_size, activation=tf.nn.relu,name="fc_GEN")
#
                # An atom of zeros. This is part of the (t=0) input of the RNN: input is concat([zmol, z_matrix]);
                y0 = tf.zeros([b_size, self.charset_length], dtype=tf.float32)

                logP = tf.zeros([b_size], dtype=tf.float32) 

                # concatenates z_matrix to the atom of zeros as the first input of the RNN
                y_z = tf.concat([y0, z_matrix], axis=1)

                # initial state of the RNN during the unrolling

                s0 = tf.zeros([self.Ngru,b_size,2*self.gru_hidden], dtype=tf.float32)
                 
                #output tensor unrolling
                h2 = tf.zeros([b_size, 1, self.charset_length], dtype=tf.float32)

                i0 = tf.constant(0)  # counter

                # preparing the unrolling loop
                # condition
                time_steps = self.max_length

                def c(i, s0, h2, y_z,logP):
                    return i < time_steps
#
                # body

                def b(i, s0, h2, y_z,logP):
                    with tf.variable_scope("generation", reuse=True):
                        # GRU Recurrent NN
                        with tf.variable_scope("rnn"):
                            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.gru_hidden,state_is_tuple=False) for _ in range(self.Ngru)])  # stacking 3 GRUs
                            #s0 = tuple(s0[i] for i in range(self.Ngru))
                            #s0 = tf.tuple(tf.map_fn(lambda x: x , s0))
                            s0 = tuple(tf.unstack(s0) )
                            
                            outputs,s0 = cell(y_z, s0)
                            s0 = tf.convert_to_tensor(s0)     
                            
                        dlayer = tf.layers.dense(outputs, kernel_initializer=self.initW, bias_initializer=self.initBias,units=self.charset_length, activation=tf.nn.softmax, name="output")


                        ## samples dlayer probabilities
                        logits  = tf.log(dlayer)
                         
                        samples = tf.reshape(tf.multinomial( logits,1),[-1,]) # reshape to a vector of shape=(self.batchsize,)l  multinomial returns one integer sample per element in the batch
                        dlayer  = tf.one_hot(samples,depth=self.charset_length,axis=1, dtype=tf.float32) # onehot symbol
                        
                        logP = logP + tf.reduce_sum(dlayer*logits,[1])
  
                    return [i+1,s0,tf.concat([h2,tf.reshape(dlayer,[b_size,1, self.charset_length])],axis=1),tf.concat([dlayer, z_matrix],axis=1),logP]

                # unrolling using tf.while_loop
                ii, s0, h2, y_z,logP = tf.while_loop(
                        c, b, loop_vars=[i0, s0, h2, y_z,logP],
                        shape_invariants=[i0.get_shape(), s0.get_shape(), tf.TensorShape(
                            [None, None, self.charset_length]), y_z.get_shape(),logP.get_shape()])  # shape invariants required
                # since h2 increases size as rnn unrolls

                h2=tf.slice(h2, [0, 1, 0], [-1, -1, -1])  # cuts the initial zeros that inserted to start the while_loop
                
            else:
                 
                with tf.variable_scope("generation", reuse=reu):

                    # #fully connected layer applied to latent vector z
                    b_size = tf.shape(molecules)[0]     
                    zt =  tf.get_variable('z', shape=[1, self.latent_rep_size])
                    z = tf.tile(zt, [b_size, 1])
  
                    z_matrix = tf.layers.dense(inputs=z, units=self.latent_rep_size, activation=tf.nn.relu, name="fc_GEN")

                    # An atom of zeros. This is part of the (t=0) input of the RNN: input is concat([zmol, z_matrix]); inputs at time t=1,2,.. maxlength-1  are concat([molecule[:,t,:], z_matrix])
                    zmol = tf.zeros([b_size, 1, self.charset_length], dtype=tf.float32)

                    # creates the new "molecule" input, i.e., zmol,mol_1,mol_2,...,mol_{maxlength-1}
                    mol = tf.slice(tf.concat([zmol, molecules], axis=1), [0, 0, 0], [b_size, self.max_length,
                                                                                     self.charset_length])  # creates

                    # concatenates z_matrix to all the different parts of the molecule to create the extended input of
                    h1 = tf.stack([tf.concat([t, z_matrix], axis=1) for t in tf.unstack(mol, axis=1)], axis=1)

                    # GRU Recurrent NN
                    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.gru_hidden,state_is_tuple=False) for _ in range(self.Ngru)])  # stacking 3 GRUs
   
                    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=h1, dtype=tf.float32)  # cells to the input
                    # applying a fully connected layer with a softmax non linearity on top to each time output separately
                    in_dense = tf.reshape(outputs, [b_size*self.max_length, self.gru_hidden])
                    dlayer = tf.layers.dense(in_dense, kernel_initializer=self.initW, bias_initializer=self.initBias,units=self.charset_length, activation=tf.nn.softmax, name="output")
                    #dlayer = tf.layers.dense(in_dense, units=self.charset_length, activation=tf.nn.softmax, name="output") 
                    #print(dlayer)
  
                    h2 = tf.reshape(dlayer, [-1, self.max_length, self.charset_length])
                    logP = tf.zeros([b_size], dtype=tf.float32)
#
#                     # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generation'):
#                     # print i
#
#
        return h2,logP
#
#
                     
            
            

