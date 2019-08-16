

# reverse samples for a sequence of gates ob TFIM ham
# TODO: change to append version and then reshape
def reverse_samples_ham2(Ndataset, batch_size, Nqubit, target_vocab_size, hl, hlx, tau, ansatz):

    ## it ensures at least one batch size samples, since Ncall can be zero
    Ncalls = Ndataset /batch_size
    samples,lP = ansatz.sample(batch_size) # get samples from the model

    #samples = tf.constant(np.array(list(it.product(range(4), repeat = Nqubit)),dtype=np.uint8))
    #lP = ansatz(samples)

    lP = np.reshape(lP,[-1,1]) ## necessary for concatenate
    update_Pi = tf.zeros([batch_size,], tf.float32)


    flip,co = flip2_reverse_tf(samples,hlx,target_vocab_size,site=[Nqubit-2, Nqubit-1])
    #flip,co = flip2_reverse_swift(samples,hlx,target_vocab_size,site=[Nqubit-2, Nqubit-1])
    flip = tf.cast(flip, dtype=tf.uint8) # c are configurations
    Pj = tf.exp(ansatz(flip))
    co_Pj = tf.reshape(co*Pj,(batch_size, 16))
    co_Pj_sum = tf.reduce_sum(co_Pj, axis=1)
    update_Pi += co_Pj_sum
    for i in range(Nqubit-2):
        flip,co = flip2_reverse_tf(samples,hl,target_vocab_size,site=[i,i+1])
        #flip,co = flip2_reverse_swift(samples,hl,target_vocab_size,site=[i,i+1])
        flip = tf.cast(flip, dtype=tf.uint8) # c are configurations
        Pj = tf.exp(ansatz(flip))
        co_Pj = tf.reshape(co*Pj,(batch_size, 16))
        co_Pj_sum = tf.reduce_sum(co_Pj, axis=1)
        update_Pi += co_Pj_sum

    update_Pi = np.reshape(update_Pi,[-1,1])
    #update_Pi = tau * update_Pi
    update_Pi = (tf.exp(lP) - tau * update_Pi) / tf.exp(lP)

    for k in range(int(Ncalls)):
        sa,llpp = ansatz.sample(batch_size)
        samples = np.vstack((samples,sa))
        llpp =np.reshape(llpp,[-1,1])
        lP =  np.vstack((lP,llpp))
        up_pi = tf.zeros([batch_size,], tf.float32)

        fp,coef = flip2_reverse_tf(sa,hlx,target_vocab_size,site=[Nqubit-2,Nqubit-1])
        #fp,coef = flip2_reverse_swift(sa,hlx,target_vocab_size,site=[Nqubit-2,Nqubit-1])
        fp = tf.cast(fp, dtype=tf.uint8) # c are configurations
        pj = tf.exp(ansatz(fp))
        coef_pj = tf.reshape(coef*pj,(batch_size, 16))
        coef_pj_sum = tf.reduce_sum(coef_pj, axis=1)
        up_pi += coef_pj_sum
        for i in range(Nqubit-2):
            fp,coef = flip2_reverse_tf(sa,hl,target_vocab_size,site=[i,i+1])
            #fp,co = flip2_reverse_swift(sa,hl,target_vocab_size,site=[i,i+1])
            fp = tf.cast(fp, dtype=tf.uint8) # c are configurations
            pj = tf.exp(ansatz(fp))
            coef_pj = tf.reshape(coef*pj,(batch_size, 16))
            coef_pj_sum = tf.reduce_sum(coef_pj, axis=1)
            up_pi += coef_pj_sum

        up_pi = np.reshape(up_pi,[-1,1])
        #up_pi = tau * update_pi
        up_pi = (tf.exp(llpp) - tau * up_pi) / tf.exp(llpp)
        update_Pi = np.vstack((update_Pi, up_pi))

    #update_Pi = tau * update_Pi
    #update_Pi = (tf.exp(lP) - tau * update_Pi) / tf.exp(lP)
    samples = tf.stop_gradient(samples)
    update_Pi = tf.stop_gradient(update_Pi)
    lP = tf.stop_gradient(lP)

    return (samples, lP, update_Pi)

