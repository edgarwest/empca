import tensorflow as tf
import numpy as np

"""
todo: make weighte version, with
"""
def uwpca(nvar, nobs, nvec):
    P_true = np.random.randn(nvar, nvec)
    C_true = np.random.randn(nvec, nobs)
    X_true = np.dot(P_true,C_true)
    assert X_true.shape ==(nvar, nobs), 'shape error'

    P = tf.Variable(np.random.randn(nvar, nvec).astype(np.float32))
    C = tf.Variable(np.random.randn(nvec, nobs).astype(np.float32))
    X_est = tf.matmul(P,C)

    loss = tf.reduce_sum((X_est-X_true)**2)

    alpha = tf.constant(1e-4)
    regP = alpha*tf.reduce_sum(P**2)
    regC = alpha*tf.reduce_sum(C**2)

    objective = loss +regP + regC

    train_step = tf.train.AdamOptimizer(0.001).minimize(objective)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for n in range(10000):
            sess.run(train_step)
            if (n+1)%1000 ==0:
                print ('iter %i, %f' %(n+1, sess.run(objective)))
uwpca(100, 80, 3)