import tensorflow as tf
import numpy as np


a = tf.placeholder(tf.float32, [None, 3,4])
b = tf.constant([[[3, 2, 1, 0],[3, 2, 1, 0],[3, 2, 1, 0]], [[3, 2, 1, 0],[3, 2, 1, 0],[3, 2, 1, 0]]], dtype=tf.float32)
c = tf.placeholder(tf.float32, [None,3,2])

s = tf.reduce_sum(a, axis=-1, keep_dims=True)
s = tf.not_equal(s, 0)
s = tf.cast(s, tf.float32)
e = tf.multiply(c, s)


a_ = np.array([[[-1, -1, -1, 0],[0, 0, 0, 0],[0, 0, 0, 0]], [[0, 0, 0, 0],[1, 2, 1, 0],[3, 2, 1, 0]]])
c_ = np.array([[[3, 20],[3, 0],[3, 0]], [[1, 0],[3, 0],[1, 0]]])
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
#bm_s, aa_s, sum_a_s, mult_a = sess.run([bm, aa_, bull_a, mult_a],feed_dict={a_:a})
#print sum_a_s
#print "zeros_the_arrays", mult_a
#print "a", aa_s
#print bm_s
#print "condition test"
print(sess.run([s, e], feed_dict={a:a_, c:c_}))

#print("sqia", sess.run(ra))
