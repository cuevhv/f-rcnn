import tensorflow as tf
import numpy as np


a_ = tf.placeholder(shape=[None,3,2], dtype=tf.float32)
b_ = tf.constant([[[3, 2, 1, 0],[3, 2, 1, 0],[3, 2, 1, 0]], [[3, 2, 1, 0],[3, 2, 1, 0],[3, 2, 1, 0]]], dtype=tf.float32)
#b_ = tf.constant([[[3, 2],[4, 5],[6, 7]], [[3, 2],[4, 5],[6, 7]]], dtype=tf.float32)
#b_ = tf.constant([[[3, 2],[4, 5],[6, 7]]], dtype=tf.float32)
sum_a = tf.reduce_sum(a_, axis=-1, keep_dims=True)
bull_a = tf.reduce_sum(a_, axis=-1)
sum_a = tf.cast(sum_a, tf.float32)

aa_ = tf.concat([sum_a, sum_a], -1)
aa_ = tf.cast(aa_, tf.bool)
#print aa
bm = tf.boolean_mask(b_, tf.cast(bull_a, tf.bool))

mult_a = tf.multiply(b_, sum_a)


#a = np.array([[[0, 1],[0,0],[1,0]]])
a = np.array([[[0, 1],[0,0],[1,0]], [[0, 0],[1,0],[0,0]]])

C = tf.Variable([[[3, 2, 1, 0],[3, 2, 1, 0],[3, 2, 1, 0]], [[3, 2, 1, 0],[3, 2, 1, 0],[3, 2, 1, 0]]], dtype=tf.float32)
D = tf.Variable([[[3, 2, 1, 0],[3, 2, 1, 0],[3, 2, 1, 0]], [[3, 2, 1, 0],[3, 2, 1, 0],[3, 2, 1, 0]]], dtype=tf.float32)
F = tf.subtract(C,D)
F = tf.Variable([[[0.5, -2, 1, 0],[0.5, 1, 1, 0],[0, 1, 1, 0]], [[0, 1, 1, 0],[0, 1, 1, 0],[0, 1, 1, 0]]], dtype=tf.float32)

comparison = tf.less(tf.abs(F), tf.constant([1,1,1,1], dtype=tf.float32))
conditional_assignment = F.assign(tf.where (comparison, tf.square(F)*0.5, tf.abs(F)-0.5))
weird_m = 0.5*tf.square(F)*tf.to_float(comparison)+(tf.abs(F)-0.5)*(1-tf.to_float(comparison))
#conditional_assignment = tf.cast(conditional_assignment, tf.float32)
x = tf.constant([[1, 1, 1], [1, 1, 1]])

rs = tf.reduce_sum(conditional_assignment, -1, keep_dims=True)#, axis=-1, keep_dims=True)
ra = tf.reduce_mean(rs)

#conditional_assignment = E.assign(tf.where (comparison, tf.ones_like(E), E))


#a= np.expand_dims(a, axis = 0)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
#bm_s, aa_s, sum_a_s, mult_a = sess.run([bm, aa_, bull_a, mult_a],feed_dict={a_:a})
#print sum_a_s
#print "zeros_the_arrays", mult_a
#print "a", aa_s
#print bm_s
#print "condition test"
print(sess.run([F, conditional_assignment, weird_m]))

#print("sqia", sess.run(ra))
