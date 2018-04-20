import os
import time

import pandas
import numpy as np
import skimage.io as io
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
slim = tf.contrib.slim
import pathlib

"""Keras you are my hope"""
import keras as K
import keras.layers as KL


def netvgg_conv(inputs, is_training = True):
    inputs = tf.cast(inputs, tf.float32)
    inputs = ((inputs / 255.0) -0.5)*2

    with tf.variable_scope("vgg_16"):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = inputs
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope = 'conv1')
            net = slim.max_pool2d(net, [2, 2], scope = 'pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope = 'pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope = 'pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope = 'pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope = 'pool5')
            #net3 = slim.conv2d(net, 2*num_anchors, [1, 1], scope='prob_object')
            #print(net3)
            #net2 = slim.conv2d(net, 4*num_anchors, [1, 1], scope='bbox')
            #print(net2)
        net2 = slim.flatten(net)
        w_init = tf.contrib.layers.xavier_initializer()
        w_reg = slim.l2_regularizer(0.0005)
        net2 = slim.fully_connected(net2, 4096,
                                   weights_initializer = w_init,
                                   weights_regularizer = w_reg,
                                   scope = 'fc6')
        net2 = slim.dropout(net2, keep_prob = 0.5, is_training = is_training)
        net2 = slim.fully_connected(net2, 4096,
                                   weights_initializer = w_init,
                                   weights_regularizer = w_reg,
                                   scope = 'fc7')
        net2 = slim.dropout(net2, keep_prob = 0.5, is_training = is_training)
        net2 = slim.fully_connected(net2, 1000,
                                   weights_initializer = w_init,
                                   weights_regularizer = w_reg,
                                   scope = 'fc8')
        #print("SHAPE!!!!", net2)
    return net, net2


def rp_network(vgg_conv, num_anchors, is_training = False):
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    rpn_cls_score = slim.conv2d(vgg_conv, 2*num_anchors, [1, 1], weights_initializer = initializer)
    rpn_cls_score_reshape = _reshape_layer(rpn_cls_score, 2, "rpn_cls_score_reshape")
    rpn_cls_prob_reshape  = _softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")

    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis = 1, name= "rpn_cls_pred")
    rpn_cls_prob = _reshape_layer(rpn_cls_prob_reshape, num_anchors*2, "rpn_cls_prob")
    print("RESHAPE!!!!!!!", rpn_cls_score)
    print("RESHAPE!!!!!!!", rpn_cls_score_reshape)

def _reshape_layer(bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
        # change the channel to the caffe format
        to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
        # then force it to have channel 2
        reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
        reshaped_2 = KL.Reshape(-1, )(to_caffe)
        #print("TO_CAFFE!!!!!", reshaped)
        # then swap the channel back
        to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
    return to_tf

def _softmax_layer(bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
        input_shape = tf.shape(bottom)
        bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
        reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
        return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

def losses(logits, labels):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

def optimize(losses):
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(lr, global_step,
                                             num_iter*decay_per, decay_rate, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(losses, global_step=global_step)#,
                #var_list=slim.get_model_variables("finetune"))
    return train_op


"""Data needed so far"""
im_width=224
im_height=224
anchor_scales=(8, 16, 32)
anchor_ratios=(0.5, 1, 2)
num_scales = len(anchor_scales)
num_ratios = len(anchor_ratios)

num_anchors = num_scales*num_ratios

print "good till here1"

flag = True

tf.reset_default_graph()
im_placeholder = tf.placeholder(tf.uint8, [None, im_height, im_width, 3])
vgg_transfer, logits = netvgg_conv(im_placeholder, is_training=False)
if flag == True:
    new_net = rp_network(vgg_transfer, num_anchors, is_training=False)
prediction = tf.nn.softmax(logits)
predicted_labels = tf.argmax(prediction, 1)
print "good till here2"

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    path = pathlib.Path('./../weights/vgg16_weights.npz')
    if(path.is_file()):
        print("it's a mattafaca")
        init_weights = np.load(path)
        #print type(init_weights)
        #print(init_weights.files)
        #print(tf.all_variables())
        assign_op, feed_dict_init = slim.assign_from_values({'vgg_16/conv1/conv1_1/weights' : init_weights['conv1_1_W'],
                                                             'vgg_16/conv1/conv1_1/biases' : init_weights['conv1_1_b'],
                                                             'vgg_16/conv1/conv1_2/weights' : init_weights['conv1_2_W'],
                                                             'vgg_16/conv1/conv1_2/biases' : init_weights['conv1_2_b'],
                                                             'vgg_16/conv2/conv2_1/weights' : init_weights['conv2_1_W'],
                                                             'vgg_16/conv2/conv2_1/biases' : init_weights['conv2_1_b'],
                                                             'vgg_16/conv2/conv2_2/weights' : init_weights['conv2_2_W'],
                                                             'vgg_16/conv2/conv2_2/biases' : init_weights['conv2_2_b'],
                                                             'vgg_16/conv3/conv3_1/weights' : init_weights['conv3_1_W'],
                                                             'vgg_16/conv3/conv3_1/biases' : init_weights['conv3_1_b'],
                                                             'vgg_16/conv3/conv3_2/weights' : init_weights['conv3_2_W'],
                                                             'vgg_16/conv3/conv3_2/biases' : init_weights['conv3_2_b'],
                                                             'vgg_16/conv3/conv3_3/weights' : init_weights['conv3_3_W'],
                                                             'vgg_16/conv3/conv3_3/biases' : init_weights['conv3_3_b'],
                                                             'vgg_16/conv4/conv4_1/weights' : init_weights['conv4_1_W'],
                                                             'vgg_16/conv4/conv4_1/biases' : init_weights['conv4_1_b'],
                                                             'vgg_16/conv4/conv4_2/weights' : init_weights['conv4_2_W'],
                                                             'vgg_16/conv4/conv4_2/biases' : init_weights['conv4_2_b'],
                                                             'vgg_16/conv4/conv4_3/weights' : init_weights['conv4_3_W'],
                                                             'vgg_16/conv4/conv4_3/biases' : init_weights['conv4_3_b'],
                                                             'vgg_16/conv5/conv5_1/weights' : init_weights['conv5_1_W'],
                                                             'vgg_16/conv5/conv5_1/biases' : init_weights['conv5_1_b'],
                                                             'vgg_16/conv5/conv5_2/weights' : init_weights['conv5_2_W'],
                                                             'vgg_16/conv5/conv5_2/biases' : init_weights['conv5_2_b'],
                                                             'vgg_16/conv5/conv5_3/weights' : init_weights['conv5_3_W'],
                                                             'vgg_16/conv5/conv5_3/biases' : init_weights['conv5_3_b'],
                                                             'vgg_16/fc6/weights' : init_weights['fc6_W'],
                                                             'vgg_16/fc6/biases' : init_weights['fc6_b'],
                                                             'vgg_16/fc7/weights' : init_weights['fc7_W'],
                                                             'vgg_16/fc7/biases' : init_weights['fc7_b'],
                                                             'vgg_16/fc8/weights' : init_weights['fc8_W'],
                                                             'vgg_16/fc8/biases' : init_weights['fc8_b'],
                                                             })
        #print(assign_op, feed_dict_init)
        sess.run(assign_op, feed_dict_init)
        img = Image.open('data/cat1.jpg')
        img = np.array(img.resize((im_width,im_height), Image.ANTIALIAS))
        pred_lbl, proba = sess.run([predicted_labels, prediction], feed_dict={im_placeholder:np.expand_dims(img, axis=0)})
        print(pred_lbl)
        #for k in init_weights.files:
            #print(k)

        #print init_weights['conv1_1_W']
        #assign_op, feed_dict_init = slim.assign_from_values({'conv1/weights' : init_weights['conv1_w'],})
        #sess.run(assign_op, feed_dict_init)
#with tf.Session() as sess:
#    print "good till here3"#
#    saver = tf.train.Saver()
#    print("SESS!!!!!!!!!!!!!!!!!!!!!!!!!!!", sess)
#    saver.restore(sess, './vgg_16.ckpt')
    #sess.run(tf.local_variables_initializer())
    #sess.run(tf.global_variables_initializer())

    #img = Image.open('cat02.jpg')
    #print("shape!!!!!", im.size)
    #img = np.array(img.resize((im_width,im_height), Image.ANTIALIAS))
    #prob = sess.run(prediction, feed_dict={im_placeholder:np.expand_dims(img, axis=0)})
    #print(prob[0][1])
