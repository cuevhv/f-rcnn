from config import cfg


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

class netvgg:
    def __init__(self, is_training):
        self.inputs = []
        self.is_training = is_training
    def _conv_layers(self, inputs):
        self.inputs = inputs
        self.inputs = tf.cast(self.inputs, tf.float32)
        self.inputs = ((self.inputs / 255.0) -0.5)*2

        with tf.variable_scope("vgg_16"):
            with slim.arg_scope(vgg.vgg_arg_scope()):
                self.net = self.inputs
                self.net = slim.repeat(self.net, 2, slim.conv2d, 64, [3, 3], scope = 'conv1')
                self.net = slim.max_pool2d(self.net, [2, 2], scope = 'pool1')
                self.net = slim.repeat(self.net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                self.net = slim.max_pool2d(self.net, [2, 2], scope = 'pool2')
                self.net = slim.repeat(self.net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                self.net = slim.max_pool2d(self.net, [2, 2], scope = 'pool3')
                self.net = slim.repeat(self.net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                self.net = slim.max_pool2d(self.net, [2, 2], scope = 'pool4')
                self.net = slim.repeat(self.net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            if cfg.MODEL.POOL5 == True:
                self.net = slim.max_pool2d(self.net, [2, 2], scope = 'pool5')
        with tf.variable_scope("vgg_16"):
            if cfg.MODEL.VGG16_FC == True:
                self.net = slim.flatten(self.net)
                w_init = tf.contrib.layers.xavier_initializer()
                w_reg = slim.l2_regularizer(0.0005)
                self.net = slim.fully_connected(self.net, 4096,
                                           weights_initializer = w_init,
                                           weights_regularizer = w_reg,
                                           scope = 'fc6')
                self.net = slim.dropout(self.net, keep_prob = 0.5, is_training = self.is_training)
                self.net = slim.fully_connected(self.net, 4096,
                                           weights_initializer = w_init,
                                           weights_regularizer = w_reg,
                                           scope = 'fc7')
                self.net = slim.dropout(self.net, keep_prob = 0.5, is_training = self.is_training)
                self.net = slim.fully_connected(self.net, 1000,
                                           weights_initializer = w_init,
                                           weights_regularizer = w_reg,
                                           scope = 'fc8')
        print("SHAPE!!!!", self.net)
        return self.net

    def load_weights(self, sess):
        print "Loading weights"
        path = pathlib.Path('./../../weights/vgg16_weights.npz')
        if(path.is_file()):
            print("it's a mattafaca")
            init_weights = np.load(path)
            #print type(init_weights)
            #print(init_weights.files)
            #print(tf.all_variables())
            if cfg.MODEL.VGG16_FC == True:

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
            else:
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
                                                                      })
            sess.run(assign_op, feed_dict_init)
        print("weights_loaded")

    def losses(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return loss

    def optimize(self, losses):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(lr, global_step,
                                                 num_iter*decay_per, decay_rate, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(losses, global_step=global_step)#,
                    #var_list=slim.get_model_variables("finetune"))
        return train_op
    def run_example(self):

        tf.reset_default_graph()
        im_width=224
        im_height=224
        #print "good till here1"
        im_placeholder = tf.placeholder(tf.uint8, [None, im_height, im_width, 3])
        logits = self._conv_layers(im_placeholder)
        prediction = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(prediction, 1)
        #print "good till here2"

        img = Image.open('../data/cat1.jpg')
        img = np.array(img.resize((im_width,im_height), Image.ANTIALIAS))

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(tf.local_variables_initializer())
            #cfg_training = False
            #if cfg_training == False:
            if cfg.MODEL.TRAIN == False:
                self.load_weights(sess)
            pred_lbl, proba = sess.run([predicted_labels, prediction], feed_dict={im_placeholder:np.expand_dims(img, axis=0)})
            print pred_lbl
            if pred_lbl[0] == 999:
                print("CAT")
                img = Image.fromarray(img, 'RGB')
                img.show()
