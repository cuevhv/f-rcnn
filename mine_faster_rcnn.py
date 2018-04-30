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
import keras as K
from keras.layers.core import Activation, Reshape

import read_voc
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def netvgg(inputs, is_training = True):
    inputs = tf.cast(inputs, tf.float32)
    inputs = ((inputs / 255.0) -0.5)*2
    num_anchors = 9
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

            net_cnn = net
            net1 = slim.conv2d(net, 2*num_anchors, [1, 1], scope= "prob")

            net1 = Reshape((-1, 2), input_shape=(net1.shape[1], net1.shape[2], net1.shape[3]))(net1)
            #net1 = tf.reshape(net1, [0, -1, -1, 2])
            net2 = slim.conv2d(net, 4*num_anchors, [1, 1], scope='bbox')
            net2 = Reshape((-1, 4), input_shape=(net2.shape[1], net2.shape[2], net2.shape[3]))(net2)

            net = slim.max_pool2d(net, [2, 2], scope = 'pool5')

        net = slim.flatten(net)

        w_init = tf.contrib.layers.xavier_initializer()
        w_reg = slim.l2_regularizer(0.0005)
        net = slim.fully_connected(net, 4096,
                                   weights_initializer = w_init,
                                   weights_regularizer = w_reg,
                                   scope = 'fc6')
        net = slim.dropout(net, keep_prob = 0.5, is_training = is_training)
        net = slim.fully_connected(net, 4096,
                                   weights_initializer = w_init,
                                   weights_regularizer = w_reg,
                                   scope = 'fc7')
        net = slim.dropout(net, keep_prob = 0.5, is_training = is_training)
        net = slim.fully_connected(net, 1000,
                                   weights_initializer = w_init,
                                   weights_regularizer = w_reg,
                                   scope = 'fc8')
        print("SHAPE!!!!", net)
    return net_cnn, net2, net1, net
#########################
def gen_anchor_bx(bbx_size, bbx_ratio, im_width, im_height):
    bbx_size = [8, 16, 32]
    bbx_ratio = [0.5, 1, 1.5]
    num_anchors = len(bbx_size)*len(bbx_ratio)
    centres = [[i, j] for i in range(8, im_width, 16) for j in range(8, im_height, 16)]
    return centres

def draw_bbx(bbxs_sizes_img, fig1, sze_of_img, im_width, im_height):
    rec_patches = []
    for bbx in bbxs_sizes_img:
        #print "bbx", bbx
        #print sze_of_img[0]/float(im_width)
        #print ((int(bbx[0]*im_width/float(sze_of_img[0])),
        #                                  int(bbx[1]*im_height/float(sze_of_img[1]))),
        #                                 int((bbx[2]-bbx[0])*im_width/float(sze_of_img[0])),
        #                                 int((bbx[3]-bbx[1])*im_height/float(sze_of_img[1])))
        colors = np.random.random((1, 3))
        fig1.add_patch(patches.Rectangle((int(bbx[0]*im_width/float(sze_of_img[0])),
                                          int(bbx[1]*im_height/float(sze_of_img[1]))),
                                         int((bbx[2]-bbx[0])*im_width/float(sze_of_img[0])),
                                         int((bbx[3]-bbx[1])*im_height/float(sze_of_img[1])),
                                             linewidth=3,
                                             edgecolor=colors[0],
                                             facecolor='none'))
    # Add the patch to the Axes

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

tf.reset_default_graph()
im_width = 224
im_height = 224
print "good till here1"
bbx_size = [8, 16, 32]
bbx_ratio = [0.5, 1, 1.5]
centres = gen_anchor_bx(bbx_size, bbx_ratio, im_width, im_height)
####Loading data######
type_data = 'person'
shw_example = False
JPEG_images, Annotation_images, df = read_voc.load_data_full(type_data, shw_example)
bbxs_sizes = read_voc.getting_all_bbx(Annotation_images)



print "centres", centres
im_placeholder = tf.placeholder(tf.uint8, [None, im_height, im_width, 3])
net_cnn, net2, net1, logits = netvgg(im_placeholder, is_training=False)
prediction = tf.nn.softmax(logits)
predicted_labels = tf.argmax(prediction, 1)
print "Net1 ", net1.shape, "Net2", net2.shape
print "good till here2"

#vgg
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
        #img = Image.open('data/cat1.jpg')
        img = Image.open(JPEG_images[1])
        sze_of_img = img.size

        img = np.array(img.resize((im_width,im_height), Image.ANTIALIAS))
        _,fig1 = plt.subplots(1)
        fig1.imshow(img)
        draw_bbx(bbxs_sizes[1], fig1, sze_of_img, im_width, im_height)
        plt.show()

        net_cnn_s, net2_s, net1_s, pred_lbl, proba = sess.run([net_cnn, net2, net1,
                                                               predicted_labels, prediction],
                                                      feed_dict={im_placeholder:np.expand_dims(img, axis=0)})
        print(pred_lbl)
        print(net_cnn_s.shape)
        print(net1_s.shape)
        print(net2_s.shape)
