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
            net1 = slim.conv2d(net, 2*num_anchors, [1, 1], scope= "prob", activation_fn=tf.nn.sigmoid)

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
    #bbx_size = [8, 16, 32]
    #bbx_ratio = [1, 0.5, 2]
    num_anchors = len(bbx_size)*len(bbx_ratio)
##    centres = [[i, j, m*1/k, m*k] for i in range(8, im_width, 16) for j in range(8, im_height, 16)\
##                for m in bbx_size for k in bbx_ratio]
    centres = [] # [cx, cy, w, h]
    centres_mmxy = [] # [xmin, ymin, xmax, ymax]
    for i in range(8, im_width, 16):
        for j in range(8, im_height, 16):
            for m in bbx_size:
                for k in bbx_ratio:
                    centres.append([j, i, int(m*1/k), int(m*k)])
                    centres_mmxy.append([int(j-m*1/(2*k)+1), int(i-m*k/2+1), int(j+m*1/(2*k)), int(i+m*k/2)])
                    #for :
                    #
    print centres_mmxy[0], centres[0], "centres"
    return centres, centres_mmxy
def bbx_minmax_to_centre_size(bbx):
    w = bbx[2]-bbx[0]+1
    h = bbx[3]-bbx[1]+1
    xc = w/2+bbx[0]-1
    yc = h/2+bbx[1]-1

    return [int(xc), int(yc), int(w), int(h)]

def encoding_bbx(target, anchor):
    tx = target[0]-anchor[0]/float(anchor[2])
    ty = target[1]-anchor[1]/float(anchor[3])
    tw = np.log(target[2]/float(anchor[2]))
    th = np.log(target[3]/float(anchor[3]))
    return [(tx), (ty), (tw), (th)]
def decoding_bbx(enc_target, anchor):
    px = enc_target[0]*float(anchor[2])+anchor[0]
    py = enc_target[1]*float(anchor[3])+anchor[1]
    pw = np.exp(enc_target[2])*float(anchor[2])
    ph = np.exp(enc_target[3])*float(anchor[3])

    return[px, py, pw, ph], [px-pw/2+1, py-ph/2+1, px+pw/2, py+ph/2]

def get_training_data(centres_mmxy, target_data):
    #[xmin, ymin, xmax, ymax]
    dl = []
    n_anchor = []
    new_output = []
    encoded_data = []
    candidates = []
    count = [0, 0, 0]
    for anchor in centres_mmxy:
        previous_iou = 0
        mrk = 0
        is_t = 0
        for target in target_data:
            iou = read_voc.bb_intersection_over_union(anchor, target)
            if target == [2, 0, 162, 206]:
                dl.append(iou)
                #print max(dl)
            if iou > 0.7:
                is_t = 1
                if iou > previous_iou:
                    anchor_con = bbx_minmax_to_centre_size(anchor)
                    target_con = bbx_minmax_to_centre_size(target)
                    candidates = encoding_bbx(target_con, anchor_con)
                    candidates.append(iou)
                    if target == [2, 0, 162, 206]:
                        print("fucking cand", candidates)
                        print("fucking anchors", anchor)
                        print("fucking targets", target)
                    previous_iou = iou

                    #dec_target =
                    #candidates =
            elif iou < 0.3:
                mrk += 1
        if candidates == []:
            pass
            #encoded_data.append([0, 0, 0, 0])
        else:
            encoded_data.append(candidates)
            n_anchor.append(anchor)
        candidates = []
        if is_t == 1:
            count[0] = 1+count[0]
            new_output.append([1, 0])
        elif (mrk == len(target_data)) and (is_t == 0):
            new_output.append([0, 1])
            count[1] = 1+count[1]
        elif is_t == 0:
            new_output.append([0, 0])
            count[2] = 1+count[2]
    #print "letsee", len(new_output), count, len(encoded_data), encoded_data
    return new_output, encoded_data, n_anchor

def new_get_training_data(centres_mmxy, target_data):
    #[xmin, ymin, xmax, ymax]
    dl = []
    n_anchor =[]
    previous_iou = list(-1*np.ones(len(centres_mmxy)))
    new_output = np.zeros([len(centres_mmxy), 2])
    new_output[:,1] = np.ones(new_output[:,1].shape)
    new_output = [list(x) for x in new_output]

    encoded_data = np.zeros([len(centres_mmxy), 4])
    encoded_data = [list(x) for x in encoded_data]
    print("enc_dat", len(encoded_data))
    candidates = []
    count = [0, 0, 0]
    for target in target_data:

        target_count = 0
        is_t = 0
        anchor_count = 0
        keep_old = -1
        for anchor in centres_mmxy:
            iou = read_voc.bb_intersection_over_union(anchor, target)

            if iou > 0.7:
                new_output[anchor_count] =[1, 0]
                target_count = 1
                if iou > previous_iou[anchor_count]:
                    anchor_con = bbx_minmax_to_centre_size(centres_mmxy[anchor_count])
                    target_con = bbx_minmax_to_centre_size(target)
                    print('encoded_data', anchor_count, encoding_bbx(target_con, anchor_con))
                    print('ecasdf', encoded_data[anchor_count])
                    encoded_data[anchor_count] = encoding_bbx(target_con, anchor_con)
                    n_anchor.append(centres_mmxy[anchor_count])
            else:
                if iou > keep_old:
                    best_anch = anchor_count
                    keep_old = iou
            if (iou >= 0.3) and (iou <= 0.7) and (iou > previous_iou[anchor_count]):
                new_output[anchor_count] =[0, 0]

            if iou > previous_iou[anchor_count]:
                previous_iou[anchor_count] = iou
                #keep_old = anchor_count
            anchor_count += 1
        if target_count == 0:
            anchor_con = bbx_minmax_to_centre_size(centres_mmxy[best_anch])
            target_con = bbx_minmax_to_centre_size(target)
            encoded_data[best_anch] = encoding_bbx(target_con, anchor_con)
            new_output[best_anch] =[1, 0]
            n_anchor.append(centres_mmxy[best_anch])

    #print "letsee", len(new_output), count, len(encoded_data), encoded_data
    return new_output, encoded_data, n_anchor

def draw_bbx(bbxs_sizes_img, fig1, sze_of_img, im_width, im_height, is_anchor = False):
    rec_patches = []
    if is_anchor:
        im_width = sze_of_img[0]
        im_height = sze_of_img[1]
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

def resizing_targets(bbxs_sizes_img, sze_of_img, im_width, im_height):
    #print "bbx", bbxs_sizes_img, len(bbxs_sizes_img)
    for n in range(len(bbxs_sizes_img)):
        #print "nnn", bbxs_sizes_img[n]
        bbxs_sizes_img[n][0] = int(bbxs_sizes_img[n][0]*im_width/float(sze_of_img[0]))
        bbxs_sizes_img[n][1] = int(bbxs_sizes_img[n][1]*im_height/float(sze_of_img[1]))
        bbxs_sizes_img[n][2] = int(bbxs_sizes_img[n][2]*im_width/float(sze_of_img[0]))
        bbxs_sizes_img[n][3] = int(bbxs_sizes_img[n][3]*im_height/float(sze_of_img[1]))
    return bbxs_sizes_img

def losses(logits, labels):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

def optimize(losses):
    global_step = tf.contrib.framework.get_or_create_global_step()
    lr = 0.1
    learning_rate = tf.train.exponential_decay(lr, global_step,
                                             100000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(losses, global_step=global_step)#,
                #var_list=slim.get_model_variables("finetune"))
    return train_op

tf.reset_default_graph()
im_width = 224
im_height = 224
print "good till here1"
bbx_size = [8, 64, 128]#[8, 64, 128]#[8, 16, 32]
bbx_ratio = [1, 1/1.5, 1.5]#[1, 0.5, 2]
centres, centres_mmxy = gen_anchor_bx(bbx_size, bbx_ratio, im_width, im_height)
print 'centres', len(centres)
print 'centres_mmxy', len(centres_mmxy)
####Loading data######
type_data = 'person'
shw_example = False
JPEG_images, Annotation_images, df = read_voc.load_data_full(type_data, shw_example)
bbxs_sizes = read_voc.getting_all_bbx(Annotation_images)
#print "bbxs_sizes", bbxs_sizes

im_placeholder = tf.placeholder(tf.float32, [None, im_height, im_width, 3])
y_ = tf.placeholder(tf.float32, [None, 9*14*14, 2])
y_reg = tf.placeholder(tf.float32, [None, 9*14*14, 4])

net_cnn, net2, net1, logits = netvgg(im_placeholder, is_training=False)

sum_y = tf.reduce_sum(y_, axis=-1, keep_dims=True)
sum_y = tf.cast(sum_y, tf.float32)

bull_a = tf.reduce_sum(y_, axis=-1)
bull_a = tf.cast(bull_a, tf.bool)

mult_net1 = tf.multiply(net1, sum_y)
mult_net2 = tf.multiply(net2, sum_y)

bm_net1 = tf.boolean_mask(mult_net1, bull_a)
bm_y_ = tf.boolean_mask(y_, bull_a)
bm_net1 = tf.argmax(bm_net1, -1)
bm_y_ = tf.argmax(bm_y_, -1)

sft_max_w_logit = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits = mult_net1)
cross_entropy = tf.reduce_mean(sft_max_w_logit)
###train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = optimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
argmax_1 = tf.argmax(net1,-1)
argmax_2 = tf.argmax(y_,-1)
correct_prediction = tf.equal(bm_net1, bm_y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###
print(y_reg, net2)
F = mult_net2-y_reg
comparison = tf.stop_gradient(tf.to_float(tf.less(tf.abs(F), tf.constant([1,1,1,1], dtype=tf.float32))))
weird_m = 0.5*tf.square(F)*comparison+(tf.abs(F)-0.5)*(1-comparison)
rs = tf.reduce_sum(weird_m, -1, keep_dims=True)#, axis=-1, keep_dims=True)
ra = tf.reduce_mean(rs)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy+ra)
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy+ra)

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
        show_img_ = True

        img = np.array(img.resize((im_width,im_height), Image.ANTIALIAS))
        if show_img_:
            _,fig1 = plt.subplots(1)
            #fig1.axis([-600, 600, -600, 600])
            fig1.imshow(img)
        bbxs_sizes[1] = resizing_targets(bbxs_sizes[1], sze_of_img, im_width, im_height)
        print("suizerfasdf", bbxs_sizes[1])
        #training_data, encoded_training, selected_anchors = get_training_data(centres_mmxy, bbxs_sizes[1])
        training_data, encoded_training, selected_anchors = new_get_training_data(centres_mmxy, bbxs_sizes[1])
        #print("hahaah",np.sum(np.array(training_data)-np.array(training_data1), axis=0))
        #print(selected_anchors2)
        #print("list", list(np.array(training_data)-np.array(training_data1)))
        y_hat = np.array(training_data)
        print np.expand_dims(y_hat, axis=0).shape, y_hat.shape
        draw_bbx(selected_anchors, fig1, sze_of_img, im_width, im_height, True)
        #draw_bbx(bbxs_sizes[1], fig1, sze_of_img, im_width, im_height, True)
        #draw_bbx(bbxs_sizes[1], fig1, sze_of_img, im_width, im_height, False)
        #draw_bbx(centres_mmxy[19*5:19*5+10], fig1, sze_of_img, im_width, im_height, True)
        plt.show()
        ay= np.expand_dims(y_hat, axis=0)
        q = np.where((ay == [0,0]).all(axis=-1))
        for i in range(5500):
            mult_net2_s, bm_y_s, bm_net1_s, argmax_1s, argmax_2s, _, net_cnn_s, net2_s, net1_s, pred_lbl, proba, x_entropy, sft_max_w_logit_s = sess.run([mult_net2, bm_y_, bm_net1, argmax_1, argmax_2, train_step, net_cnn, net2, net1,
                                                               predicted_labels, prediction, cross_entropy, sft_max_w_logit],
                                                      feed_dict={im_placeholder:np.expand_dims(img, axis=0), y_:np.expand_dims(y_hat, axis=0), y_reg: np.expand_dims(encoded_training, axis=0)})


            #print sft_max_w_logit_s.shape
            #print net1_s[0][1]
            if i % 50 == 0:


                print(i, x_entropy)
                print("argmax1", argmax_1s, net1_s[0][1])
                print("argmax2", argmax_2s, net1_s[0][1])
                tr_acc = accuracy.eval(feed_dict={im_placeholder:np.expand_dims(img, axis=0), y_:np.expand_dims(y_hat, axis=0)})
                print('tr_acc', tr_acc)
        print(pred_lbl)
        print(net_cnn_s.shape)
        print(net1_s.shape)
        print(net2_s.shape)
        print(net2_s)
        print(np.sum(mult_net2_s - np.array(encoded_training),axis=-2)/1764)
        g = np.greater(mult_net2_s, np.array([0,0,0,0]))
        b = np.any(g, axis=-1)
        dec = []
        c_dec = []
        for x in range(0,len(centres)):
            #print(mult_net2_s[0][x])
            dec.append(decoding_bbx( mult_net2_s[0][x],centres[x])[0])
            c_dec.append(decoding_bbx( mult_net2_s[0][x],centres[x])[1])
        dec = np.expand_dims(np.array(dec), 0)
        c_dec =  np.expand_dims(np.array(c_dec), 0)
        print dec.shape, g.shape, mult_net2_s.shape, dec[b].shape
        print c_dec[b]
        if show_img_:
            _,fig2 = plt.subplots(1)
            #fig1.axis([-600, 600, -600, 600])
            fig2.imshow(img)
            draw_bbx(list(c_dec[b]), fig2, sze_of_img, im_width, im_height, True)
            plt.show()
        #anchors2 = net2_s[b]
        #print(anchors2)
        ##print(x_entropy)
        ##print("res", net1_s)
        ##print("pred", y_hat)
        ##m1 = np.argmax(net1_s, axis = -1)
        ##m2 = np.argmax(y_hat, axis = -1)
        #print(np.sum(np.equal(m1, m2)))/1764.0

        ##print(bm_y_s, bm_net1_s)
