import os
import time
import cv2

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
    """
        Output: probability and regression
        net1: probability
        net2: regression
    """
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

            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            net_cnn = net
            net1 = slim.conv2d(net, 2*num_anchors, [1, 1], scope= "prob",
            weights_initializer=initializer, activation_fn=None)#tf.nn.sigmoid)

            net1 = Reshape((-1, 2), input_shape=(net1.shape[1], net1.shape[2], net1.shape[3]))(net1)
            #net1 = tf.reshape(net1, [0, -1, -1, 2])
            input_shape = tf.shape(net1)

            #rshpsssss = tf.reshape(net1, [-1, input_shape[-1]])
            #print "rshpsssss", rshpsssss
            #net1 = tf.nn.softmax(net1)
            net2 = slim.conv2d(net, 4*num_anchors, [1, 1], scope='bbox', weights_initializer=initializer, activation_fn=None)
            net2 = Reshape((-1, 4), input_shape=(net2.shape[1], net2.shape[2], net2.shape[3]))(net2)

            net = slim.max_pool2d(net, [2, 2], scope = 'pool5')


    return net_cnn, net2, net1
#########################
def gen_anchor_bx(bbx_size, bbx_ratio, im_width, im_height):
    """
        Output: Anchor box size
        centres: [centre_x, centre_y, width, heigth]
        centres_mmxy: [x_min, y_min, x_max, y_max]
    """
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
    """
        Output: transform minxy-maxxy bbx format to [centre_x, centre_y, width, heigth]
    """
    w = bbx[2]-bbx[0]+1
    h = bbx[3]-bbx[1]+1
    xc = w/2+bbx[0]-1
    yc = h/2+bbx[1]-1

    return [int(xc), int(yc), int(w), int(h)]

def encoding_bbx(target, anchor):
    """
        Output: Encoding Anchor box
        from [centre_x, centre_y, width, heigth] ->
        [(tx), (ty), (tw), (th)]
    """
    tx = (target[0]-anchor[0])/float(anchor[2])
    ty = (target[1]-anchor[1])/float(anchor[3])
    tw = np.log(target[2]/float(anchor[2]))
    th = np.log(target[3]/float(anchor[3]))
    return [(tx), (ty), (tw), (th)]

def decoding_bbx(enc_target, anchor):
    """
        Output: Decoding Anchor box
        from [(tx), (ty), (tw), (th)] ->
        [x_min, y_min, x_max, y_max]
    """
    px = enc_target[0]*float(anchor[2])+anchor[0]
    py = enc_target[1]*float(anchor[3])+anchor[1]
    pw = np.exp(enc_target[2])*float(anchor[2])
    ph = np.exp(enc_target[3])*float(anchor[3])

    return[px, py, pw, ph], [px-pw/2+1, py-ph/2+1, px+pw/2, py+ph/2]

def new_get_training_data(centres_mmxy, target_data):
    #[xmin, ymin, xmax, ymax]
    """
        Output: Decoding Anchor box
        new_output: p(object) label [1, 0] or [0, 1]
        encoded_data: regresion [(tx), (ty), (tw), (th)]

        NOT USED OUTPUTS!
        n_anchor: [x_min, y_min, x_max, y_max]
        n_anchor_normal_form: [centre_x, centre_y, width, heigth]
    """
    true_anchors = []
    dl = []
    n_anchor =[]
    n_anchor_normal_form = []
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
                    print('ecasdf', target_con, anchor_con)
                    print('encoded_data', anchor_count, encoding_bbx(target_con, anchor_con))
                    encoded_data[anchor_count] = encoding_bbx(target_con, anchor_con)
                    n_anchor.append(centres_mmxy[anchor_count])
                    n_anchor_normal_form.append(anchor_con)
                    true_anchors.append(anchor_count)
            else:
                if (iou >= keep_old) and (anchor_count not in true_anchors):
                    best_anch = anchor_count
                    keep_old = iou
            if (iou >= 0.3) and (iou <= 0.7) and (iou > previous_iou[anchor_count]):
                new_output[anchor_count] =[0, 1] #[0, 0]

            if iou > previous_iou[anchor_count]:
                previous_iou[anchor_count] = iou
                #keep_old = anchor_count
            anchor_count += 1
        if target_count == 0:
            print "IT'S HERE!!!!!!!asdfasfasdfdf!!!EALA", centres_mmxy[best_anch]
            anchor_con = bbx_minmax_to_centre_size(centres_mmxy[best_anch])
            target_con = bbx_minmax_to_centre_size(target)
            encoded_data[best_anch] = encoding_bbx(target_con, anchor_con)
            new_output[best_anch] =[1, 0]
            n_anchor.append(centres_mmxy[best_anch])
            n_anchor_normal_form.append(anchor_con)
            true_anchors.append(best_anch)

    #print "letsee", len(new_output), count, len(encoded_data), encoded_data
    return new_output, encoded_data, n_anchor, n_anchor_normal_form

def draw_bbx(bbxs_sizes_img, fig1, sze_of_img, im_width, im_height, is_anchor = False):
    """
        Output: Draws the bounding boxes.
        If is_anchor == True, then we don't resize, ow, we resized the bounding box
    """
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
    """
        Output: resizes the bounding box training output to the fixed size of the CNN
    """
    #print "bbx", bbxs_sizes_img, len(bbxs_sizes_img)
    for n in range(len(bbxs_sizes_img)):
        #print "nnn", bbxs_sizes_img[n]
        bbxs_sizes_img[n][0] = int(bbxs_sizes_img[n][0]*im_width/float(sze_of_img[0]))
        bbxs_sizes_img[n][1] = int(bbxs_sizes_img[n][1]*im_height/float(sze_of_img[1]))
        bbxs_sizes_img[n][2] = int(bbxs_sizes_img[n][2]*im_width/float(sze_of_img[0]))
        bbxs_sizes_img[n][3] = int(bbxs_sizes_img[n][3]*im_height/float(sze_of_img[1]))
    return bbxs_sizes_img


def load_data(n_examples, im_width, im_height, type_data):
    """
        Output: Batch training data
        sze_of_img_all: Img size of each image in the batch
        group_img: stacked resized training images [n, 244,244,3]
        np.array(training_data): traning probability output
        np.array(encoded_training): training regression output

        NOTE USED
        np.array(selected_anchors):
        np.array(selected_anchors_normal):
    """
    shw_example = False
    JPEG_images, Annotation_images, df = read_voc.load_data_full(type_data, shw_example)
    bbxs_sizes = read_voc.getting_all_bbx(Annotation_images, type_data, df)
    training_data = []
    encoded_training = []
    selected_anchors = []
    selected_anchors_normal = []
    cnt = 0
    sze_of_img_all = np.zeros([len(n_examples), 2])
    group_img = np.zeros([len(n_examples), im_width, im_height, 3], dtype=np.uint8)
    for n_example in n_examples:
        img = Image.open(JPEG_images[n_example])
        sze_of_img = img.size
        show_img_ = False
        #print bbxs_sizes[n_example]
        img = np.array(img.resize((im_width,im_height), Image.ANTIALIAS))
        group_img[cnt] = img
        sze_of_img_all[cnt] = sze_of_img
        cnt += 1

        bbxs_sizes[n_example] = resizing_targets(bbxs_sizes[n_example], sze_of_img, im_width, im_height)
        print("bbxs_sizes first example", bbxs_sizes[n_example])
        #training_data, encoded_training, selected_anchors = get_training_data(centres_mmxy, bbxs_sizes[1])
        single_training_data, single_encoded_training, single_selected_anchors, single_selected_anchors_normal = new_get_training_data(centres_mmxy, bbxs_sizes[n_example])
        training_data.append(single_training_data)
        encoded_training.append(single_encoded_training)
        selected_anchors.append(single_selected_anchors)
        selected_anchors_normal.append(single_selected_anchors_normal)
        print img.dtype, group_img.dtype, np.expand_dims(img, axis=0).dtype
        if show_img_:
            _,fig1 = plt.subplots(1)

            #fig1.axis([-600, 600, -600, 600])

            #print newImaa.shape
            fig1.imshow(group_img[cnt-1], vmin=0, vmax=255)
            #fig1.imshow(img, vmin=0, vmax=255)
            draw_bbx(single_selected_anchors, fig1, sze_of_img, im_width, im_height, True)
            plt.show()
    return sze_of_img_all, group_img, np.array(training_data), np.array(encoded_training), np.array(selected_anchors), np.array(selected_anchors_normal)

def losses(logits, labels):
    '''
        NOT USED
    '''
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

def optimize(losses):
    '''
        NOT USED
    '''
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
bbx_size = [8, 16, 32]#[8, 64, 128]#[8, 64, 128]#[8, 16, 32]
bbx_ratio = [1, 1/1.5, 1.5]#[1, 0.5, 2]


centres, centres_mmxy = gen_anchor_bx(bbx_size, bbx_ratio, im_width, im_height)
print 'centres', len(centres)
print 'centres_mmxy', len(centres_mmxy)
####Loading data######
type_data = ''
shw_example = False
n_example = 215#25#432
JPEG_images, Annotation_images, df = read_voc.load_data_full(type_data, shw_example)
bbxs_sizes = read_voc.getting_all_bbx(Annotation_images, type_data, df)
#print "bbxs_sizes", bbxs_sizes

im_placeholder = tf.placeholder(tf.float32, [None, im_height, im_width, 3])

y_ = tf.placeholder(tf.float32, [None, 9*14*14, 2], name = 'ob_prob')
y_reg = tf.placeholder(tf.float32, [None, 9*14*14, 4], name = 'ob_reg')

net_cnn, net2, net1 = netvgg(im_placeholder, is_training=False)

sum_yreg = tf.reduce_sum(y_reg, axis=-1, keep_dims=True)
#sum_yreg = tf.cast(sum_yreg, tf.float32)
print "SUM!!!!!!!!!!!!!!", sum_yreg
sum_yreg = (tf.not_equal(sum_yreg, 0))
sum_yreg = tf.cast(sum_yreg, tf.float32)

print "SUM!!!!!!!!!!!!!!", sum_yreg
sum_y = tf.reduce_sum(y_, axis=-1, keep_dims=True)
sum_y = tf.cast(sum_y, tf.float32)
print "SUM!!!!!!!!!!!!!!", sum_y


bull_a = tf.reduce_sum(y_, axis=-1)
bull_a = tf.cast(bull_a, tf.bool)


mult_net1 = tf.multiply(net1, sum_y)
mult_net11 = mult_net1
#mult_net11 = net1*(2*sum_yreg-1) ## If you want to use the mask of the regressor
print mult_net11
mult_net2 = tf.multiply(net2, sum_yreg)

bm_net1 = tf.boolean_mask(mult_net1, bull_a)
bm_y_ = tf.boolean_mask(y_, bull_a)
bm_net1 = tf.argmax(bm_net1, -1)
bm_y_ = tf.argmax(bm_y_, -1)

#sft_max_w_logit = tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(y_, perm=[0,2,1]), logits = tf.transpose(mult_net11, perm=[0,2,1]))
sft_max_w_logit = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits = mult_net11)
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
hard_l2 = 0.5*tf.square(F)*comparison+(tf.abs(F)-0.5)*(1-comparison)
rs = tf.reduce_sum(hard_l2, -1, keep_dims=True)#, axis=-1, keep_dims=True)
ra = tf.reduce_mean(rs)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy+ra)
#train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy+ra)


print "Net1 ", net1.shape, "Net2", net2.shape
print "good till here2"

###LOAD DATA #####
n_examples =[215, 25]#, 5, 10, 11, 12,13,14,15,16,17,18,19,20]
n_examples = range(1, 64)
sze_of_img_all, img_all, training_data_all, encoded_training_all, c, selected_anchors_normal_all = load_data(n_examples, im_width, im_height, type_data)
print "SHAPES!asdfasdaaKKKKK"
#print ga.shape, a.shape, b.shape, c.shape, d.shape
#vgg
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    path = pathlib.Path('./../weights/vgg16_weights.npz')
    if(path.is_file()):
        #print("it's a mattafaca")
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
                                                             'vgg_16/conv5/conv5_3/biases' : init_weights['conv5_3_b']

                                                             })
        #print(assign_op, feed_dict_init)

        sess.run(assign_op, feed_dict_init)
        #img = Image.open('data/cat1.jpg')
        """
        NOT USED REFER TO BACKUP
        img = Image.open(JPEG_images[n_example])
        sze_of_img = img.size
        show_img_ = False
        print bbxs_sizes[n_example]
        img = np.array(img.resize((im_width,im_height), Image.ANTIALIAS))
        if show_img_:
            _,fig1 = plt.subplots(1)
            #fig1.axis([-600, 600, -600, 600])
            fig1.imshow(img)
        bbxs_sizes[n_example] = resizing_targets(bbxs_sizes[n_example], sze_of_img, im_width, im_height)
        print("bbxs_sizes first example", bbxs_sizes[n_example])
        #training_data, encoded_training, selected_anchors = get_training_data(centres_mmxy, bbxs_sizes[1])
        training_data, encoded_training, selected_anchors, selected_anchors_normal = new_get_training_data(centres_mmxy, bbxs_sizes[n_example])

        #encoded_training = np.array(encoded_training,dtype='f')
        #print "shape!!!!!!!!!!!!!!!!!!!!!!!", encoded_training.shape
        #print("hahaah",np.sum(np.array(training_data)-np.array(training_data1), axis=0))
        #print(selected_anchors2)
        #print("list", list(np.array(training_data)-np.array(training_data1)))

        if show_img_:
            draw_bbx(selected_anchors, fig1, sze_of_img, im_width, im_height, True)
            #draw_bbx(bbxs_sizes[n_example], fig1, sze_of_img, im_width, im_height, True)
            #draw_bbx(bbxs_sizes[1], fig1, sze_of_img, im_width, im_height, False)
            #draw_bbx(centres_mmxy[19*5:19*5+10], fig1, sze_of_img, im_width, im_height, True)
            plt.show()
        """
        y_hat = training_data_all
        ay= training_data_all
        q = np.where((ay == [0,0]).all(axis=-1))
        g2 = np.greater(y_hat[:,:,0], np.array([0.7]))
        ##
        sma = [0 , 0]
        for xta in y_hat:
            if np.array_equal(xta, [1, 0]):
                print xta
            if np.array_equal(xta, [0, 1]):
                sma[0] += 1
            if np.array_equal(xta, [0, 0]):
                sma[1] += 1
        print sma
        ##
        saver = tf.train.Saver()

        #g2 = np.not_equal(np.expand_dims(encoded_training, axis=0)[:,:,0], np.array([0]))
        for i in range(5000):
#            mult_net2_s, bm_y_s, bm_net1_s, argmax_1s, argmax_2s, _, net_cnn_s, net2_s, net1_s, pred_lbl, proba, x_entropy, sft_max_w_logit_s = sess.run([mult_net2, bm_y_, bm_net1, argmax_1, argmax_2, train_step, net_cnn, net2, net1,
#                                                               predicted_labels, prediction, cross_entropy, sft_max_w_logit],
#                                                      feed_dict={im_placeholder:np.expand_dims(img, axis=0), y_:np.expand_dims(y_hat, axis=0), y_reg: np.expand_dims(encoded_training, axis=0)})
            mult_net1_s, mult_net2_s, bm_y_s, bm_net1_s, argmax_1s, argmax_2s, _, net_cnn_s, net2_s, net1_s = sess.run([mult_net11, mult_net2,
                                                bm_y_, bm_net1, argmax_1, argmax_2, train_step, net_cnn, net2, net1],
                                                      feed_dict={im_placeholder:img_all, y_:y_hat, y_reg: encoded_training_all})


            #print sft_max_w_logit_s.shape
            #print net1_s[0][1]
            if i % 50 == 0:


                print(i)
                print("argmax1", argmax_1s, net1_s[0][1])
                print("argmax2", argmax_2s, net1_s[0][1], np.sum(net1_s[0][1]))
                tr_acc = accuracy.eval(feed_dict={im_placeholder:img_all, y_:y_hat})
                print('tr_acc', tr_acc)
                print("predicted", mult_net2_s[g2])






        print(net_cnn_s.shape)
        print(net1_s.shape)
        print(net2_s.shape)
        print(net2_s)
        print(np.sum(mult_net2_s - encoded_training_all,axis=-2)/1764)
        g = np.greater(mult_net2_s, np.array([0,0,0,0]))
        b = np.any(g, axis=-1)
        print "check if it works :("
        for prx in mult_net1_s[0]:
            #print np.argmax(prx)
            if prx[0] > 0.5:
                print prx

        print ":'(", np.sum(mult_net1_s[0], keepdims=True)
        g2 = np.greater(mult_net1_s[:,:,0], np.array([0.7]))
        dec = []
        c_dec = []
        #print("centres", mult_net2_s[g2])
        print "START X"
        for x in range(0,len(centres)):
            #print(mult_net2_s[0][x])
            dec.append(decoding_bbx( mult_net2_s[0][x],centres[x])[0])
            c_dec.append(decoding_bbx( mult_net2_s[0][x],centres[x])[1])
            #target
            #c_dec.append(decoding_bbx(np.expand_dims(encoded_training, axis=0)[0][x],centres[x])[1])
        dec = np.expand_dims(np.array(dec), 0)
        c_dec =  np.expand_dims(np.array(c_dec), 0)
        print "wtf", dec.shape, g.shape, mult_net2_s.shape, g2[0].shape,
        print "tf2", dec[0][g2[0]].shape
        #print c_dec[g2]
        show_img_ = True

        print("predicted", mult_net2_s[0][g2[0]])

        print("target", encoded_training_all[0][g2[0]])

        print "target2", selected_anchors_normal_all[0]

        if show_img_:
            _,fig2 = plt.subplots(1)
            #fig1.axis([-600, 600, -600, 600])
            fig2.imshow(img_all[0])
            draw_bbx(list(c_dec[0][g2[0]]), fig2, sze_of_img_all[0], im_width, im_height, True)
            plt.show()


        ##print(bm_y_s, bm_net1_s)
        for ii in range(len(n_examples)):
            dec = []
            c_dec = []
            #print("centres", mult_net2_s[g2])
            print "START X"
            for x in range(0,len(centres)):
                #print(mult_net2_s[0][x])
                dec.append(decoding_bbx( mult_net2_s[ii][x],centres[x])[0])
                c_dec.append(decoding_bbx( mult_net2_s[ii][x],centres[x])[1])
                #target
                #c_dec.append(decoding_bbx(np.expand_dims(encoded_training, axis=0)[0][x],centres[x])[1])
            dec = np.expand_dims(np.array(dec), 0)
            c_dec =  np.expand_dims(np.array(c_dec), 0)
            print "wtf", dec.shape, g.shape, mult_net2_s.shape, g2[ii].shape,
            #print c_dec[g2]
            show_img_ = True

            print("predicted", mult_net2_s[ii][g2[ii]])

            print("target", encoded_training_all[ii][g2[ii]])

            print "target2", selected_anchors_normal_all[ii]

            if show_img_:
                print ii
                _,fig2 = plt.subplots(1)
                #fig1.axis([-600, 600, -600, 600])
                fig2.imshow(img_all[ii])
                draw_bbx(list(c_dec[0][g2[ii]]), fig2, sze_of_img_all[ii], im_width, im_height, True)
                plt.show()
        save_path = saver.save(sess, "/home/hanz/Documents/2018_lib/f-rcnn/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
