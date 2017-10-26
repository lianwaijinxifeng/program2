import tensorflow as tf
import inference
import os
import random as raandom
import math
import numpy as np
from PIL import Image
from pylab import *


Batchsize=100
LEARNING_RATE_BASE=0.001
STEPS=100000

datapath='./peddatatr'
testpath='./peddatate'
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def fc_layer(input,name,n_out):
    n_in=input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        biases=tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
        activation=tf.nn.relu_layer(input,kernel,biases,name=scope)
        return activation

def inf(im1,im2):
    mod1=inference.sameornot(im1,reuse=False)
    mod2=inference.sameornot(im2,reuse=True)
    net=tf.concat([mod1,mod2],3)
    net = tf.contrib.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME',weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope='conv6')
    net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
    net = tf.contrib.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME',weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope='conv7')
    net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
    net = tf.contrib.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME',weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope='conv8')
    net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
    shp=net.get_shape()
    flattened_shape=shp[1].value*shp[2].value*shp[3].value
    net=tf.reshape(net,[-1,flattened_shape],name='resh1')
    net=fc_layer(net,'fc1',4096)
    net=tf.nn.dropout(net,0.5)
    net=fc_layer(net,'fc2',512)
    net=tf.nn.dropout(net,0.5)
    net=fc_layer(net,'fc3',2)
    softmax=tf.nn.softmax(net)
    predictions=tf.argmax(softmax,1)
    return softmax,predictions

def get_batch(path):
    input1 = np.zeros([Batchsize, 192, 64, 3])
    input2 = np.zeros([Batchsize, 192, 64, 3])
    labin = np.zeros([Batchsize, 1])
    seqdir = os.listdir(path)
    numseq = len(seqdir)
    seqnum = raandom.randrange(numseq)
    targetpath = path + '/' + seqdir[seqnum]
    targetlst = os.listdir(targetpath)
    numtar = len(targetlst)
    # get one batch
    for j in range(Batchsize):
        if j % 2 == 0:
            tarnum = raandom.randrange(numtar)
            picpath = targetpath + '/' + targetlst[tarnum]
            # get two pic in this path, resize and add to batch
            picdir = os.listdir(picpath)
            numpic = len(picdir)
            picnum1 = raandom.randrange(numpic)
            picnum2 = raandom.randrange(numpic)
            pic1path = picpath + '/' + picdir[picnum1]
            pic2path = picpath + '/' + picdir[picnum2]
            img1 = array(Image.open(pic1path))
            img2 = array(Image.open(pic2path))
            input1[j] = img1
            input2[j] = img2
            labin[j] = 1
            # print pic1path
            # print pic2path
            # print labin[j]


        else:
            tarnum1 = raandom.randrange(numtar)
            picpath1 = targetpath + '/' + targetlst[tarnum1]
            tarnum2 = raandom.randrange(numtar)
            picpath2 = targetpath + '/' + targetlst[tarnum2]
            # get two pic in these two pathes, resize and add to batch, don't forget to check if tarnum1==tarnum2
            picdir1 = os.listdir(picpath1)
            numpic1 = len(picdir1)
            picdir2 = os.listdir(picpath2)
            numpic2 = len(picdir2)
            picnum1 = raandom.randrange(numpic1)
            picnum2 = raandom.randrange(numpic2)
            pic1path = picpath1 + '/' + picdir1[picnum1]
            pic2path = picpath2 + '/' + picdir2[picnum2]
            img1 = array(Image.open(pic1path))
            img2 = array(Image.open(pic2path))
            input1[j] = img1
            input2[j] = img2
            if tarnum1 == tarnum2:
                labin[j] = 1
            else:
                labin[j] = 0
                # print pic1path
                # print pic2path
                # print labin[j]
    return input1,input2,labin



def main(self):
    img_input1 = tf.placeholder(tf.float32, [Batchsize, 192,64, 3])
    img_input2=tf.placeholder(tf.float32,[Batchsize,192,64,3])
    ground_truth_input = tf.placeholder(tf.int64, [Batchsize, 1], name='GroundTruthInput')
    y,predic=inf(img_input1,img_input2)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, STEPS / 3, 0.3, staircase=True)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                          labels=tf.squeeze(ground_truth_input,
                                                                                            squeeze_dims=[1]))))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(predic,tf.squeeze(ground_truth_input,squeeze_dims=[1]))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # gtsq=tf.squeeze(ground_truth_input,squeeze_dims=[1])

    saver=tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(STEPS):
            input1,input2,labin=get_batch(datapath)
            # get the batch and train one step
            _, loss_value, gs= sess.run([train_step, loss,global_step],
                                                         feed_dict={img_input1: input1,img_input2:input2,
                                                                    ground_truth_input: labin})

            print gs,loss_value

            if gs%100==0:
                test1,test2,testgt=get_batch(testpath)
                accu = sess.run([evaluation_step],feed_dict={img_input1:test1,img_input2:test2,ground_truth_input:testgt})
                print 'accuracy is',accu



if __name__ =='__main__':
    tf.app.run()
