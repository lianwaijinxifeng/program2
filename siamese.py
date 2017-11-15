import tensorflow as tf
import inference
import os
import random as raandom
import math
import numpy as np
from PIL import Image
from pylab import *

model_path='./model/'
model_name='first.ckpt'
logdir='./log_2p'

Batchsize=100
LEARNING_RATE_BASE=0.01
STEPS=100000
margin=3

datapath='./peddatatr'
testpath='./peddatate'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def fc_layer(input,name,n_out):
    n_in=input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        biases=tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
        activation = tf.nn.leaky_relu(tf.matmul(input, kernel) + biases)
        # activation=tf.nn.relu_layer(input,kernel,biases,name=scope)
        return activation

# def inf(im1,im2,istraining):
#     mod1=inference.sameornot(im1,istraining,reuse=False)
#     mod2=inference.sameornot(im2,istraining,reuse=True)
#     net=tf.concat([mod1,mod2],3)
#     # net = tf.contrib.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.leaky_relu, padding='SAME',weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope='conv6')
#     # net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
#     # net = tf.contrib.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.leaky_relu, padding='SAME',weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope='conv7')
#     # net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
#     # net = tf.contrib.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.leaky_relu, padding='SAME',weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope='conv8')
#     # net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
#
#     # net=tf.contrib.layers.conv2d(net,4096,[3,1],activation_fn=tf.nn.leaky_relu,padding='VALID',weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope='fc1')
#     # net=tf.contrib.layers.conv2d(net,4096,[1,1],activation_fn=tf.nn.leaky_relu,padding='SAME',weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope='fc2')
#     # net=tf.contrib.layers.conv2d(net,512,[1,1],activation_fn=tf.nn.leaky_relu,padding='SAME',weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope='fc3')
#     # net=tf.contrib.layers.conv2d(net,2,[1,1],activation_fn=tf.nn.leaky_relu,padding='SAME',weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope='fc4')
#     shp=net.get_shape()
#     flattened_shape=shp[1].value*shp[2].value*shp[3].value
#     net=tf.reshape(net,[-1,flattened_shape],name='resh')
#     softmax=tf.nn.softmax(net)
#     predictions=tf.argmax(softmax,1)
#     return softmax,predictions

def get_batch(path):
    input1 = np.zeros([Batchsize, 192, 64, 3])
    input2 = np.zeros([Batchsize, 192, 64, 3])
    labin = np.zeros([Batchsize, 1])
    seqdir = os.listdir(path)
    numseq = len(seqdir)
    # seqnum = raandom.randrange(numseq)
    # targetpath = path + '/' + seqdir[seqnum]
    # targetlst = os.listdir(targetpath)
    # numtar = len(targetlst)
    # get one batch
    for j in range(Batchsize):
        if j % 2 == 0:
            seqnum = raandom.randrange(numseq)
            targetpath = path + '/' + seqdir[seqnum]
            targetlst = os.listdir(targetpath)
            numtar = len(targetlst)


            tarnum = raandom.randrange(numtar)
            picpath = targetpath + '/' + targetlst[tarnum]
            # get two pic in this path, resize and add to batch
            picdir = os.listdir(picpath)
            numpic = len(picdir)
            picnum1 = raandom.randrange(numpic)
            picnum2 = raandom.randrange(numpic)
            pic1path = picpath + '/' + picdir[picnum1]
            pic2path = picpath + '/' + picdir[picnum2]
            img1 = array(Image.open(pic1path),dtype=float)
            img2 = array(Image.open(pic2path),dtype=float)
            img1 -= np.mean(img1)
            img2 -= np.mean(img2)
            input1[j] = img1
            input2[j] = img2
            labin[j] = 1
            # print pic1path
            # print pic2path
            # print labin[j]


        else:
            seqnum1 = raandom.randrange(numseq)
            targetpath1 = path + '/' + seqdir[seqnum1]
            targetlst1 = os.listdir(targetpath1)
            numtar1 = len(targetlst1)
            seqnum2 = raandom.randrange(numseq)
            targetpath2 = path + '/' + seqdir[seqnum2]
            targetlst2 = os.listdir(targetpath2)
            numtar2 = len(targetlst2)

            tarnum1 = raandom.randrange(numtar1)
            picpath1 = targetpath1 + '/' + targetlst1[tarnum1]
            tarnum2 = raandom.randrange(numtar2)
            picpath2 = targetpath2 + '/' + targetlst2[tarnum2]


            # get two pic in these two pathes, resize and add to batch, don't forget to check if tarnum1==tarnum2
            picdir1 = os.listdir(picpath1)
            numpic1 = len(picdir1)
            picdir2 = os.listdir(picpath2)
            numpic2 = len(picdir2)
            picnum1 = raandom.randrange(numpic1)
            picnum2 = raandom.randrange(numpic2)
            pic1path = picpath1 + '/' + picdir1[picnum1]
            pic2path = picpath2 + '/' + picdir2[picnum2]
            img1 = array(Image.open(pic1path),dtype=float)
            img2 = array(Image.open(pic2path),dtype=float)
            img1 -= np.mean(img1)
            img2-=np.mean(img2)
            input1[j] = img1
            input2[j] = img2
            if tarnum1 == tarnum2 and seqnum1==seqnum2:
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
    ist=tf.placeholder(tf.bool)
    ground_truth_input = tf.placeholder(tf.int64, [Batchsize, 1], name='GroundTruthInput')

    mod1=inference.sameornot(img_input1,ist,reuse=False)
    mod2=inference.sameornot(img_input2,ist,reuse=True)
    # y,predic=inf(img_input1,img_input2,ist)


    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, STEPS / 3, 0.3, staircase=True)
    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
    #                                                                       labels=tf.squeeze(ground_truth_input,
    #                                                                                         squeeze_dims=[1]))))

    loss,d=inference.contrastive_loss(mod1,mod2,ground_truth_input,margin)

    tf.summary.scalar('loss',loss)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss,global_step=global_step)

    with tf.name_scope('evaluation'):
        # correct_prediction = tf.equal(predic,tf.squeeze(ground_truth_input,squeeze_dims=[1]))
        prediction=tf.concat([d,tf.constant(0.5*margin, shape=[100,1], dtype=tf.float32)],1)

        prediction=tf.argmax(prediction,1)
        correct_prediction=tf.equal(prediction,tf.squeeze(ground_truth_input,squeeze_dims=[1]))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accu',evaluation_step)
    # gtsq=tf.squeeze(ground_truth_input,squeeze_dims=[1])
    merged=tf.summary.merge_all()


    saver=tf.train.Saver()
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
        test_writer=tf.summary.FileWriter(logdir+'/test')
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(STEPS):
            input1,input2,labin=get_batch(datapath)
            # get the batch and train one step
            istrain=True
            summary,_, loss_value, gs= sess.run([merged,train_step, loss,global_step],
                                                         feed_dict={img_input1: input1,img_input2:input2,ist:istrain,
                                                                    ground_truth_input: labin})
            train_writer.add_summary(summary,i)

            print gs,loss_value

            if gs%100==0:
                accu=np.zeros([10])
                for index in range(10):
                    test1,test2,testgt=get_batch(testpath)
                    istrain=False
                    summary,accu[index] = sess.run([merged,evaluation_step],feed_dict={img_input1:test1,img_input2:test2,ist:istrain,ground_truth_input:testgt})
                    test_writer.add_summary(summary,i)
                accu=np.mean(accu)
                print 'accuracy is',accu
            if gs % 1000 == 0:
                saver.save(sess, os.path.join(model_path, model_name), global_step=gs)
        train_writer.close()
        test_writer.close()
        # saver.save(sess, os.path.join(model_path, model_name), global_step=gs)



if __name__ =='__main__':
    tf.app.run()
