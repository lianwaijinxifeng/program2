import tensorflow as tf


def slipnet(input,istraining,reuse=False,s=''):
    with tf.variable_scope(s+"conv1") as scope:
        # net=tf.contrib.layers.separable_conv2d(input,64,[3,3],depth_multiplier=1,activation_fn=tf.nn.leaky_relu,reuse=reuse,scope=scope)
        net = tf.contrib.layers.conv2d(input, 64, [3, 3], activation_fn=tf.nn.leaky_relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        net = tf.contrib.layers.batch_norm(net, is_training=istraining, scope=scope, reuse=reuse)

    with tf.variable_scope(s+"conv2") as scope:
        # net=tf.contrib.layers.separable_conv2d(net,128,[3,3],depth_multiplier=1,activation_fn=tf.nn.leaky_relu,reuse=reuse,scope=scope)

        net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.leaky_relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        net = tf.contrib.layers.batch_norm(net, is_training=istraining, scope=scope, reuse=reuse)

    with tf.variable_scope(s+"conv3") as scope:
        # net=tf.contrib.layers.separable_conv2d(net,256,[3,3],depth_multiplier=1,activation_fn=tf.nn.leaky_relu,reuse=reuse,scope=scope)
        #
        net = tf.contrib.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.leaky_relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.batch_norm(net, is_training=istraining, scope=scope, reuse=reuse)

    with tf.variable_scope(s+"conv4") as scope:
        # net=tf.contrib.layers.separable_conv2d(net,256,[3,3],depth_multiplier=1,activation_fn=tf.nn.leaky_relu,reuse=reuse,scope=scope)

        net = tf.contrib.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.leaky_relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.batch_norm(net, is_training=istraining, scope=scope, reuse=reuse)

    with tf.variable_scope(s+"conv5") as scope:
        # net=tf.contrib.layers.separable_conv2d(net,256,[3,3],depth_multiplier=1,activation_fn=tf.nn.leaky_relu,reuse=reuse,scope=scope)

        net = tf.contrib.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.leaky_relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        net = tf.contrib.layers.batch_norm(net, is_training=istraining, scope=scope, reuse=reuse)

    with tf.variable_scope(s+"conv6") as scope:
        # net=tf.contrib.layers.separable_conv2d(net,512,[3,3],depth_multiplier=1,activation_fn=tf.nn.leaky_relu,reuse=reuse,scope=scope)

        net = tf.contrib.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.leaky_relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        # net=tf.contrib.layers.batch_norm(net,is_training=istraining,scope=scope,reuse=reuse)

    with tf.variable_scope(s+"conv7") as scope:
        # net=tf.contrib.layers.separable_conv2d(net,512,[3,3],depth_multiplier=1,activation_fn=tf.nn.leaky_relu,reuse=reuse,scope=scope)

        net = tf.contrib.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.leaky_relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        # net=tf.contrib.layers.batch_norm(net,is_training=istraining,scope=scope,reuse=reuse)

    with tf.variable_scope(s+"conv8") as scope:
        # net=tf.contrib.layers.separable_conv2d(net,512,[3,3],depth_multiplier=1,activation_fn=tf.nn.leaky_relu,reuse=reuse,scope=scope)

        net = tf.contrib.layers.conv2d(net, 512, [3, 3], stride=1, activation_fn=tf.nn.leaky_relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

    with tf.variable_scope(s+"fc1") as scope:
        net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.leaky_relu, padding='VALID',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        # net=tf.nn.dropout(net,keep_prob=0.5)
    with tf.variable_scope(s+"fc2") as scope:
        net = tf.contrib.layers.conv2d(net, 128, [1, 1], activation_fn=tf.nn.leaky_relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        # net=tf.nn.dropout(net,keep_prob=0.5)
    # with tf.variable_scope("fc3") as scope:
    #     net = tf.contrib.layers.conv2d(net, 512, [1, 1], activation_fn=tf.nn.leaky_relu, padding='SAME',
    #                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #                                    scope=scope, reuse=reuse)
    shp = net.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    net = tf.reshape(net, [-1, flattened_shape], name='resh')
    return net


def sameornot(input, istraining,reuse=False):
    with tf.name_scope("model"):
        net0,net1=tf.split(input,num_or_size_splits=2,axis=1)
        net0=slipnet(net0,istraining,reuse,'net0/')
        net1=slipnet(net1,istraining,reuse,'net1/')
        # net2=slipnet(net2,istraining,reuse,'net2/')
        # print tf.shape(net0)
    net=tf.stack([net0,net1],axis=1)
    # print tf.shape(net)
    shp = net.get_shape()
    flattened_shape = shp[1].value * shp[2].value
    net = tf.reshape(net, [-1, flattened_shape], name='resh_fi')

    return net



def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keep_dims=True))
        y=tf.cast(y,tf.float32)
        loss1= y * tf.square(d)
        loss2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
        return tf.reduce_mean(loss1 + loss2) /2,d
