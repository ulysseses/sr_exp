from __future__ import division, absolute_import, print_function
from six.moves import range, zip

import re
import numpy as np
import tensorflow as tf
from tensorflow.python.control_flow_ops import case

FLAGS = tf.app.flags.FLAGS


def _act_summ(x):
    """
    Helper to create summaries for activations.

    Create a histogram summary
    Create a sparsity summary

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _relu_std(fw, n_chans):
    """
    ReLU initialization based on "Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification" by Kaiming He et al.

    Args:
      fw: filter width
      n_chans: filter depth
    Returns:
      see below
    """
    return np.sqrt(2.0 / (fw*fw*n_chans))


def _arr_initializer(arr):
    ''' https://github.com/tensorflow/tensorflow/issues/434 '''
    def _initializer(_, dtype=tf.float32):
        return tf.constant(arr, dtype=dtype)
    return _initializer


def _conv_layer(shp, prev):
    """
    Helper function to create `conv = W*x + b` layer.

    Args:
      shp: shape of conv filter
      prev: input to conv layer
    Returns:
      bias: output of `W*x + b`
    """
    with tf.device('/cpu:0' if FLAGS.dev_assign else None):
        w = tf.get_variable('w', shp, tf.float32,
            tf.truncated_normal_initializer(stddev=_relu_std(shp[0], shp[2])))
        b = tf.get_variable('b', [shp[3]], dtype=tf.float32,
            initializer=tf.constant_initializer(0.))

    tf.add_to_collection('decay_vars', w)
    w_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', w.op.name)
    b_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', b.op.name)
    tf.histogram_summary(w_name, w)
    tf.histogram_summary(b_name, b)
    
    conv = tf.nn.conv2d(prev, w, [1, 1, 1, 1], 'SAME', name='conv_w')
    bias = tf.nn.bias_add(conv, b, name='conv_b')
    return bias


def _st(x, thresh, slope=None, name=None):
    """
    L1 Soft threshold operator.

    Args:
      x: input
      thresh: threshold
      slope: here for interfacing; used only for _gst
      name: name assigned to this operation
    Returns:
      soft threshold of `x`
    """
    with tf.name_scope(name):
        return tf.mul(tf.sign(x),
                      tf.nn.relu(tf.nn.bias_add(tf.abs(x), -thresh)))


def _gst_init(thresh, p):
    def actual_gst(y, l, J=2):
        tau = (2*l*(1-p))**(1/(2-p)) + l*p*((2*l*(1-p))**((p-1)/(2-p)))
        y_abs = np.abs(y)
        if y_abs <= tau:
            return 0.
        x = y_abs
        for j in range(J):
            x = y_abs - l*p*x**(p-1)
        return np.sign(y) * x

    tau0 = (2*thresh*(1-p))**(1/(2-p)) + thresh*p*((2*thresh*(1-p))**((p-1)/(2-p)))
    slope0 = actual_gst(tau0 + 2, thresh) - actual_gst(tau0 + 1, thresh)

    return tau0, slope0


def _gst(x, thresh, slope=None, name=None):
    """
    Learnable and approximate form of GST, with modifiable slope in addition to the
    modifiable thresh.

    Args:
      x: input
      thresh: threshold
      slope: here for interfacing; used only for _gst
      name: name assigned to this operation
    Returns:
      soft threshold of `x`
    """
    with tf.name_scope(name):
        return tf.mul(tf.sign(x),
                      tf.select(tf.less(tf.abs(x), thresh),
                      tf.zeros_like(x, dtype=tf.float32),
                      tf.nn.bias_add(tf.mul(slope, tf.abs(x)), thresh)))


def _lista(x, w_e, w_s, thresh, prox_op, T, slope=None):
    """
    Learned Iterative Shrinkage-Thresholding Algorithm (LISTA). LISTA is an
    approximately sparse encoder. It approximates (in an L2 sense) a sparse code
    of `x` according to dictionary `w_e`. Note that during backpropagation, `w_e`
    isn't strictly a dictionary (i.e. dictionary atoms are not strictly normalized).

    LISTA is a differentiable version of the iterative FISTA algorithm.

    Args:
      x: [n, n_f] tensor
      w_e: [n_f, n_c] encoder tensor
      w_s: [n_c, n_f] mutual inhibition tensor
      thresh: threshold
      prox_op: proximal operator
      T: number of iterations
      slope: here for interfacing; used only for _gst
    Returns:
      z: LISTA output
    """
    b = tf.matmul(x, w_e, name='b')
    z = prox_op(b, thresh, name='z0')
    for t in range(T):
        with tf.name_scope('itr_%02d' % t):
            c = b + tf.matmul(z, w_s, name='c')
            z = prox_op(c, thresh, slope=slope, name='z')
    return z


def _lcod(x, w_e, w_s, thresh, prox_op, T, slope=None):
    """
    Learned Coordinate Descent (LCoD). LCoD is an approximately sparse encoder. It
    approximates (in an L2 sense) a sparse code of `x` according to dictionary `w_e`.
    Note that during backpropagation, `w_e` isn't strictly a dictionary (i.e.
    dictionary atoms are not strictly normalized).

    LCoD is a differentiable version of greedy coordinate descent.

    Args:
      x: [n, n_f] tensor
      w_e: [n_f, n_c] encoder tensor
      w_s: [n_c, n_f] mutual inhibition tensor
      thresh: threshold
      prox_op: proximal operator
      T: number of iterations
      slope: here for interfacing; used only for _gst
    Returns:
      z: LCoD output
    """
    def f1():
        '''k == 0'''
        forget_z = tf.concat(1, [z_k, tf.zeros(tf.pack([n, n_c - 1]),
                                               dtype=tf.float32)],
                             name='forget_z')
        update_z = tf.concat(1, [z_bar_k, tf.zeros(tf.pack([n, n_c - 1]),
                                                   dtype=tf.float32)],
                             name='update_z')
        return forget_z, update_z

    def f2():
        '''k == n_c - 1'''
        forget_z = tf.concat(1, [tf.zeros(tf.pack([n, n_c - 1]),
                                          dtype=tf.float32),
                                 z_k],
                             name='forget_z')
        update_z = tf.concat(1, [tf.zeros(tf.pack([n, n_c - 1]),
                                          dtype=tf.float32),
                                 z_bar_k],
                             name='update_z')
        return forget_z, update_z

    def f3():
        '''k > 0 and k < n_c - 1'''
        forget_z = tf.concat(1, [tf.zeros(tf.pack([n, k]),
                                          dtype=tf.float32),
                                 z_k,
                                 tf.zeros(tf.pack([n, n_c - (k + 1)]),
                                          dtype=tf.float32)],
                             name='forget_z')
        update_z = tf.concat(1, [tf.zeros(tf.pack([n, k]),
                                          dtype=tf.float32),
                                 z_bar_k,
                                 tf.zeros(tf.pack([n, n_c - (k + 1)]),
                                          dtype=tf.float32)],
                             name='update_z')
        return forget_z, update_z

    n_c = w_s.get_shape().as_list()[0]
    b = tf.matmul(x, w_e, name='b0')  # [n, n_c]
    z = tf.zeros_like(b, dtype=tf.float32, name='z0')
    for t in range(T):
        with tf.name_scope('itr_%02d' % t):
            if t != T - 1:
                z_bar = prox_op(b, thresh, slope=slope, name='z_bar')
                # L1 norm greedy heuristic
                tmp = z_bar - z
                tmp2 = tf.abs(tmp)
                tmp3 = tf.reduce_sum(tmp2, 1)
                k = tf.argmax(tmp3, 0, name='k')  # tf.int32
                e = tf.slice(tmp, tf.pack([0, k]), [-1, 1], name='e')
                s_slice = tf.slice(w_s, tf.pack([0, k], [1, -1], name='s_slice'))
                b = tf.add(b, tf.matmul(e, s_slice), name='b')
                z_bar_k = tf.slice(z_bar, tf.pack([0, k]), [-1, 1], name='z_bar_k')
                z_k = tf.slice(z, tf.pack([0, k]), [-1, 1], name='z_k')
                forget_z, update_z = case({tf.equal(k, 0): f1, tf.equal(k, n_c - 1)},
                    default=f3, exclusive=False)
                z = tf.identity(z - forget_z + update_z, name='z')
            else:
                z = prox_op(b, thresh, slope=slope, name='z')
    return z


def inference(x, conf):
    """
    Sparse Coding Based Network for Super-Resolution. This network differs from SCN
    in 5 things:

    1. We try out LCoD instead of LISTA
    2. We try out an approximate form of non-fixed-point iteration GST instead of
      ST. GST is used in non-convex sparse coding.
    3. We use a residual-learning approach instead of normalizing input and
      de-normalizing output.
    4. We try out deep, non-linear convolutional layers instead of
      shallow, linear ones.
    5. We try to enforce Z sparsity with an additional L1 cost

    Args:
      x: input
      conf: configuration dictionary
    Returns:
      y: output
    """
    cw = conf['cw']
    n_layers = conf['n_layers']
    fw = conf['fw']
    n_chans = conf['n_chans']
    subnet_name = conf['subnet_name'].lower()
    prox_name = conf['prox_name'].lower()
    n_c = conf['n_c']
    T = conf['T']
    thresh0 = conf['thresh0']
    
    n_f = cw*cw*n_chans

    if subnet_name == 'lista':
        subnet = _lista
    elif subnet_name == 'lcod':
        subnet == _lcod
    else:
        raise ValueError('subnet_name must be "lista" or "lcod"')

    if prox_name == 'st':
        prox_op = _st
    elif prox_name == 'gst':
        p = conf['p']
        tau0, slope0 = _gst_init(thresh0, p)
        prox_op = _gst
    else:
        raise ValueError("prox_name must be 'st' or 'gst'")
    
    # Feature Extraction
    for i in range(n_layers):
        with tf.variable('extract_%02d' % i):
            if i == 0:
                shp = [fw, fw, 1, n_chans]
                prev = x
            else:
                shp = [fw, fw, n_chans, n_chans]
            cl = _conv_layer(shp, prev)
            prev = tf.nn.relu(cl, name='relu')
            _act_summ(prev)
    
    # Sub-network
    with tf.variable_scope(subnet_name):
        with tf.device('/cpu:0' if FLAGS.dev_assign else None):
            e = np.random.randn(n_f, n_c).astype(np.float32) * _relu_std(1, n_f)
            if subnet_name == 'lista':
                L = 5.0
                e /= L
                s = (np.eye(n_c) - e.T.dot(e) / L).astype(np.float32)
            else:
                s = (np.eye(n_c) - e.T.dot(e)).astype(np.float32)
            w_e = tf.get_variable('w_e',
                [n_f, n_c],
                dtype=tf.float32,
                initializer=_arr_initializer(e))
            w_s = tf.get_variable('w_s',
                [n_c, n_c],
                dtype=tf.float32,
                initializer=_arr_initializer(s))
            thresh = tf.get_variable('thresh',
                [n_c],
                dtype=tf.float32,
                initializer=tf.constant_initializer(
                    thresh0 if prox_name == 'st' else tau0))
            if prox_name == 'st':
                slope = None
            else:
                slope = tf.get_variable('slope',
                    [n_c],
                    dtype=tf.float32, trainable=True,
                    initializer=tf.constant_initializer(slope0))
            w_d = tf.get_variable('w_d',
                [n_c, n_f],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0., _relu_std(1, n_c)))

        tf.add_to_collection('decay_vars', w_e)
        tf.add_to_collection('decay_vars', w_s)
        tf.add_to_collection('decay_vars', w_d)

        w_e_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', w_e.op.name)
        w_s_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', w_s.op.name)
        thresh_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', thresh.op.name)
        w_d_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', w_d.op.name)
        tf.histogram_summary(w_conv_name, w_conv)
        tf.histogram_summary(w_e_name, w_e)
        tf.histogram_summary(w_s_name, w_s)
        tf.histogram_summary(thresh_name, thresh)
        if prox_name == 'gst':
            slope_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', slope.op.name)
            tf.histogram_summary(slope_name, slope)
        tf.histogram_summary(w_d_name, w_d)

        x1 = tf.reshape(prev, [-1, n_f]), name='x1')
        z = subnet(x1, w_e, w_s, thresh, prox_op, T, slope=slope)

        prev = tf.matmul(z, w_d, name='y0')
        prev = tf.reshape(prev, [-1, cw, cw, n_chans])

    # Reconstruction
    for i in range(n_layers):
        with tf.variable('recon_%02d' % i):
            if i == n_layers - 1:
                shp = [fw, fw, n_chans, 1]
            else:
                shp = [fw, fw, n_chans, n_chans]
            cl = _conv_layer(shp, prev)
            if i != n_layers - 1:
                prev = tf.nn.relu(cl, name='relu')
                _act_summ(prev)

    # Residual Learning
    y = tf.add(cl, x)
    y = tf.identity(y, name='y')
    return y


def loss(y, Y, conf, scope=None):
    """
    L2-loss model on top of the network raw output.
    
    Args:
      y: network output tensor
      Y: ground truth tensor
      conf: configuration dictionary
      scope: unique prefix string identifying the tower, e.g. 'tower_00'
    Returns:
      total_loss: total loss Tensor
    """
    l2_reg = conf['l2_reg']
    sq_loss = tf.nn.l2_loss(y - Y, name='sq_loss')
    tf.add_to_collection('losses', sq_loss)
    if l2_reg > 0:
        with tf.name_scope('decay'):
            for decay_var in tf.get_collection('decay_vars', scope=scope):
                weight_decay = tf.mul(tf.nn.l2_loss(decay_var), l2_reg,
                                      name=decay_var.op.name)
                tf.add_to_collection('losses', weight_decay)
        total_loss = tf.add_n(tf.get_collection('losses', scope=scope),
                              name='total_loss')
    else:
        total_loss = sq_loss

    # Add loss summaries
    for loss in tf.get_collection('losses', scope=scope) + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', loss.op.name)
        tf.scalar_summary(loss_name, loss)

    return total_loss
