from __future__ import division, absolute_import, print_function
from six.moves import range, zip

import re
import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops

FLAGS = tf.app.flags.FLAGS


def _init_shift(n_f, pw):
    """
    Initialize shift filter.

    Args:
      n_f: input dimension
      pw: patch width
    Returns:
      shift: shift convolutional filter
    """
    shift = np.zeros((pw, pw, n_f, pw*pw), dtype='float32')
    for f in range(n_f):
        ind = 0
        for i in range(pw):
            for j in range(pw):
                shift[i, j, f, ind] = 1.
                ind += 1
    return shift


def _init_stitch(pw):
    """
    Initialize stitch filter.

    Args:
      pw: patch width
    Returns:
      stitch: stitch convolutional filter
    """
    stitch = np.zeros((pw, pw, 1, pw*pw), dtype='float32')
    ind = 0
    for i in range(0, pw):
        for j in range(0, pw):
            stitch[pw - i - 1, pw - j - 1, 0, ind] = 1. / (pw*pw)
            ind += 1
    return stitch


def _init_mean(mw):
    """
    Initialize mean filter.

    Args:
      mw: mean filter width
    Returns:
      mean: mean filter
    """
    mean = np.ones((mw, mw, 1, 1), dtype='float32') / (mw*mw)
    return mean


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


def _st(x, thresh, name='st'):
    """
    L1 Soft threshold operator.

    Args:
      x: input
      thresh: threshold variable
      name: name assigned to this operation
    Returns:
      soft threshold of `x`
    """
    with tf.name_scope('st'):
        return tf.mul(tf.sign(x), tf.nn.relu(tf.nn.bias_add(tf.abs(x), -thresh)))


def _gst_init(l, p, J=2):
    """
    Function to initialize GST tau/slope sensibly.

    argmin(x) (1/2)||y - x||^2 + l*|x|^p

    Args:
      l: lambda parameter
      p: non-convexity (0 <= p)
      J: number of GST iterations to run
    Returns:
      tau0: initial x-intercept value
      y0: initial y-intercept value
      slope0: initial gst slope value
    """
    def actual_gst(y):
        tau = (2*l*(1-p))**(1/(2-p)) + l*p*((2*l*(1-p))**((p-1)/(2-p)))
        y_abs = np.abs(y)
        if y_abs <= tau:
            return 0.
        x = y_abs
        for j in range(J):
            x = y_abs - l*p*x**(p-1)
        return np.sign(y) * x

    tau0 = (2*l*(1-p))**(1/(2-p)) + l*p*((2*l*(1-p))**((p-1)/(2-p)))
    y0 = actual_gst(tau0 + 1e-4)
    slope0 = actual_gst(tau0 + 2) - actual_gst(tau0 + 1)

    return tau0, y0, slope0


def _gst(x, tau, y, slope, name='gst'):
    """
    Learnable and approximate form of GST, with modifiable slope in addition to the
    modifiable tau.

    Args:
      x: input
      tau: tau variable
      y: y-intercept variable
      slope: slope variable
      name: name assigned to this operation
    Returns:
      generalized soft threshold of `x`
    """
    with tf.name_scope(name):
        return tf.mul(tf.sign(x),
                      tf.select(tf.less(tf.abs(x), tau),
                      tf.zeros_like(x, dtype=tf.float32),
                      tf.nn.bias_add(tf.mul(slope, tf.abs(x)), y)))


def _lista(x, w_e, w_s, prox_op, T, *prox_vars):
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
      prox_op: proximal operator
      T: number of iterations
      prox_vars: variables passed to prox_op
    Returns:
      z: LISTA output
    """
    b = tf.matmul(x, w_e, name='b')
    with tf.name_scope('itr_00'):
        z = prox_op(b, *prox_vars, name='z')
    for t in range(1, T+1):
        with tf.name_scope('itr_%02d' % t):
            c = b + tf.matmul(z, w_s, name='c')
            z = prox_op(c, *prox_vars, name='z')
    return z


def _lcod(x, w_e, w_s, prox_op, T, *prox_vars):
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
      prox_op: proximal operator
      T: number of iterations
      prox_vars: variables passed to prox_op
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
    
    n = tf.identity(tf.shape(x)[0], name='n')
    n_c = tf.to_int32(w_s.get_shape().as_list()[0], name='n_c')
    with tf.name_scope('itr_00'):
        b = tf.matmul(x, w_e, name='b')  # [n, n_c]
        z = tf.zeros_like(b, dtype=tf.float32, name='z')  # [n, n_c]
    for t in range(1, T+1):
        with tf.name_scope('itr_%02d' % t):
            if t != T:
                z_bar = prox_op(b, *prox_vars, name='z_bar')  # [n, n_c]
                # L1 norm greedy heuristic
                with tf.name_scope('greedy_heuristic'):
                    tmp = z_bar - z  # [n, n_c]
                    tmp2 = tf.abs(tmp)  # [n, n_c]
                    tmp3 = tf.reduce_sum(tmp2, 0)  # [n_c]
                    k = tf.to_int32(tf.argmax(tmp3, 0, name='k'))  # tf.int32
                    e = tf.slice(tmp, tf.pack([0, k]), tf.pack([n, 1]),
                                 name='e')  # [n, 1]
                with tf.name_scope('update_b'):
                    s_slice = tf.slice(w_s, tf.pack([k, 0]), tf.pack([1, n_c]),
                                       name='s_slice')  # [1, n_c]
                    b = tf.add(b, tf.matmul(e, s_slice), name='b')  # [n, n_c]
                with tf.name_scope('update_z'):
                    z_bar_k = tf.slice(z_bar, tf.pack([0, k]), tf.pack([n, 1]),
                                       name='z_bar_k')  # [n, 1]
                    z_k = tf.slice(z, tf.pack([0, k]), tf.pack([n, 1]),
                                   name='z_k')  # [n, 1]
                    tup = control_flow_ops.case({tf.equal(k, 0): f1,
                                                 tf.equal(k, n_c - 1): f2},
                                                default=f3,
                                                exclusive=False)  # [n, n_c]
                    forget_z, update_z = tup
                    z = tf.add_n([z, -forget_z, update_z], name='z')  # [n, n_c]
            else:
                z = prox_op(b, *prox_vars, name='z')
    return z


def inference(x, conf):
    """
    Sparse Coding Based Network for Super-Resolution. This is a convolutional
    neural network formulation of Coupled Dictionary Learning. Features are
    extracted via convolutional filters. Overlapping patches of the feature maps
    are obtained via a layer of patch extraction convolutional filters. The
    resulting feature maps are normalized and fed through LISTA sub-network of
    `T` iterations. The LISTA output patches are de-normalized with `scale_norm`
    and stitched back into their respective positions to re-construct the final
    output image.

    Args:
      x: input layer
      conf: configuration dictionary
    Returns:
      y: output
    """
    fw = conf['fw']
    pw = conf['pw']
    ps = conf['ps']
    n_chans = conf['n_chans']
    n_c = conf['n_c']
    T = conf['T']
    cropw = conf['cropw']
    res = conf['res']
    mw = conf['mw']
    thresh0 = conf['thresh0']
    subnet_name = conf['subnet_name'].lower()
    prox_name = conf['prox_name'].lower()
    low_rank = conf['low_rank']
    
    n_f = n_chans*pw*pw

    # Determine subnet and proximal operators
    if subnet_name == 'lista':
        subnet = _lista
    elif subnet_name == 'lcod':
        subnet = _lcod
    else:
        raise ValueError('subnet_name must be "lista" or "lcod"')

    # Determine proximal operator
    prox_vars = []
    with tf.device('/cpu:0' if FLAGS.dev_assign else None):
        if prox_name == 'st':
            prox_op = _st
            thresh = tf.get_variable('thresh',
                [n_c],
                dtype=tf.float32,
                initializer=tf.constant_initializer(thresh0))
            prox_vars.append(thresh)
        elif prox_name == 'gst':
            prox_op = _gst
            p = conf['p']
            tau0, y0, slope0 = _gst_init(thresh0, p)
            tau = tf.get_variable('tau',
                [n_c],
                dtype=tf.float32,
                initializer=tf.constant_initializer(tau0))
            y_intercept = tf.get_variable('y_intercept',
                [n_c],
                dtype=tf.float32,
                initializer=tf.constant_initializer(y0))
            slope = tf.get_variable('slope',
                [n_c],
                dtype=tf.float32,
                initializer=tf.constant_initializer(slope0))
            prox_vars.append(tau)
            prox_vars.append(y_intercept)
            prox_vars.append(slope)
        else:
            raise ValueError('prox_name must be "st" or "gst"')
    for prox_var in prox_vars:
        name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', prox_var.op.name)
        tf.histogram_summary(name, prox_var)

    # Get dimensions
    with tf.name_scope('dimensions'):
        bs, h, w = [tf.shape(x)[i] for i in range(3)]
        bs = tf.identity(bs, name='bs')
        h = tf.identity(h, name='h')
        w = tf.identity(w, name='w')

    # Initialize constant filters
    with tf.variable_scope('const_filt'):
        w_shift1 = tf.constant(_init_shift(n_chans, pw), name='w_shift1')
        w_shift2 = tf.constant(_init_shift(1, pw), name='w_shift2')
        w_stitch = tf.constant(_init_stitch(pw), name='w_stitch')
        w_mean = tf.constant(_init_mean(mw), name='w_mean')
    
    # Initialize feature extraction filters
    with tf.device('/cpu:0' if FLAGS.dev_assign else None):
        w_conv = tf.get_variable('w_conv', [fw, fw, 1, n_chans], tf.float32,
            initializer=tf.truncated_normal_initializer(0., _relu_std(fw, 1)))
    tf.add_to_collection('decay_vars', w_conv)
    name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', w_conv.op.name)
    tf.histogram_summary(name, w_conv)
    
    # Feature Extraction
    # [bs, h, w, 1] -> [bs, h, w, n_chans]
    x_conv = tf.nn.conv2d(x, w_conv, [1, 1, 1, 1], 'SAME', name='x_conv')

    if not res:
        # Normalize each patch
        # [bs, h, w, n_chans] -> [bs, h, w, n_chans]
        x_b4_shift = tf.nn.l2_normalize(x_conv, 3, name='x_normed')
    else:
        # Residual training; don't need normalization
        x_b4_shift = x_conv

    # Shift with pw*pw dirac delta filters to create overlapping patches.
    # Only obtain patches every `ps` strides.
    # A patch is resembled as the flattened array along the last dimension.
    # [bs, h, w, n_chans] --> [bs, h//ps, w//ps, n_chans*pw*pw]
    x_shift = tf.nn.depthwise_conv2d(x_b4_shift, w_shift1, [1, ps, ps, 1], 'SAME',
        name='x_shift')
    
    # 4D tensor -> matrix
    # [bs, h//ps, w//ps, n_chans*pw*pw] -> [bs*(h//ps)*(w//ps), n_chans*pw*pw]
    x_in = tf.reshape(x_shift, [-1, n_f], name='x_in')

    # Feed into sub-network
    with tf.variable_scope(subnet_name):
        with tf.device('/cpu:0' if FLAGS.dev_assign else None):
            # Initial values
            e = np.random.randn(n_f, n_c).astype(np.float32) * _relu_std(1, n_f)
            if subnet_name == 'lista':
                L = 5.
                e /= L
                s = (np.eye(n_c) - e.T.dot(e) / L).astype(np.float32)
            else:
                s = (np.eye(n_c) - e.T.dot(e)).astype(np.float32)
            # Encoder
            w_e = tf.get_variable('w_e',
                [n_f, n_c],
                dtype=tf.float32,
                initializer=_arr_initializer(e))
            # S matrix
            if low_rank > 0:
                u_svd, s_svd, vh_svd = np.linalg.svd(s)
                u_svd1 = u_svd[:, :low_rank] * np.sqrt(s_svd[:low_rank])
                vh_svd1 = (vh_svd[:low_rank, :].T * np.sqrt(s_svd[:low_rank])).T
                u_svd1 = u_svd1.astype(np.float32)
                vh_svd1 = vh_svd1.astype(np.float32)
                w_s1 = tf.get_variable('w_s1',
                    [n_c, low_rank],
                    dtype=tf.float32,
                    initializer=_arr_initializer(u_svd1))
                w_s2 = tf.get_variable('w_s2',
                    [low_rank, n_c],
                    dtype=tf.float32,
                    initializer=_arr_initializer(vh_svd1))
                w_s = tf.matmul(w_s1, w_s2, name='w_s')
            else:
                w_s = tf.get_variable('w_s',
                    [n_c, n_c],
                    dtype=tf.float32,
                    initializer=_arr_initializer(s))
            # Decoder
            w_d = tf.get_variable('w_d',
                [n_c, pw*pw],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0., _relu_std(1, n_c)))

        # Sub-network
        # [bs*(h//ps)*(w//ps), n_f] -> [bs*(h//ps)*(w//ps), n_c]
        z = subnet(x_in, w_e, w_s, prox_op, T, *prox_vars)
        tf.add_to_collection('l1_decay', z)

        # Decode
        # [bs*(h//ps)*(w//ps), n_c] -> [bs*(h//ps)*(w//ps), pw*pw]
        y_out = tf.matmul(z, w_d, name='y_out')

        for var in [w_e, w_s, w_d]:
            name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', var.op.name)
            tf.histogram_summary(name, var)
            tf.add_to_collection('decay_vars', var)

    # matrix --> 4D tensor
    # [bs*(h//ps)*(w//ps), pw*pw] -> [bs, h//ps, w//ps, pw*pw]
    y_out = tf.reshape(y_out, tf.pack([bs, h // ps, w // ps, pw*pw]))

    if not res:
        # Obtain the norm for each overlapping patch of x
        with tf.name_scope('denorm'):
            scale_norm = 1.1
            x_shift2 = tf.nn.conv2d(x, w_shift2, [1, ps, ps, 1], 'SAME',
                                    name='x_shift2')
            x_norm = tf.pow(tf.reduce_sum(tf.pow(x_shift2, 2), 3, True), 0.5,
                            name='x_norm')
            y_unit = tf.nn.l2_normalize(y_out, 3, name='y_unit')
            prev = tf.mul(y_unit, x_norm * scale_norm, name='y_denorm')
    else:
        # Residual training
        prev = y_out

    # Average overlapping images together
    with tf.name_scope('overlap_avg'):
        mask_input = tf.ones(tf.pack([bs, h // ps, w // ps, pw*pw]),
                             dtype=tf.float32)
        mask = tf.nn.deconv2d(mask_input, w_stitch,
            tf.pack([bs, h, w, 1]), [1, ps, ps, 1], 'SAME', name='mask')
        y_stitch = tf.nn.deconv2d(prev, w_stitch, tf.pack([bs, h, w, 1]),
            [1, ps, ps, 1], 'SAME')
        y_stitch = tf.div(y_stitch, mask + 1e-8, name='y_stitch')

    if not res:
        # Add the mean filter response of x to y
        with tf.name_scope('mean_skip'):
            x_mean = tf.nn.conv2d(x, w_mean, [1, 1, 1, 1], 'SAME', name='x_mean')
            prev = tf.add(y_stitch, x_mean)
    else:
        # Add x to y
        with tf.name_scope('res_skip'):
            prev = tf.add(y_stitch, x)

    # Crop to remove convolution boundary effects
    with tf.variable_scope('crop'):
        crop_begin = [0, cropw, cropw, 0]
        crop_size = tf.pack([-1, h - 2*cropw, w - 2*cropw, 1])
        y_crop = tf.slice(prev, crop_begin, crop_size, name='y_crop')

    y = tf.identity(y_crop, name='y')
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
    which_gpu = int(scope[-3:-1])
    l1_reg = conf['l1_reg']
    l2_reg = conf['l2_reg']
    sq_loss = tf.nn.l2_loss(y - Y, name='sq_loss')
    tf.add_to_collection('losses', sq_loss)
    if l1_reg > 0:
        with tf.name_scope('l1_decay'):
            z = tf.get_collection('l1_decay')[which_gpu]
            l1_decay = tf.mul(tf.reduce_sum(tf.abs(z)), l1_reg, name='l1')
            tf.add_to_collection('losses', l1_decay)
    if l2_reg > 0:
        with tf.name_scope('l2_decay'):
            for decay_var in tf.get_collection('decay_vars')[which_gpu*4:]:
                weight_decay = tf.mul(tf.nn.l2_loss(decay_var), l2_reg,
                                      name=decay_var.op.name)
                tf.add_to_collection('losses', weight_decay)

    total_loss = tf.add_n(tf.get_collection('losses', scope=scope),
                          name='total_loss')

    # Add loss summaries
    for loss in tf.get_collection('losses', scope=scope) + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', loss.op.name)
        tf.scalar_summary(loss_name, loss)

    return total_loss
