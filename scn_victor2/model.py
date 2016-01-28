from __future__ import division, absolute_import, print_function
from six.moves import range, zip

import re
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def _shape(t, static=True):
    """
    Helper function to obtain the static/dynamic shape of tensor `t`.

    Args:
      t: tensor
      static (True): static/dynamic
    Returns:
      static or dynamic shape of `t`
    """
    if static:
        return t.get_shape().as_list()
    else:
        return [tf.shape(t)[i] for i in range(t.get_shape().ndims)]


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


def _st(x, thresh, name=None):
    """
    L1 Soft threshold operator.

    Args:
      x: input
      thresh: threshold
      name: name assigned to this operation
    Returns:
      soft threshold of `x`
    """
    return tf.mul(tf.sign(x), tf.nn.relu(tf.nn.bias_add(tf.abs(x), -thresh)),
                  name=name)


def _lista(x, w_e, w_s, thresh, prox_op, T):
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
    Returns:
      z: LISTA output
    """
    b = tf.matmul(x, w_e, name='b')
    z = prox_op(b, thresh, name='z0')
    for t in range(T):
        with tf.name_scope('itr_%02d' % t):
            c = b + tf.matmul(z, w_s, name='c')
            z = prox_op(c, thresh, name='z')
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
    L = 5
    scale_norm = 1.1
    
    fw = conf['fw']
    pw = conf['pw']
    mw = conf['mw']
    cropw = conf['cropw']
    n_chans = conf['n_chans']
    n_c = conf['n_c']
    T = conf['T']
    thresh0 = conf['thresh0']

    n_f = pw*pw*n_chans
    
    
    with tf.name_scope('shapes'):
        bs, h, w = _shape(x, static=False)[:-1]
        bs = tf.identity(bs, name='bs')
        h = tf.identity(h, name='h')
        w = tf.identity(w, name='w')

    # Initialize constants
    with tf.variable_scope('const_filt'):
        w_shift1 = tf.constant(_init_shift(n_chans, pw), name='w_shift1')
        w_shift2 = tf.constant(_init_shift(1, pw), name='w_shift2')
        w_stitch = tf.constant(_init_stitch(pw), name='w_stitch')
        w_mean = tf.constant(_init_mean(mw), name='w_mean')

    with tf.device('/cpu:0' if FLAGS.dev_assign else None):
        w_conv = tf.get_variable('w_conv', [fw, fw, 1, n_chans], tf.float32,
            initializer=tf.truncated_normal_initializer(0., _relu_std(fw, 1)))
    tf.add_to_collection('decay_vars', w_conv)
    w_conv_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', w_conv.op.name)
    tf.histogram_summary(w_conv_name, w_conv)

    # Variable convolutional layer: extract features
    x_conv = tf.nn.conv2d(x, w_conv, [1, 1, 1, 1], 'SAME', name='X_conv')

    # Constant convolutional layer: shift to create overlap
    # A 1D vector along the last dimension of `X_overlap` represents
    # a flattened patch.
    # Each of these patches overlaps one another with stride=1.
    x_shift1 = tf.nn.depthwise_conv2d(x_conv, w_shift1, [1, 2, 2, 1], 'SAME',
                                      name='X_shift1')

    # L2 normalization: normalize w.r.t. last dimension
    # This effectively normalizes each patch
    x_unit = tf.nn.l2_normalize(x_shift1, 3, name='X_unit')

    # Feed into LISTA
    with tf.variable_scope('lista'):
        with tf.device('/cpu:0' if FLAGS.dev_assign else None):
            e = np.random.randn(n_f, n_c).astype(np.float32) * _relu_std(1, n_f)
            e /= L
            s = (np.eye(n_c) - e.T.dot(e) / L).astype(np.float32)
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
                initializer=tf.constant_initializer(thresh0))
            w_d = tf.get_variable('w_d',
                [n_c, pw*pw],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0., _relu_std(1, n_c)))

        tf.add_to_collection('decay_vars', w_e)
        tf.add_to_collection('decay_vars', w_s)
        tf.add_to_collection('decay_vars', w_d)

        w_e_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', w_e.op.name)
        w_s_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', w_s.op.name)
        thresh_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', thresh.op.name)
        w_d_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', w_d.op.name)
        tf.histogram_summary(w_e_name, w_e)
        tf.histogram_summary(w_s_name, w_s)
        tf.histogram_summary(thresh_name, thresh)
        tf.histogram_summary(w_d_name, w_d)

        x_unit = tf.reshape(x_unit, [-1, n_f])
        z = _lista(x_unit, w_e, w_s, thresh, _st, T)

        y0 = tf.matmul(z, w_d, name='y0')
        y0 = tf.reshape(y0, tf.pack([-1, h // 2, w // 2, pw*pw]))

    # Obtain the norm for each overlapping patch of x
    with tf.name_scope('denorm'):
        x_shift2 = tf.nn.conv2d(x, w_shift2, [1, 2, 2, 1], 'SAME', name='X_shift2')
        x_norm = tf.pow(tf.reduce_sum(tf.pow(x_shift2, 2), 3, True), 0.5, name='X_norm')
        y_unit = tf.nn.l2_normalize(y0, 3, name='y_unit')
        y_denorm = tf.mul(y_unit, x_norm * scale_norm, name='y_denorm')

    # Average overlaping images together
    mask = tf.nn.deconv2d(tf.ones_like(y_denorm, dtype=tf.float32), w_stitch,
        tf.pack([bs, h, w, 1]), [1, 2, 2, 1], 'SAME', name='mask')
    y_stitch = tf.nn.deconv2d(y_denorm, w_stitch, tf.pack([bs, h, w, 1]), [1, 2, 2, 1],
                              'SAME', name='y_stitch')
    y_stitch = tf.div(y_stitch, mask + 1e-8)

    # Add the mean filter response of X to y
    x_mean = tf.nn.conv2d(x, w_mean, [1, 1, 1, 1], 'SAME', name='X_mean')
    y_out = tf.add(y_stitch, x_mean, name='y_out')

    # Crop to remove convolution boundary effects
    with tf.variable_scope('crop'):
        crop_begin = tf.convert_to_tensor([0, cropw, cropw, 0], dtype='int32',
                                          name='begin')
        crop_size = tf.pack([-1, h - 2*cropw, w - 2*cropw, 1])
        y_crop = tf.slice(y_out, crop_begin, crop_size, name='y_crop')

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
