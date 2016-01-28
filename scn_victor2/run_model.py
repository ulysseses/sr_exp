from __future__ import division, absolute_import, print_function
from six.moves import range, zip

import re
import os
import time
from datetime import datetime
import numpy as np
import scipy.misc as sm
import tensorflow as tf

from utils import preproc, tools

FLAGS = tf.app.flags.FLAGS
from scn_victor2 import model


def eval_epoch(Xs, Ys, y, sess, stream, cropw):
    """
    Evaluate the model against a dataset, and return the PSNR.

    Args:
      Xs: example placeholders list
      Ys: label placeholders list
      y: model output tensor
      sess: session
      stream: DataStream for the dataset
      cropw: crop border
    Returns:
      psnr: PSNR of model's inference on dataset
    """
    se = 0.
    for X_c, y_c in stream.get_epoch_iterator():
        y_c = y_c[:, cropw:-cropw, cropw:-cropw]
        chunk_size = X_c.shape[0]
        gpu_chunk = chunk_size // FLAGS.num_gpus
        dict_input1 = [(Xs[i], X_c[i*gpu_chunk : \
                                   ((i + 1)*gpu_chunk) \
                                   if (i != FLAGS.num_gpus - 1) \
                                   else chunk_size]) \
                       for i in range(FLAGS.num_gpus)]
        dict_input2 = [(Ys[i], y_c[i*gpu_chunk : \
                                   ((i + 1)*gpu_chunk) \
                                   if (i != FLAGS.num_gpus - 1) \
                                   else chunk_size]) \
                       for i in range(FLAGS.num_gpus)]
        feed = dict(dict_input1 + dict_input2)
        y_eval = sess.run(y, feed_dict=feed)
        se += np.sum((y_eval - y_c) ** 2.0)
    rmse = np.sqrt(se / (stream.dataset.num_examples * y_c.shape[1] * y_c.shape[2]))
    psnr = 20 * np.log10(1.0 / rmse)
    return psnr


def train(conf, ckpt=None):
    """
    Train model for a number of steps.
    
    Args:
      conf: configuration dictionary
      ckpt: restore from ckpt
    """
    cropw = conf['cropw']
    mb_size = conf['mb_size']
    path_tmp = conf['path_tmp']
    n_epochs = conf['n_epochs']
    cw = conf['cw']
    grad_norm_thresh = conf['grad_norm_thresh']

    tools.reset_tmp(path_tmp)

    # Prepare data
    tr_stream, te_stream = tools.prepare_data(conf)
    n_tr = tr_stream.dataset.num_examples
    n_te = te_stream.dataset.num_examples

    with tf.Graph().as_default(), tf.device('/cpu:0' if FLAGS.dev_assign else None):
        # Exponential decay learning rate
        global_step = tf.get_variable('global_step', [],
            initializer=tf.constant_initializer(0), dtype=tf.int32,
            trainable=False)
        lr = tools.exp_decay_lr(global_step, n_tr, conf)

        # Create an optimizer that performs gradient descent
        opt = tf.train.AdamOptimizer(lr)

        # Placeholders
        Xs = [tf.placeholder(tf.float32, [None, cw, cw, 1], name='X_%02d' % i) \
              for i in range(FLAGS.num_gpus)]
        Ys = [tf.placeholder(tf.float32, [None, cw - 2*cropw, cw - 2*cropw, 1],
                             name='Y_%02d' % i) \
              for i in range(FLAGS.num_gpus)]

        # Calculate the gradients for each model tower
        tower_grads = []
        y_splits = []
        for i in range(FLAGS.num_gpus):
            with tf.device(('/gpu:%d' % i) if FLAGS.dev_assign else None):
                with tf.name_scope('%s_%02d' % (FLAGS.tower_name, i)) as scope:
                    # Calculate the loss for one tower. This function constructs
                    # the entire model but shares the variables across all towers.
                    y_split = model.inference(Xs[i], conf)
                    y_splits.append(y_split)
                    total_loss = model.loss(y_split, Ys[i], conf, scope)

                    # Calculate the gradients for the batch of data on this tower.
                    gvs = opt.compute_gradients(total_loss)

                    # Optionally clip gradients.
                    if grad_norm_thresh > 0:
                        gvs = tools.clip_by_norm(gvs, grad_norm_thresh)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(gvs)
                    
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summs = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

        y = tf.concat(0, y_splits, name='y')

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        gvs = tools.average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_grad_op = opt.apply_gradients(gvs, global_step=global_step)

        # Add a summary to track the learning rate.
        summs.append(tf.scalar_summary('learning_rate', lr))

        # Add histograms for gradients.
        for g, v in gvs:
            if g:
                v_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', v.op.name)
                summs.append(tf.histogram_summary(v_name + '/gradients', g))

        # Tensorflow boilerplate
        sess, saver, summ_writer, summ_op = tools.tf_boilerplate(summs, conf, ckpt)

        # Baseline error
        bpsnr_tr = tools.baseline_psnr(tr_stream)
        bpsnr_te = tools.baseline_psnr(te_stream)
        print('approx baseline psnr_tr=%.3f' % bpsnr_tr)
        print('approx baseline psnr_te=%.3f' % bpsnr_te)

        # Train
        format_str = ('%s| %04d PSNR=%.3f (Tr: %.1fex/s; %.1fs/batch)'
                      '(Te: %.1fex/s; %.1fs/batch)')
        step = 0
        for epoch in range(n_epochs):
            print('--- Epoch %d ---' % epoch)
            # Training
            for X_c, y_c in tr_stream.get_epoch_iterator():
                y_c = y_c[:, cropw:-cropw, cropw:-cropw]
                chunk_size = X_c.shape[0]
                gpu_chunk = chunk_size // FLAGS.num_gpus
                dict_input1 = [(Xs[i], X_c[i*gpu_chunk : \
                                           ((i + 1)*gpu_chunk) \
                                           if (i != FLAGS.num_gpus - 1) \
                                           else chunk_size]) \
                               for i in range(FLAGS.num_gpus)]
                dict_input2 = [(Ys[i], y_c[i*gpu_chunk : \
                                           ((i + 1)*gpu_chunk) \
                                           if (i != FLAGS.num_gpus - 1) \
                                           else chunk_size]) \
                               for i in range(FLAGS.num_gpus)]
                feed = dict(dict_input1 + dict_input2)
                
                start_time = time.time()
                sess.run(apply_grad_op, feed_dict=feed)
                duration_tr = time.time() - start_time

                if step % 20 == 0:
                    feed2 = dict(dict_input1)
                    
                    start_time = time.time()
                    y_eval = sess.run(y, feed_dict=feed2)
                    duration_eval = time.time() - start_time
                    
                    psnr = tools.eval_psnr(y_c, y_eval)
                    ex_per_step_tr = mb_size * FLAGS.num_gpus / duration_tr
                    ex_per_step_eval = mb_size * FLAGS.num_gpus / duration_eval
                    print(format_str % (datetime.now().time(), step, psnr,
                          ex_per_step_tr, float(duration_tr / FLAGS.num_gpus),
                          ex_per_step_eval, float(duration_eval / FLAGS.num_gpus)))

                if step % 100 == 0:
                    summ_str = sess.run(summ_op, feed_dict=feed)
                    summ_writer.add_summary(summ_str, step)

                if step % 400 == 0:
                    saver.save(sess, os.path.join(path_tmp, 'ckpt'),
                        global_step=step)

                step += 1

            # Evaluation
            psnr_tr = eval_epoch(Xs, Ys, y, sess, tr_stream, cropw)
            psnr_te = eval_epoch(Xs, Ys, y, sess, te_stream, cropw)
            print('approx psnr_tr=%.3f' % psnr_tr)
            print('approx psnr_te=%.3f' % psnr_te)
            saver.save(sess, os.path.join(path_tmp, 'ckpt'),
                       global_step=step)            

        saver.save(sess, os.path.join(path_tmp, 'ckpt'),
                   global_step=step)
        tr_stream.close()
        te_stream.close()


def infer(img, Xs, y, sess, conf, save=None):
    """
    Upsample with our neural network.

    Args:
      img: image to upsample
      Xs: input placeholders list
      y: model inference
      sess: session
      conf: configuration dictionary
      save: optional save path
    Returns:
      hr: inferred image
    """
    cw = conf['cw']
    stride = cw // 2
    mb_size = conf['mb_size']
    path_tmp = conf['path_tmp']
    cropw = conf['cropw']

    # Bi-cubic up-sample and pre-process
    start_time0 = time.time()
    lr_ycc = preproc.rgb2ycc(img)
    lr_y = preproc.byte2unit(lr_ycc[:, :, 0])
    h0, w0 = lr_y.shape
    lr_y = preproc.padcrop(lr_y, cw)
    h1, w1 = lr_y.shape

    # Fill into a data array
    n_y, n_x = preproc._num_crops(lr_y, cw, stride, tup=True)
    crops_in = np.empty((n_y*n_x, cw, cw, 1), dtype='float32')
    for i, crop in enumerate(preproc._crop_gen(lr_y, cw, stride)):
        crops_in[i] = crop[..., np.newaxis]

    # Infer
    crops_out = np.empty((n_y*n_x, cw - 2*cropw, cw - 2*cropw, 1), dtype='float32')
    start_time1 = time.time()
    for i in range(0, n_y*n_x, FLAGS.num_gpus * mb_size):
        X_c = crops_in[i : i + FLAGS.num_gpus * mb_size]
        chunk_size= X_c.shape[0]
        gpu_chunk = chunk_size // FLAGS.num_gpus
        dict_input1 = [(Xs[j], X_c[j*gpu_chunk : \
                                   ((j + 1)*gpu_chunk) \
                                   if (j != FLAGS.num_gpus - 1) \
                                   else chunk_size]) \
                       for j in range(FLAGS.num_gpus)]
        feed = dict(dict_input1)
        tmp = sess.run(y, feed_dict=feed)
        crops_out[i : i + FLAGS.num_gpus * mb_size] = tmp
    gpu_time = time.time() - start_time1
    
    # Fill crops into y channel
    h2 = cw-2*cropw + (n_y - 1)*stride
    w2 = cw-2*cropw + (n_x - 1)*stride
    hr_y = np.zeros((h2, w2), dtype='float32')
    mask = 1e-8 * np.ones_like(hr_y, dtype='float32')
    
    y = 0
    for i in range(n_y):
        x = 0
        for j in range(n_x):
            hr_y[y : y + cw-2*cropw, x : x + cw-2*cropw] += \
                crops_out[i*n_x + j, :, :, 0]
            mask[y : y + cw-2*cropw, x : x + cw-2*cropw] += 1.
            x += stride
        y += stride
    
    hr_y /= mask
    hr_y = preproc.unit2byte(hr_y)

    # Combine y with cb & cr, then convert to rgb
    hr_y = hr_y[:h0 - 2*cropw, :w0 - 2*cropw]
    hr_ycc = lr_ycc[cropw:-cropw, cropw:-cropw]
    hr_ycc[:, :, 0] = hr_y
    hr = preproc.ycc2rgb(hr_ycc)
    total_time = time.time() - start_time0
    
    print('total time: %.1f | gpu time: %.1f' % (total_time, gpu_time))

    # Save
    if save:
        sm.imsave(save, hr)

    return hr


def eval_te(conf, ckpt):
    """
    Evaluate against the entire test set of images.

    Args:
      conf: configuration dictionary
      ckpt: checkpoint path
    Returns:
      psnr: psnr of entire test set
    """
    cw = conf['cw']
    sr = conf['sr']
    cropw = conf['cropw']
    path_te = conf['path_te']
    fns_te = preproc._get_filenames(path_te)
    n = len(fns_te)

    with tf.Graph().as_default(), tf.device('/cpu:0' if FLAGS.dev_assign else None):
        # Placeholders
        Xs = [tf.placeholder(tf.float32, [None, cw, cw, 1], name='X_%02d' % i) \
              for i in range(FLAGS.num_gpus)]

        y_splits = []
        for i in range(FLAGS.num_gpus):
            with tf.device(('/gpu:%d' % i) if FLAGS.dev_assign else None):
                with tf.name_scope('%s_%02d' % (FLAGS.tower_name, i)) as scope:
                    y_split = model.inference(Xs[i], conf)
                    y_splits.append(y_split)
                    tf.get_variable_scope().reuse_variables()
        y = tf.concat(0, y_splits, name='y')
        
        # Restore
        saver = tf.train.Saver(tf.trainable_variables())
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        saver.restore(sess, ckpt)

        # Iterate over each image, and calculate error
        tmse = 0
        bl_tmse = 0
        for fn in fns_te:
            lr, gt = preproc.lr_hr(sm.imread(fn), sr)
            hr = infer(lr, Xs, y, sess, conf)
            # Evaluate
            gt = gt[cropw:-cropw, cropw:-cropw]
            diff = gt.astype(np.float32) - hr.astype(np.float32)
            mse = np.mean(diff ** 2)
            tmse += mse
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            lr = lr[cropw:-cropw, cropw:-cropw]
            bl_diff = gt.astype(np.float32) - lr.astype(np.float32)
            bl_mse = np.mean(bl_diff ** 2)
            bl_tmse += bl_mse
            bl_psnr = 20 * np.log10(255.0 / np.sqrt(bl_mse))
            
            print('hr PSNR: %.3f, lr PSNR % .3f for %s' % \
                (psnr, bl_psnr, fn.split('/')[-1]))
        rmse = np.sqrt(tmse / n)
        psnr = 20 * np.log10(255. / rmse)
        bl_rmse = np.sqrt(bl_tmse / n)
        bl_psnr = 20 * np.log10(255. / bl_rmse)
        print('total test PSNR: %.3f' % psnr)
        print('total baseline PSNR: %.3f' % bl_psnr)
        return psnr, bl_psnr
