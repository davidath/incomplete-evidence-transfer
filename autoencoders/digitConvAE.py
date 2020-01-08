#############################################################################
# Convolutional autoencoder (Used in MNIST dataset)
# For convinience: Conv0,1..N in ini represent Convolutional layers
#                  DeConv0,1,...N in ini represent Deconvolutional layers
#                  Encoder0,1,...N in ini represent FC-layers after Conv
#############################################################################

import tensorflow as tf
from utils import get_act_dictionary
from tensorflow.contrib.layers import xavier_initializer as xavier_init
import numpy as np
from tqdm import *


# Build Encoder of ConvAE

def build_encoder(incoming, cp, SEED, scope='px_encoder', reuse=False):
    # Get activation ditionary
    act_dict = get_act_dictionary()

    # Enable/Disable sharing weights
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope(scope):
        # Reshape input to [samples, height, width, channels]
        try:
            channels = cp.getint('Input', 'Channels')
            height = width = int(
                np.sqrt(cp.getint('Input', 'Width') / channels))
            cae = tf.reshape(
                incoming, shape=[-1, height, width, channels])
        except:
            raise ValueError('width x height matrix must be square')
        # Add convolutional layers
        for sect in [i for i in cp.sections() if 'Conv' in i]:
            cae = tf.layers.conv2d(cae,
                                   filters=cp.getint(sect, 'Filters'),
                                   kernel_size=cp.getint(
                                       sect, 'Fsize'),
                                   strides=cp.getint(sect, 'Stride'),
                                   padding=cp.get(sect, 'Pad'),
                                   activation=act_dict['ReLU'],
                                   name='Pre_' + sect,
                                   data_format='channels_last')
            cae = tf.contrib.layers.batch_norm(cae,
                                               scope=sect.split('Conv')[1])

        # Store shape for later reshaping
        last_shape = cae.shape[1:]
        cae = tf.layers.flatten(cae)
        keep_prob = float(cp.get('Dropout', 'Rate'))
        cae = tf.layers.dropout(cae, rate=keep_prob)
        # Most inner layer of AE / Last layer of encoder
        sect = 'Encoder0'
        cae = tf.layers.dense(cae,
                              cp.getint(sect, 'Width'),
                              activation=act_dict[cp.get(
                                  sect, 'Activation')],
                              name='Pre_' + sect,
                              kernel_initializer=xavier_init
                              (uniform=False, seed=SEED)
                              )
        return cae, last_shape


# Build Decoder of ConvAE


def build_decoder(incoming, cp, SEED, last_shape,
                  scope='px_decoder', reuse=False):
    # Get activation ditionary
    act_dict = get_act_dictionary()

    # Enable/Disable sharing weights
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope(scope):
        cae = incoming
        for sect in [i for i in cp.sections() if 'Encoder' in i]:
            if sect != 'Encoder0':
                cae = tf.layers.dense(cae,
                                      cp.getint(sect, 'Width'),
                                      activation=act_dict[cp.get(
                                          sect, 'Activation')],
                                      name='Pre_' + sect,
                                      kernel_initializer=xavier_init
                                      (uniform=False, seed=SEED)
                                      )

        # Reshape to [sample, width, height, channels]
        cae = tf.reshape(cae, shape=[-1, last_shape[0], last_shape[1],
                                     last_shape[2]])
        decon_lst = [i for i in cp.sections() if 'DeCon' in i]
        decon_num = len(decon_lst)
        # Add deconvolutional layers
        for sect in decon_lst:
            cae = tf.layers.conv2d_transpose(cae,
                                             filters=cp.getint(
                                                 sect, 'Filters'),
                                             kernel_size=cp.getint(
                                                 sect, 'Fsize'),
                                             strides=cp.getint(sect, 'Stride'),
                                             padding=cp.get(sect, 'Pad'),
                                             activation=act_dict['ReLU'],
                                             name='Pre_De_' + sect,
                                             data_format='channels_last')
            # Return last layer before reconstruction
            if sect == decon_lst[decon_num - 2]:
                prev = cae
                prev = tf.contrib.layers.max_pool2d(cae, (2,2))
                prev = tf.layers.flatten(prev)
        # Flatten output
        cae = tf.layers.flatten(cae)
        return cae, prev

# Reconstruction loss (build mean squared error loss)


def recon_loss(X, out, scope):
    # Initialize loss
    with tf.name_scope(scope):
        ae_lr = tf.placeholder(tf.float32, shape=[])
        ae_mse = tf.reduce_mean(tf.square(X - out), name='ae_mse')
        # Optimizer
        ae_optimizer = tf.train.MomentumOptimizer(learning_rate=ae_lr,
                                                  momentum=0.9)
        ae_t_op = ae_optimizer.minimize(ae_mse)
    return ae_mse, ae_lr, ae_t_op


# Evidence transfer cross entropy loss


def cond_loss(Q, k, vl, e_id):
    with tf.name_scope('cond_loss' + str(e_id)):
        cond_lr = tf.placeholder(tf.float32, shape=[])
        approx = tf.keras.losses.categorical_crossentropy(k, Q)
        approx = tf.reduce_mean(approx)
        cond_loss = approx
        cond_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        cond_t_op = cond_opt.minimize(cond_loss, var_list=vl)
    return cond_lr, cond_loss, cond_t_op

# Multiple cross entropy combined loss for each additional evidence


def EviTRAM_loss(X, out, Z, Qs, ks, vl):
    px_mse, px_lr, px_t_op = recon_loss(X, out, 'RECN')
    with tf.name_scope('multi_cond'):
        multi_loss = tf.constant(0.0)
        for pos, Q in enumerate(Qs):
            cond_lr, cond_l, cond_t_op = cond_loss(Qs[pos], ks[pos], vl, pos)
            multi_loss += cond_l
        multi_loss = tf.constant(0.25) * tf.reduce_mean(multi_loss)
        multi_loss = px_mse + multi_loss
        multi_loss = tf.reduce_mean(multi_loss)
        #  multi_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        multi_opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        multi_t_op = multi_opt.minimize(multi_loss, var_list=vl)
    return multi_loss, cond_lr, multi_t_op, px_mse

# Build layers and losses for ConvAE (Pre EviTRAM)


def build_px(cp, SEED):
    # Initialize I/O tensors
    conv_In = tf.placeholder(tf.float32,
                             shape=[None, cp.getint('Input', 'Width')],
                             name='conv_IN')
    # Building network
    conv_Z, last_shape = build_encoder(conv_In, cp, SEED)
    conv_Xrec, conv_Prev = build_decoder(conv_Z, cp, SEED, last_shape)

    # Building losses
    px_mse, px_lr, px_t_op = recon_loss(conv_In, conv_Xrec, scope='RECON_LOSS')

    # Create / Return tensor dictionary
    ret_dict = {'conv_in': conv_In, 'conv_z': conv_Z, 'conv_out': conv_Xrec,
                'conv_prev': conv_Prev, 'px_train': px_t_op, 'px_mse': px_mse,
                'px_lr': px_lr}
    return ret_dict

# Build layers and losses for ConvAE (EviTRAM)


def build_EviTRAM(cp, SEED):
    # Initialize I/O tensors
    conv_In = tf.placeholder(tf.float32,
                             shape=[None, cp.getint('Input', 'Width')],
                             name='conv_IN')

    # Initiliaze placeholders for each source of evidence
    sect = 'Experiment'
    ev_paths = [cp.get(sect, i) for i in cp.options(sect) if 'evidence' in i]
    ks_IN = []
    for ev_path_id, ev_path in enumerate(ev_paths):
        ks_IN.append(tf.placeholder(tf.float32, shape=[
            None, cp.getint('Q' + str(ev_path_id), 'Width')],
            name='k_IN' + str(ev_path_id)))

    # Building network
    conv_Z, last_shape = build_encoder(conv_In, cp, SEED)
    conv_Xrec, conv_Prev = build_decoder(conv_Z, cp, SEED, last_shape)

    # Initialize additional prediction layer to minimize cross entropy, 
    # for each source of evidence
    Qs = []
    for ev_path_id, ev_path in enumerate(ev_paths):
        with tf.name_scope('COND' + str(ev_path_id)):
            # Get activation ditionary
            act_dict = get_act_dictionary()
            sect = 'Q' + str(ev_path_id)
            Q = tf.layers.dense(conv_Xrec,
                                cp.getint(sect, 'Width'),
                                activation=act_dict[cp.get(
                                    sect, 'Activation')],
                                name='Pre_' + sect,
                                kernel_initializer=xavier_init(uniform=False,
                                                               seed=SEED),
                                reuse=tf.AUTO_REUSE)
            Qs.append(Q)

    TV = [v for v in tf.trainable_variables() if 'Pre_' in v.name or
              'beta' in v.name]

    # Building loss of EviTRAM
    cond_loss, cond_lr, cond_t_op, px_mse = EviTRAM_loss(conv_In, conv_Xrec,
                                                 conv_Z, Qs, ks_IN, TV)

    ret_dict = {'conv_in': conv_In, 'conv_z': conv_Z, 'conv_out': conv_Xrec,
                'conv_prev': conv_Prev, 'Qs': Qs, 'ks_IN': ks_IN, 'TV': TV,
                'evitram_t_op': cond_t_op, 'evitram_loss': cond_loss, 
                'px_mse': px_mse, 'evitram_lr': cond_lr}
    
    return ret_dict




# Perform training loop, called from train.py


def train(train_dict, pae_dict):
    # Initialize training parameters
    maxepochs = train_dict['cp'].getint('Hyperparameters', 'MaxEpochs')
    batch_size = train_dict['cp'].getint('Hyperparameters', 'BatchSize')
    lr_decay = train_dict['cp'].getint('Hyperparameters', 'DecayEpoch')
    ae_lr = float(train_dict['cp'].get('Hyperparameters', 'LearningRate'))

    # Epoch loop
    try:
        for epoch in range(maxepochs + 1):
            # Batch loop
            for row in tqdm(xrange(0, train_dict['data'].shape[0], batch_size), ascii=True):
                # Get batch
                idx = slice(row, row + batch_size)
                X_batch = train_dict['data'][idx]

                # Forward and Backprop
                train_dict['sess'].run(pae_dict['px_train'],
                         feed_dict={pae_dict['conv_in']: X_batch,
                                    pae_dict['px_lr']: ae_lr})

            _ll, summary_str = train_dict['sess'].run([pae_dict['px_mse'], train_dict['sumr']],
                                        feed_dict={pae_dict['conv_in']: \
                                            X_batch})

            train_dict['fw'].add_summary(summary_str, epoch)

            # Save condition
            if epoch % 100 == 0:
                save_path = train_dict['saver'].save(train_dict['sess'], 
                                                        train_dict['savestr'])
            if epoch == lr_decay:
                ae_lr = ae_lr / 10.0
            save_path = train_dict['saver'].save(train_dict['sess'], 
                                                        train_dict['savestr'])
    except KeyboardInterrupt:
        save_path = train_dict['saver'].save(train_dict['sess'], 
                                                        train_dict['savestr'])

# Perform training loop, called from train.py

def evitram_train(td, evitramd):
    # Initialize experiment parameters
    maxepochs = td['cp'].getint('Hyperparameters', 'MaxEpochs')
    batch_size = td['cp'].getint('Hyperparameters', 'BatchSize')
    lr_decay = td['cp'].getint('Hyperparameters', 'DecayEpoch')

    # Epoch loop
    try:
        for epoch in range(maxepochs + 1):
            # Batch loop
            for row in tqdm(xrange(0, td['data'].shape[0], batch_size), ascii=True):
                # Get batch
                idx = slice(row, row + batch_size)
                X_batch = td['data'][idx]
                fd = {}
                fd[evitramd['conv_in']] = X_batch
                for ev_path_id, ev_path in enumerate(td['ev_paths']):
                    fd[evitramd['ks_IN'][ev_path_id]] = td['EV'][ev_path_id][idx]
                td['sess'].run(evitramd['evitram_t_op'], feed_dict=fd)

            _ll, rrec,  summary_str = td['sess'].run([evitramd['evitram_loss'],
                                                      evitramd['px_mse'],
                                                      td['sumr']],
                                                   feed_dict=fd)
            #  print epoch, mse_ll, p, rrec, llo
            td['fw'].add_summary(summary_str, epoch)

            #  Save condition
            if epoch % 100 == 0:
                save_path = td['saver'].save(td['sess'], td['savestr'])
        save_path = td['saver'].save(td['sess'], td['savestr'])
    except KeyboardInterrupt:
        save_path = td['saver'].save(td['sess'], td['savestr'])

