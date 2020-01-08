#############################################################################
# Stacked autoencoder (Used in CIFAR-10, REUTERS and 20newsgroups)
# For convinience: Encoder0,1, ... N are encoder hidden layers
#                  Decoder0,1, ... N are decoder hidden layers
#                  Input0,1, ... N are the input placeholders
#############################################################################

import tensorflow as tf
from utils import get_act_dictionary
from utils import log
from tensorflow import truncated_normal_initializer as normal_init
from tensorflow.contrib.layers import xavier_initializer as xavier_init
import numpy as np
from tqdm import *


# Create autoencoder networks from configuration file for greedy layerwise
# training


def build_ae(incoming, num, cp, SEED, reuse=False):
    act_dict = get_act_dictionary()
    enclabel = 'Encoder' + str(num)
    declabel = 'Decoder' + str(num)
    # Enable/Disable sharing weights
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope(enclabel):
        # Input
        encoder = incoming
        # Dropout
        encoder = tf.layers.dropout(
            encoder, float(cp.get('Dropout', 'rate')), seed=SEED)
        # Hidden Layer(s)
        for sect in [i for i in cp.sections() if enclabel in i]:
            if cp.get('Experiment', 'PREFIX') == 'reu100k':
                encoder = tf.layers.dense(encoder,
                                          cp.getint(sect, 'Width'),
                                          activation=act_dict[cp.get(
                                              sect, 'Activation')],
                                          name='Pre_' + sect,
                                          kernel_initializer=xavier_init
                                          (uniform=False, seed=SEED)
                                          )
            else:
                encoder = tf.layers.dense(encoder,
                                          cp.getint(sect, 'Width'),
                                          activation=act_dict[cp.get(
                                              sect, 'Activation')],
                                          name='Pre_' + sect,
                                          kernel_initializer=normal_init(mean=0.0,
                                                                         stddev=0.01,
                                                                         seed=SEED)
                                          )
    with tf.name_scope(declabel):
        decoder = encoder
        # Decoder Layer(s)
        for sect in [i for i in cp.sections() if declabel in i]:
            if cp.get('Experiment', 'PREFIX') == 'reu100k':
                decoder = tf.layers.dense(decoder,
                                          cp.getint(sect, 'Width'),
                                          activation=act_dict[cp.get(
                                              sect, 'Activation')],
                                          name='Pre_' + sect,
                                          kernel_initializer=xavier_init
                                          (uniform=False, seed=SEED)
                                          )
            else:
                decoder = tf.layers.dense(decoder,
                                          cp.getint(sect, 'Width'),
                                          activation=act_dict[cp.get(
                                              sect, 'Activation')],
                                          name='Pre_' + sect,
                                          kernel_initializer=normal_init(mean=0.0,
                                                                         stddev=0.01,
                                                                         seed=SEED)
                                          )
    return decoder, encoder


# Create stacked end to end autoencoder


def build_stacked(incoming, cp, SEED, reuse=False):
    # Enable/Disable sharing weights
    if reuse:
        tf.get_variable_scope().reuse_variables()
    # Loop through encoders
    for enc in [i for i in cp.sections() if 'Encoder' in i]:
        num = enc.split('Encoder')[1]
        act_dict = get_act_dictionary()
        enclabel = 'Encoder' + str(num)
        with tf.name_scope(enclabel):
            # Input
            if num == '0':
                sda = incoming
                if cp.get('Experiment', 'PREFIX') == '20ng':
                    sda = tf.contrib.layers.batch_norm(sda, scope='Pre_INBN',
                                                       reuse=tf.AUTO_REUSE)
                sda = tf.layers.dropout(
                    sda, float(cp.get('Dropout', 'rate')), seed=SEED)
            # Hidden layers
            for sect in [i for i in cp.sections() if enclabel in i]:
                sda = tf.layers.dense(sda,
                                      cp.getint(sect, 'Width'),
                                      activation=act_dict[cp.get(
                                          sect, 'Activation')],
                                      name='Pre_' + sect)
    encoder = sda
    # Loop through Decoders
    for dec in reversed([i for i in cp.sections() if 'Decoder' in i]):
        num = dec.split('Decoder')[1]
        declabel = 'Decoder' + str(num)
        maxaenum = cp.getint('Experiment', 'AENUM')
        with tf.name_scope(declabel):
            # End to enc autoencoder
            for sect in [i for i in cp.sections() if declabel in i]:
                sda = tf.layers.dense(sda,
                                      cp.getint(sect, 'Width'),
                                      activation=act_dict[cp.get(
                                          sect, 'Activation')],
                                      name='Pre_' + sect)
                if sect == 'Decoder1':
                    prev = sda
    return sda, encoder, prev


# Create reconstruction loss for layerwise autoencoders


def build_layer_loss(X, out, num, scope):
    aeid = 'AE' + num
    # Initialize each loss
    with tf.name_scope(scope):
        ae_lr = tf.placeholder(tf.float32, shape=[])
        ae_mse = tf.reduce_mean(tf.square(X - out), name=aeid + '_mse')
        # Optimizer
        ae_optimizer = tf.train.MomentumOptimizer(
            learning_rate=ae_lr, momentum=0.9)
        vl = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='Pre_Encoder' + num + '|Pre_Decoder' + num)
        ae_t_op = ae_optimizer.minimize(ae_mse, var_list=vl)
    return ae_mse, ae_lr, ae_t_op

# Create reconstruction loss for stacked autoencoder (end-to-end finetune)


def build_stacked_loss(X, out, scope='sda_loss'):
    # Initialize loss
    with tf.name_scope(scope):
        sda_lr = tf.placeholder(tf.float32, shape=[])
        sda_mse = tf.reduce_mean(tf.square(X - out), name='sda_mse')
        # Optimizer
        sda_optimizer = tf.train.MomentumOptimizer(
            learning_rate=sda_lr, momentum=0.9)
        vl = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='Pre_Encoder|Pre_Decoder')
        sda_t_op = sda_optimizer.minimize(sda_mse, var_list=vl)
    return sda_lr, sda_mse, sda_t_op


# Build pre EviTRAM aka create layerwise and stacked autoencoders


def build_px(cp, ae_ids, SEED):
    # Initialize list of tensors that will be used to identify each layerwise
    # autoencoder during training
    l_in_tensors = []
    l_hidden_tensors = []
    l_out_tensors = []
    l_train_tensors = []
    l_lr_tensors = []
    l_mse_tensors = []

    # Loop over all autoencoders
    for num in ae_ids:
        # Autoencoder id label
        aeid = 'AE' + num
        # Input placeholder initialization
        X = tf.placeholder(tf.float32, shape=[None, cp.getint('Input' + num, 'Width')],
                           name='IN' + num)
        l_in_tensors.append(X)

        # Initialize each layerwise network
        with tf.variable_scope(tf.get_variable_scope()):
            out, hidden = build_ae(X, num, cp, SEED)
        l_hidden_tensors.append(hidden)
        l_out_tensors.append(out)

        # Initiliaze layerwise losses
        ae_mse, ae_lr, ae_t_op = build_layer_loss(
            X, out, num, aeid + '_loss')

        l_mse_tensors.append(ae_mse)
        l_lr_tensors.append(ae_lr)
        l_train_tensors.append(ae_t_op)

    ############   Start of Stacked autoencoder initialization    ############

    # Input placeholder initialization
    sda_X = tf.placeholder(tf.float32, shape=[None, cp.getint('Input', 'Width')],
                           name='IN')
    # Network
    with tf.variable_scope(tf.get_variable_scope()):
        sda_out, sda_hidden, sda_prev = build_stacked(
            sda_X, cp, SEED, reuse=True)

    # Loss
    sda_lr, sda_mse, sda_t_op = build_stacked_loss(sda_X, sda_out)

    ############    End of Stacked autoencoder initialization     ############

    # Create return dictionary
    ret_dict = {'l_in_tensors': l_in_tensors, 'l_hidden_tensors': l_hidden_tensors,
                'l_out_tensors': l_out_tensors, 'l_train_tensors': l_train_tensors,
                'l_lr_tensors': l_lr_tensors, 'l_mse_tensors': l_mse_tensors,
                'sda_in': sda_X, 'sda_hidden': sda_hidden, 'sda_out': sda_out,
                'sda_train': sda_t_op, 'sda_lr': sda_lr,  'sda_mse': sda_mse,
                'sda_prev': sda_prev}
    return ret_dict


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


# Reconstruction loss (build mean squared error loss)


def disj_recon_loss(X, out, vl, scope):
    # Initialize loss
    with tf.name_scope(scope):
        ae_lr = tf.placeholder(tf.float32, shape=[])
        ae_mse = tf.reduce_mean(tf.square(X - out), name='ae_mse')
        # Optimizer
        ae_optimizer = tf.train.MomentumOptimizer(learning_rate=ae_lr,
                                                  momentum=0.9)
        ae_t_op = ae_optimizer.minimize(ae_mse, var_list=vl)
    return ae_mse, ae_lr, ae_t_op


# Evidence transfer cross entropy loss


def cond_loss(Q, k, vl, e_id):
    with tf.name_scope('cond_loss' + str(e_id)):
        cond_lr = tf.placeholder(tf.float32, shape=[])
        approx = tf.keras.losses.categorical_crossentropy(k, Q)
        approx = tf.reduce_mean(approx)
        cond_loss = approx
        cond_opt = tf.train.GradientDescentOptimizer(learning_rate=cond_lr)
        cond_t_op = cond_opt.minimize(cond_loss, var_list=vl)
    return cond_lr, cond_loss, cond_t_op

# Multiple cross entropy combined loss for each additional evidence


def EviTRAM_loss(X, out, Z, Qs, ks, vl, cmse=0.5, ccond=1.0):
    px_mse, px_lr, px_t_op = recon_loss(X, out, 'RECN')
    px_mse = tf.constant(cmse) * px_mse
    with tf.name_scope('multi_cond'):
        multi_loss = tf.constant(0.0)
        for pos, Q in enumerate(Qs):
            cond_lr, cond_l, cond_t_op = cond_loss(Qs[pos], ks[pos], vl, pos)
            multi_loss += cond_l
        multi_loss = tf.constant(ccond) * tf.reduce_mean(multi_loss)
        multi_loss = px_mse + multi_loss
        multi_loss = tf.reduce_mean(multi_loss)
        multi_opt = tf.train.GradientDescentOptimizer(learning_rate=cond_lr)
        multi_t_op = multi_opt.minimize(multi_loss, var_list=vl)
    return multi_loss, cond_lr, multi_t_op, px_mse

# Multiple cross entropy combined loss for each additional evidence
# disjoint


def disj_EviTRAM_loss(X, out, Z, Qs, ks, vl1, vl2):
    px_mse, px_lr, px_t_op = disj_recon_loss(X, out, vl1, 'RECN')
    with tf.name_scope('multi_cond'):
        multi_loss = tf.constant(0.0)
        for pos, Q in enumerate(Qs):
            cond_lr, cond_l, cond_t_op = cond_loss(Qs[pos], ks[pos], vl2, pos)
            multi_loss += cond_l
        multi_loss = tf.reduce_mean(multi_loss)
        multi_opt = tf.train.GradientDescentOptimizer(learning_rate=cond_lr)
        multi_t_op = multi_opt.minimize(multi_loss, var_list=vl2)
    return multi_loss, cond_lr, multi_t_op, px_mse, px_lr, px_t_op


# Build stacked autoencoder for EviTRAM

def build_EviTRAM(cp, ae_ids, SEED):
    sae_dict = build_px(cp, ae_ids, SEED)

    # Initiliaze placeholders for each source of evidence
    sect = 'Experiment'
    ev_paths = [cp.get(sect, i) for i in cp.options(sect) if 'evidence' in i]
    ks_IN = []
    for ev_path_id, ev_path in enumerate(ev_paths):
        ks_IN.append(tf.placeholder(tf.float32, shape=[
            None, cp.getint('Q' + str(ev_path_id), 'Width')],
            name='k_IN' + str(ev_path_id)))

    # Initialize additional prediction layer to minimize cross entropy,
    # for each source of evidence
    Qs = []
    for ev_path_id, ev_path in enumerate(ev_paths):
        with tf.name_scope('COND' + str(ev_path_id)):
            # Get activation ditionary
            act_dict = get_act_dictionary()
            sect = 'Q' + str(ev_path_id)
            if cp.get('Experiment', 'PREFIX') == 'CIFAR':
                Q = tf.layers.dense(sae_dict['sda_prev'],
                                    sae_dict['sda_prev'].shape.as_list()[
                                    1] * 0.2,
                                    activation=act_dict['ReLU'],
                                    name='Pre_Comp_' + sect,
                                    kernel_initializer=normal_init(mean=0.0,
                                                                   stddev=0.01,
                                                                   seed=SEED)
                                    )
                Q = tf.layers.dense(Q,
                                    cp.getint(sect, 'Width'),
                                    activation=act_dict[cp.get(
                                        sect, 'Activation')],
                                    name='Pre_' + sect,
                                    kernel_initializer=xavier_init(uniform=False,
                                                                   seed=SEED),
                                    reuse=tf.AUTO_REUSE)
            else:
                Q = tf.layers.dense(sae_dict['sda_prev'],
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

    if cp.get('Experiment', 'PREFIX') == 'CIFAR':
        cmse = 0.5
        ccond = 1.0
    else:
        cmse = 1.0
        ccond = 1.0

    if cp.get('Experiment', 'PREFIX') == 'reu100k':
        TV1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope='Pre_Encoder|Pre_Decoder')
        for v in TV:
            if v.name == 'Pre_Decoder0/kernel:0':
                TV.remove(v)
        for v in TV:
            if v.name == 'Pre_Decoder0/bias:0':
                TV.remove(v)
        # Building loss of EviTRAM
        cond_loss, cond_lr, cond_t_op, px_mse, px_lr, px_t_op = disj_EviTRAM_loss(
                                                             sae_dict['sda_in'],
                                                             sae_dict['sda_out'],
                                                             sae_dict[
                                                                 'sda_hidden'], Qs,
                                                             ks_IN, TV1, TV)
    else:
        # Building loss of EviTRAM
        cond_loss, cond_lr, cond_t_op, px_mse = EviTRAM_loss(sae_dict['sda_in'],
                                                             sae_dict['sda_out'],
                                                             sae_dict[
                                                                 'sda_hidden'], Qs,
                                                             ks_IN, TV,
                                                             cmse, ccond)

    if cp.get('Experiment', 'PREFIX') == 'reu100k':
        ret_dict = {'sda_in': sae_dict['sda_in'],
                    'sda_hidden': sae_dict['sda_hidden'],
                    'sda_out': sae_dict['sda_out'],
                    'sda_prev': sae_dict['sda_prev'], 'Qs': Qs, 'ks_IN': ks_IN,
                    'TV': TV, 'evitram_t_op': cond_t_op, 'evitram_loss': cond_loss,
                    'px_mse': px_mse, 'px_t_op': px_t_op, 'px_lr': px_lr,
                    'evitram_lr': cond_lr}
    else:
        ret_dict = {'sda_in': sae_dict['sda_in'],
                    'sda_hidden': sae_dict['sda_hidden'],
                    'sda_out': sae_dict['sda_out'],
                    'sda_prev': sae_dict['sda_prev'], 'Qs': Qs, 'ks_IN': ks_IN,
                    'TV': TV, 'evitram_t_op': cond_t_op, 'evitram_loss': cond_loss,
                    'px_mse': px_mse, 'evitram_lr': cond_lr}
    return ret_dict


# Perform training loop, called from train.py


def train(td, sae_dict):
    # Initialize experiment parameters
    laymaxepochs = td['cp'].getint('Hyperparameters', 'AEMaxEpochs')
    sdamaxepochs = td['cp'].getint('Hyperparameters', 'SDAMaxEpochs')
    batch_size = td['cp'].getint('Hyperparameters', 'BatchSize')
    lr_decay = td['cp'].getint('Hyperparameters', 'DecayEpoch')

    # Layerwise autoencoder loop
    for num in td['ae_ids']:
        if num == '0':
            prev_hidden = None

        # Autoencoder id label
        aeid = 'AE' + num
        aelr = float(td['cp'].get('Hyperparameters', 'AELearningRate'))
        ae_lr_decay = td['cp'].getint('Hyperparameters', 'AEDecayEpoch')

        # Layer ae save strings
        layeraesavestr = td['out_'] + td['cp'].get('Experiment', 'PREFIX') + \
            '_px_' + aeid + '_' + 'model.ckpt'

        if prev_hidden is None:
            ae_IN = td['data']
        # If layerwise ae is not the first one then use the previous hidden
        # as input
        else:
            ae_IN = prev_hidden

        # Layerwise ae training
        # Epoch loop
        for epoch in range(laymaxepochs + 1):
            try:
                # Batch loop
                for row in tqdm(xrange(0, td['data'].shape[0], batch_size), ascii=True):
                    # Get batch
                    idx = slice(row, row + batch_size)
                    X_batch = ae_IN[idx]
                    # Backprop
                    td['sess'].run(sae_dict['l_train_tensors'][int(num)],
                                   feed_dict={sae_dict['l_in_tensors'][int(num)]: X_batch,
                                              sae_dict['l_lr_tensors'][int(num)]: aelr})

                _ll = td['sess'].run(sae_dict['l_mse_tensors'][int(num)],
                                     feed_dict={sae_dict['l_in_tensors'][int(num)]:
                                                X_batch})

                log(str(epoch) + ' ' + str(_ll),
                    label='AE' + str(num) + '-Lrecon')

            except KeyboardInterrupt:
                save_path = td['saver'].save(td['sess'], layeraesavestr)
                prev_hidden = td['sess'].run(sae_dict['l_hidden_tensors'][int(num)],
                                             feed_dict={sae_dict['l_in_tensors'][int(num)]: ae_IN})
                break
            # Learning rate decay
            if epoch == ae_lr_decay:
                aelr = aelr / 10.0
            # Save condition
            if epoch % 100 == 0:
                save_path = td['saver'].save(td['sess'], layeraesavestr)
        save_path = td['saver'].save(td['sess'], layeraesavestr)
        prev_hidden = td['sess'].run(sae_dict['l_hidden_tensors'][int(num)],
                                     feed_dict={sae_dict['l_in_tensors'][int(num)]: ae_IN})

    saelr = float(td['cp'].get('Hyperparameters', 'LearningRate'))

    # End to end ae training
    for epoch in range(sdamaxepochs + 1):
        try:
            # Batch loop
            for row in tqdm(xrange(0, td['data'].shape[0], batch_size), ascii=True):
                    # Get batch
                idx = slice(row, row + batch_size)
                X_batch = td['data'][idx]
                # Backprop
                td['sess'].run(sae_dict['sda_train'], feed_dict={sae_dict['sda_in']: X_batch,
                                                                 sae_dict['sda_lr']: saelr})
            _ll = td['sess'].run(sae_dict['sda_mse'],
                                 feed_dict={sae_dict['sda_in']: X_batch})
            log(str(epoch) + ' ' + str(_ll), label='SDA-Lrecon')

        except KeyboardInterrupt:
            save_path = td['saver'].save(td['sess'], td['savestr'])
            prev_hidden = td['sess'].run(sae_dict['sda_hidden'],
                                         feed_dict={sae_dict['sda_in']:
                                                    td['data']})
            break
        # Learning rate decay
        if (epoch == lr_decay):
            saelr = saelr / 10.0
        # Save condition
        if epoch % 100 == 0:
            save_path = td['saver'].save(td['sess'], td['savestr'])
    save_path = td['saver'].save(td['sess'], td['savestr'])

# Perform training loop, called from train.py


def evitram_train(td, evitramd):
    # Initialize experiment parameters
    maxepochs = td['cp'].getint('Hyperparameters', 'MaxEpochs')
    batch_size = td['cp'].getint('Hyperparameters', 'BatchSize')
    lr_decay = td['cp'].getint('Hyperparameters', 'DecayEpoch')
    lr_decay2 = td['cp'].getint('Hyperparameters', 'DecayEpoch2')
    evitramlr = float(td['cp'].get('Hyperparameters', 'LearningRate'))
    if td['cp'].get('Experiment', 'PREFIX') == 'reu100k':
        mse_lr = float(td['cp'].get('Hyperparameters', 'MSELearningRate'))

    # Epoch loop
    try:
        for epoch in range(maxepochs + 1):
            # Batch loop
            for row in tqdm(xrange(0, td['data2'].shape[0], batch_size), ascii=True):
                # Get batch
                idx = slice(row, row + batch_size)
                X_batch = td['data2'][idx]
                fd = {}
                fd[evitramd['sda_in']] = X_batch
                fd[evitramd['evitram_lr']] = evitramlr
                if td['cp'].get('Experiment', 'PREFIX') == 'reu100k':
                    fd[evitramd['px_lr']] = mse_lr
                for ev_path_id, ev_path in enumerate(td['ev_paths']):
                    fd[evitramd['ks_IN'][ev_path_id]] = td[
                        'EV2'][ev_path_id][idx]
                if td['cp'].get('Experiment', 'PREFIX') == 'reu100k':
                    try:
                        td['sess'].run(evitramd['px_t_op'], feed_dict=fd)
                        td['sess'].run(evitramd['evitram_t_op'], feed_dict=fd)
                    except:
                        pass
                else:
                    try:
                        td['sess'].run(evitramd['evitram_t_op'], feed_dict=fd)
                    except:
                        pass


            _eq_len = True
            for c, i in enumerate(td['EV2']):
                try:
                    if len(td['EV2'][c]) != len(td['EV2'][c+1]):
                        _eq_len = False
                except:
                    pass
            if _eq_len:
                _ll, rrec,  summary_str = td['sess'].run([evitramd['evitram_loss'],
                                                          evitramd['px_mse'],
                                                          td['sumr']],
                                                       feed_dict=fd)
            else:
                idx = slice(0, 0 + batch_size)
                X_batch = td['data2'][idx]
                fd = {}
                fd[evitramd['sda_in']] = X_batch
                for ev_path_id, ev_path in enumerate(td['ev_paths']):
                    fd[evitramd['ks_IN'][ev_path_id]] = td['EV2'][ev_path_id][idx]
                _ll, rrec,  summary_str = td['sess'].run([evitramd['evitram_loss'],
                                                          evitramd['px_mse'],
                                                          td['sumr']],
                                                       feed_dict=fd)

            #  print epoch, mse_ll, p, rrec, llo
            td['fw'].add_summary(summary_str, epoch)

            # Learning rate decay
            if (epoch == lr_decay) or (epoch == lr_decay2):
                evitramlr = evitramlr / 10.0
                if td['cp'].get('Experiment', 'PREFIX') == 'reu100k':
                    mse_lr = mse_lr / 10.0

            #  Save condition
            if epoch % 100 == 0:
                save_path = td['saver'].save(td['sess'], td['savestr'])
        save_path = td['saver'].save(td['sess'], td['savestr'])
    except KeyboardInterrupt:
        save_path = td['saver'].save(td['sess'], td['savestr'])
