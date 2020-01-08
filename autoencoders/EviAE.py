#############################################################################
# Stacked autoencoder (For evidence)
#############################################################################
import tensorflow as tf
from utils import get_act_dictionary
from tensorflow import truncated_normal_initializer as normal_init
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
        # Hidden Layer(s)
        for sect in [i for i in cp.sections() if enclabel in i]:
            encoder = tf.layers.dense(encoder,
                                      cp.getint(sect, 'Width'),
                                      activation=act_dict[cp.get(
                                          sect, 'Activation')],
                                      name='SAE_' + sect,
                                      kernel_initializer=normal_init(mean=0.0,
                                                                     stddev=0.01, seed=SEED)
                                      )
    with tf.name_scope(declabel):
        decoder = encoder
        # Decoder Layer(s)
        for sect in [i for i in cp.sections() if declabel in i]:
            decoder = tf.layers.dense(decoder,
                                      cp.getint(sect, 'Width'),
                                      activation=act_dict[cp.get(
                                          sect, 'Activation')],
                                      name='SAE_' + sect,
                                      kernel_initializer=normal_init(mean=0.0,
                                                                     stddev=0.01, seed=SEED)
                                      )
    return decoder, encoder

# Create stacked end to end autoencoder


def build_stacked(incoming, cp, reuse=False):
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
            # Hidden layers
            for sect in [i for i in cp.sections() if enclabel in i]:
                sda = tf.layers.dense(sda,
                                      cp.getint(sect, 'Width'),
                                      activation=act_dict[cp.get(
                                          sect, 'Activation')],
                                      name='SAE_' + sect)
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
                                      name='SAE_' + sect)
    return sda, encoder


# Create reconstruction loss for layerwise autoencoders


def build_layer_loss(X, out, num, scope):
    aeid = 'AE' + num
    # Initialize each loss
    with tf.name_scope(scope):
        ae_lr = tf.placeholder(tf.float32, shape=[])
        ae_mse = tf.reduce_mean(tf.square(X - out), name=aeid + '_mse')
        # Optimizer
        ae_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        vl = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='SAE_Encoder' + num + '|SAE_Decoder' + num)
        ae_t_op = ae_optimizer.minimize(ae_mse, var_list=vl)
    return ae_mse, ae_lr, ae_t_op


# Create reconstruction loss for stacked autoencoder (end-to-end finetune)


def build_stacked_loss(X, out, scope='sda_loss'):
    # Initialize loss
    with tf.name_scope(scope):
        sda_lr = tf.placeholder(tf.float32, shape=[])
        sda_mse = tf.reduce_mean(tf.square(X - out), name='sda_mse')
        # Optimizer
        sda_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        vl = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='SAE_Encoder|SAE_Decoder')
        sda_t_op = sda_optimizer.minimize(sda_mse, var_list=vl)
    return sda_lr, sda_mse, sda_t_op


# Build pre Dec phase aka create layerwise and stacked autoencoders


def build(cp, ae_ids, SEED):
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
        sda_out, sda_hidden = build_stacked(sda_X, cp, reuse=True)

    # Loss
    sda_lr, sda_mse, sda_t_op = build_stacked_loss(sda_X, sda_out)

    ############    End of Stacked autoencoder initialization     ############

    # Create return dictionary
    ret_dict = {'l_in_tensors': l_in_tensors, 'l_hidden_tensors': l_hidden_tensors,
                'l_out_tensors': l_out_tensors, 'l_train_tensors': l_train_tensors,
                'l_lr_tensors': l_lr_tensors, 'l_mse_tensors': l_mse_tensors,
                'sda_in': sda_X, 'sda_hidden': sda_hidden, 'sda_out': sda_out,
                'sda_train': sda_t_op, 'sda_lr': sda_lr, 'sda_mse': sda_mse}
    return ret_dict

# Perform training loop, called from train.py

def train(evd, sae_dict):
    # Initialize experiment parameters
    laymaxepochs = evd['cp2'].getint('Hyperparameters', 'AEMaxEpochs')
    sdamaxepochs = evd['cp2'].getint('Hyperparameters', 'SDAMaxEpochs')
    batch_size = evd['cp2'].getint('Hyperparameters', 'BatchSize')

    # Layerwise autoencoder loop
    for num in evd['ae_ids']:
        if num == '0':
            prev_hidden = None

        # Autoencoder id label
        aeid = 'AE' + num
        # Layer ae save strings
        layeraesavestr = evd['cp2'].get('Experiment', 'ModelOutputPath') + \
            evd['cp2'].get('Experiment', 'PREFIX') + '_' + \
            evd['cp2'].get('Experiment', 'Enumber') + '_' + evd['argv2'] + \
            '_' + aeid + '_model.ckpt'

        if prev_hidden is None:
            ae_IN = evd['EV']
        # If layerwise ae is not the first one then use the previous hidden
        # as input
        else:
            ae_IN = prev_hidden

        # Layerwise ae training
        # Epoch loop
        for epoch in range(laymaxepochs + 1):
            try:
                # Batch loop
                for row in tqdm(xrange(0, evd['EV'].shape[0], batch_size), ascii=True):
                    # Get batch
                    idx = slice(row, row + batch_size)
                    X_batch = ae_IN[idx]
                    # Backprop
                    evd['sess'].run(sae_dict['l_train_tensors'][int(num)],
                             feed_dict={sae_dict['l_in_tensors'][int(num)]: \
                                 X_batch})
            except KeyboardInterrupt:
                save_path = evd['saver'].save(evd['sess'], layeraesavestr)
                prev_hidden = evd['sess'].run(sae_dict['l_hidden_tensors'][int(num)],
                                       feed_dict={
                                           sae_dict['l_in_tensors'][int(num)]: \
                                           ae_IN})
                break
            # Save condition
            if epoch % 100 == 0:
                save_path = evd['saver'].save(evd['sess'], layeraesavestr)
        save_path = evd['saver'].save(evd['sess'], layeraesavestr)
        prev_hidden = evd['sess'].run(sae_dict['l_hidden_tensors'][int(num)],
                               feed_dict={sae_dict['l_in_tensors'][int(num)]: \
                                   ae_IN})

        # End to end ae training
        for epoch in range(sdamaxepochs + 1):
            try:
                # Batch loop
                for row in tqdm(xrange(0, evd['EV'].shape[0], batch_size), ascii=True):
                        # Get batch
                    idx = slice(row, row + batch_size)
                    X_batch = evd['EV'][idx]
                    # Backprop
                    evd['sess'].run(sae_dict['sda_train'], feed_dict={
                             sae_dict['sda_in']: X_batch})
            except KeyboardInterrupt:
                save_path = evd['saver'].save(evd['sess'], evd['savestr'])
                prev_hidden = evd['sess'].run(sae_dict['sda_hidden'],
                                       feed_dict={sae_dict['sda_in']:
                                                  evd['EV']})
                break
            # Save condition
            if epoch % 100 == 0:
                save_path = evd['saver'].save(evd['sess'], evd['savestr'])
        save_path = evd['saver'].save(evd['sess'], evd['savestr'])
