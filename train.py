#! /usr/bin/env python

import tensorflow as tf
from tqdm import *
import sys
import ConfigParser
import numpy as np
import os
import utils

# Custom imports
sys.path.append('autoencoders/')
import EviAE
import ConvAE, SAE

##############################################################################
# Training mechanism of EviTRAM
##############################################################################

##########   Initialization of experiments properties, datasets etc.  ########

# Load configuration files
try:
    # Main autoencoder config file
    cp = utils.load_config(sys.argv[1])
except:
    print 'Help: ./train.py <path to main autoencoder ini file> <run number>'
    exit()

# Trying to reduce stochastic behaviours
SEED = cp.getint('Experiment', 'SEED')
tf.set_random_seed(SEED)
np.random.seed(SEED)

# Load dataset
inp_path = cp.get('Experiment', 'DATAINPUTPATH')
if inp_path == '':
    dataset = utils.load_mnist(val_size=cp.getint('Experiment',
                                                  'VALIDATIONSIZE'))
else:
    dataset = utils.load_data(inp_path)

# Create save directory if it doesn't exist (Primary AE)

directory = cp.get('Experiment', 'ModelOutputPath')
if not os.path.exists(directory):
    os.makedirs(directory)


##############################################################################
# Initializing save paths
##############################################################################

out = cp.get('Experiment', 'ModelOutputPath')
out_ = out.split('/')[0] + '/' + out.split('/')[1] + '/' + \
    out.split('/')[2] + '/'

# Pretrained model save path strings
pxfinestr = out_ + cp.get('Experiment', 'PREFIX') + '_px_model.ckpt.meta'

evitramfinestr = cp.get('Experiment', 'ModelOutputPath') + \
    cp.get('Experiment', 'PREFIX') + '_' + \
    cp.get('Experiment', 'Enumber') + '_' + \
    sys.argv[2] + '_cond_model.ckpt.meta'

# Full dataset random permutation path

perm_str = out_ + cp.get('Experiment', 'PREFIX') + '_perm.npy'

# Initialize Dataset
XX = dataset.train.images
XX_test = dataset.test.images
XX_full = np.concatenate((dataset.train.images, dataset.test.images))
utils.log(str(XX_full.shape))

p = utils.get_perm(perm_str, XX_full)

XX_full = XX_full[p]


# Init ground truth
YY = dataset.train.labels.flatten()
YY_test = dataset.test.labels.flatten()
YY_full = np.concatenate((dataset.train.labels.flatten(),
                          dataset.test.labels.flatten()))

YY_full = YY_full[p]


##############################################################################
# Start of primary autoencoder training (Capturing P(X))
##############################################################################

def px(pae_dict):
    # Initialize model save string
    pxmodelstr = pxfinestr.split('.meta')[0]

    # Variable initilization and saving
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    if cp.get('Experiment', 'PREFIX') == 'MNIST':
        # Tensorboard (comment / uncomment)
        ######################################################################

        from datetime import datetime

        now = datetime.utcnow().strftime("%m-%d_%H-%M:%S")
        root_logdir = cp.get('Experiment', 'ModelOutputPath')
        logdir = "{}/{}{}-{}/".format(root_logdir, 
                                      cp.get('Experiment', 'PREFIX') +
                                      '_' + cp.get('Experiment',
                                                   'Enumber') + '_px',
                                      sys.argv[2], now)

        tf.summary.scalar(name='xrecon loss', tensor=pae_dict['px_mse'])
        summary = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        ######################################################################

    # Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Initialize graph variables
        init.run()
        
        if cp.get('Experiment', 'PREFIX') == 'MNIST':
            train_dict = {'cp': cp, 'sess': sess, 'data': XX_full,
                          'sumr': summary, 'savestr': pxmodelstr, 
                          'saver': saver, 'fw': file_writer}
        else:
            train_dict = {'cp': cp, 'sess': sess, 'data': XX_full,
                          'savestr': pxmodelstr, 'saver': saver,
                          'ae_ids': ae_ids, 'out_': out_}

        if cp.get('Experiment', 'PREFIX') == 'MNIST':
            ConvAE.train(train_dict, pae_dict)
        else:
            SAE.train(train_dict, pae_dict)

        # Get batch size for batch output save
        batch_size = cp.getint('Hyperparameters', 'BatchSize')

        if cp.get('Experiment', 'PREFIX') == 'MNIST':
            # Save hidden/output layer results for pipeline training
            px_Z_latent = utils.run_OOM(sess, pae_dict['conv_in'], XX_full,
                                        pae_dict['conv_z'],
                                        batch_size=batch_size)
        else:
            # Save hidden/output layer results for pipeline training
            px_Z_latent = utils.run_OOM(sess, pae_dict['sda_in'], XX_full,
                                        pae_dict['sda_hidden'],
                                        batch_size=batch_size)
        #  utils.save_OOM(sess, pae_dict['conv_in'], XX_full,
        #  pae_dict['conv_out'],
        #  path=cp.get('Experiment', 'PX_XREC_TRAIN'),
        #  batch_size=batch_size)

    # Print clustering ACC
    utils.log_accuracy(cp, YY_full, px_Z_latent,
                       'PX - ACC FULL', SEED)

    # Print clustering NMI
    utils.log_NMI(cp, YY_full, px_Z_latent,
                  'PX - NMI FULL', SEED)

    sess.close()


##############################################################################
# Start of additional evidence autoencoder training ##############################################################################

def SAE_evidence(sae_dict, cp2, ae_ids):
    # Init variables and parameters
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Initialize model save string
    saemodelstr = saefinestr.split('.meta')[0]

    # Load evidence
    K = utils.load_evidence(cp2.get('Experiment', 'EVIDENCEDATAPATH'))
    EV = K.test.one

    #  _full = np.concatenate((K.train.one, K.test.one))
    #  p = utils.get_perm(perm_str, _full)

    #  EV = _full[p]

    batch_size = cp2.getint('Hyperparameters', 'BatchSize')

    # Start Session (Layerwise training)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Initialize graph variables
        init.run()

        ev_dict = {'cp2': cp2, 'sess': sess, 'saver': saver, 'EV': EV,
                   'savestr': saemodelstr, 'ae_ids': ae_ids,
                   'argv2': sys.argv[2]}

        EviAE.train(ev_dict, sae_dict)

        # Save hidden/output layer results for pipeline training
        utils.save_OOM(sess, sae_dict['sda_in'], EV, sae_dict['sda_hidden'],
                       path=cp2.get('Experiment', 'PX_Z_TRAIN'), batch_size=batch_size)
        utils.save_OOM(sess, sae_dict['sda_in'], EV, sae_dict['sda_out'],
                       path=cp2.get('Experiment', 'PX_XREC'), batch_size=batch_size)

    sess.close()

##############################################################################
# Start of evidence transfer method training
##############################################################################


def evitram():
    # Restore pretrained model
    restorestr = pxfinestr.split('.meta')[0]

    # Save model str
    evitramstr = evitramfinestr.split('.meta')[0]

    # Load pretrained evidence representations for all sources
    K = []
    _K = []
    for e in sys.argv[3:]:
	cp2 = utils.load_config(e)
	k = utils.load_evidence(cp2.get('Experiment', 'EVIDENCEDATAPATH'))
	K.append(k)
	_K.append(cp2.get('Experiment', 'PX_Z_TRAIN'))

    sect = 'Experiment'
    ev_paths = [cp.get(sect, i) for i in cp.options(sect) if 'evidence' in i]

    if cp.get('Experiment', 'PREFIX') == 'MNIST':
        evitram_dict = ConvAE.build_EviTRAM(cp, SEED)
    else:
        # Layerwise autoencoder number
        ae_ids = [str(i) for i in xrange(cp.getint('Experiment', 'AENUM'))]
        evitram_dict = SAE.build_EviTRAM(cp, ae_ids, SEED)

    # Get variables to restore from pretrained model P(x) Encoder
    var_list = tf.trainable_variables()

    for ev_path_id, ev_path in enumerate(ev_paths):
        if cp.get('Experiment', 'PREFIX') == 'MNIST':
            # Prepare "restore" variable list
            for v in var_list:
                if v.name == 'Pre_Q' + str(ev_path_id) + '/kernel:0':
                    var_list.remove(v)
            for v in var_list:
                if v.name == 'Pre_Q' + str(ev_path_id) + '/bias:0':
                    var_list.remove(v)
        else:
            # Prepare "restore" variable list
            for v in var_list:
                if v.name == 'Pre_Q' + str(ev_path_id) + '/kernel:0':
                    var_list.remove(v)
            for v in var_list:
                if v.name == 'Pre_Q' + str(ev_path_id) + '/bias:0':
                    var_list.remove(v)
            for v in var_list:
                if v.name == 'Pre_Comp_Q' + str(ev_path_id) + '/kernel:0':
                    var_list.remove(v)
            for v in var_list:
                if v.name == 'Pre_Comp_Q' + str(ev_path_id) + '/bias:0':
                    var_list.remove(v)


    ##########################################################
    # Tensorboard (comment / uncomment)
    ##########################################################

    from datetime import datetime

    now = datetime.utcnow().strftime("%m-%d_%H-%M:%S")
    root_logdir = cp.get('Experiment', 'ModelOutputPath')
    logdir = "{}/{}{}-{}/".format(root_logdir, cp.get('Experiment', 'PREFIX') +
                                  '_' + cp.get('Experiment',
                                               'Enumber') + '_cond',
                                  sys.argv[2], now)
    tf.summary.scalar(name='cond loss', tensor=evitram_dict['evitram_loss'])
    tf.summary.scalar(name='recon loss', tensor=evitram_dict['px_mse'])
    summary = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    ##########################################################

    # Initialize & restore P(x) AE weights
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list)
    saverCOND = tf.train.Saver()

    # Task outcomes
    #  EV = [np.load(i) for i in K]
    tr_EV = [i.train.one for i in K]
    te_EV = [np.load(i) for i in _K]
    tr_data = K[0].train.images
    te_data = K[0].test.images

    # Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Init values
        init.run()
        # Restore finetuned model
        saver.restore(sess, restorestr)

        train_dict = {'cp': cp, 'sess': sess, 'data': tr_data, 'data2': te_data,
                     'sumr': summary, 'savestr': evitramstr, 'saver': saverCOND,
                     'fw': file_writer, 'EV': tr_EV, 'EV2': te_EV,
                     'ev_paths': ev_paths}

        if cp.get('Experiment', 'PREFIX') == 'MNIST':
            ConvAE.evitram_train(train_dict, evitram_dict)
        else:
            SAE.evitram_train(train_dict, evitram_dict)


        # Get batch size for batch output save
        batch_size = train_dict['cp'].getint('Hyperparameters', 'BatchSize')

        if cp.get('Experiment', 'PREFIX') == 'MNIST':
            # Save hidden/output layer results for pipeline training
            px_Z_latent = utils.run_OOM(sess, evitram_dict['conv_in'], XX_full,
                                        evitram_dict['conv_z'],
                                        batch_size=batch_size)
        else:
            px_Z_latent = utils.run_OOM(sess, evitram_dict['sda_in'], XX_full,
                                        evitram_dict['sda_hidden'],
                                        batch_size=batch_size)
        #  utils.save_OOM(sess, pae_dict['conv_in'], XX_full,
        #  pae_dict['conv_out'],
        #  path=cp.get('Experiment', 'PX_XREC_TRAIN'),
        #  batch_size=batch_size)

    # Print clustering ACC
    utils.log_accuracy(cp, YY_full, px_Z_latent,
                       'COND - ACC FULL', SEED)

    # Print clustering NMI
    utils.log_NMI(cp, YY_full, px_Z_latent,
                  'COND - NMI FULL', SEED)

    sess.close()



# Training workflow

if __name__ == "__main__":
    sae_flag = False
    # Check if primary autoencoder is pretrained
    if os.path.exists(pxfinestr):
        # Train each additional evidence autoencoder
        for e in sys.argv[3:]:
            cp2 = utils.load_config(e)
            saefinestr = out_ + cp2.get('Experiment', 'ENUMBER') + \
                '_SAE_evidence_model.ckpt.meta'
            if os.path.exists(saefinestr):
                continue
            else:
                sae_flag = True
                #  Create save directory if it doesn't exist (Additional AE)
                directory2 = cp2.get('Experiment', 'SAE_DIR')
                if not os.path.exists(directory2):
                    os.makedirs(directory2)
                ae_ids = [str(i)
                          for i in xrange(cp2.getint('Experiment', 'AENUM'))]
                directory2 = cp2.get('Experiment', 'SAE_DIR')
                if not os.path.exists(directory2):
                    os.makedirs(directory2)
                sae_dict = EviAE.build(cp2, ae_ids, SEED)
                sae_dict = SAE_evidence(sae_dict, cp2, ae_ids)
        if not(sae_flag):
            evitram()
    else:
        # Train primary autoencoder
        if cp.get('Experiment', 'PREFIX') == 'MNIST':
            pae_dict = ConvAE.build_px(cp, SEED)
            px(pae_dict)
        else:
            # Layerwise autoencoder numbers
            ae_ids = [str(i) for i in xrange(cp.getint('Experiment', 'AENUM'))]
            pae_dict = SAE.build_px(cp, ae_ids, SEED)
            px(pae_dict)

