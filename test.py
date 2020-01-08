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

#############################################################################
#  This file contains all evidence transfer evaluation steps (loading models,
#  printing metrics, etc.)
#############################################################################

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


##############################################################################
# Initializing save paths
##############################################################################

out = cp.get('Experiment', 'ModelOutputPath')
out_ = out.split('/')[0] + '/' + out.split('/')[1] + '/' + \
    out.split('/')[2] + '/'

# Pretrained model save path strings
pxfinestr = out_ + cp.get('Experiment', 'PREFIX') + '_px_model.ckpt.meta'

saefinestr = out_ + cp.get('Experiment', 'ENUMBER') + \
    '_SAE_evidence_model.ckpt.meta'

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

# Get batch size in case of batch save
batch_size = cp.getint('Hyperparameters', 'BatchSize')

def px(pae_dict):
    saver = tf.train.Saver()

    # P(x) Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Restore pretrained model
        saver.restore(sess, pxfinestr.split('.meta')[0])

        if cp.get('Experiment', 'PREFIX') == 'MNIST':
            # Save hidden/output layer results for pipeline training
            px_Z_latent = utils.run_OOM(sess, pae_dict['conv_in'], XX_full,
                          pae_dict['conv_z'],
                          batch_size=batch_size)
        else:
            px_Z_latent = utils.run_OOM(sess, pae_dict['sda_in'], XX_full,
                                        pae_dict['sda_hidden'],
                                        batch_size=batch_size)

        # Print clustering ACC
        utils.log_accuracy(cp, YY_full, px_Z_latent,
                           'PX - ACC FULL', SEED)

        # Print clustering NMI
        utils.log_NMI(cp, YY_full, px_Z_latent,
                      'PX - NMI FULL', SEED)

        # Print clustering CHS score
        utils.log_CHS(cp, XX_full, px_Z_latent,
                      'PX - CHS FULL', SEED)


    sess.close()


def evitram(evitramd):
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Restore pretrained model
        saver.restore(sess, evitramfinestr.split('.meta')[0])

        if cp.get('Experiment', 'PREFIX') == 'MNIST':
            # Save hidden/output layer results for pipeline training
            px_Z_latent = utils.run_OOM(sess, evitram_dict['conv_in'], XX_full,
                          evitram_dict['conv_z'],
                          batch_size=batch_size)
        else:
            px_Z_latent = utils.run_OOM(sess, evitram_dict['sda_in'], XX_full,
                                        evitram_dict['sda_hidden'],
                                        batch_size=batch_size)

        # Print clustering ACC
        utils.log_accuracy(cp, YY_full, px_Z_latent,
                           'COND - ACC FULL', SEED)

        # Print clustering NMI
        utils.log_NMI(cp, YY_full, px_Z_latent,
                      'COND - NMI FULL', SEED)

        # Print clustering CHS score
        utils.log_CHS(cp, XX_full, px_Z_latent,
                      'COND - CHS FULL', SEED)

    sess.close()


# Test workflow

if __name__ == "__main__":
    if cp.get('Experiment', 'PREFIX') == 'MNIST':
        pae_dict = ConvAE.build_px(cp, SEED)
        px(pae_dict)
        # Reset graph for next model
        tf.reset_default_graph()
        tf.set_random_seed(SEED)
        evitram_dict = ConvAE.build_EviTRAM(cp, SEED)
        evitram(evitram_dict)
    else:
        # Layerwise autoencoder numbers
        ae_ids = [str(i) for i in xrange(cp.getint('Experiment', 'AENUM'))]
        pae_dict = SAE.build_px(cp, ae_ids, SEED)
        px(pae_dict)
        # Reset graph for next model
        tf.reset_default_graph()
        tf.set_random_seed(SEED)
        evitram_dict = SAE.build_EviTRAM(cp, ae_ids, SEED)
        evitram(evitram_dict)

