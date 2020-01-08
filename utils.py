#############################################################################
# This file contains function definitions for loading datasets, building
# networks, etc. The functions defined in this file are used in the main
# training script.
#############################################################################

import ConfigParser
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from dataset import Dataset
import sys
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score as SIL
from sklearn.metrics import calinski_harabaz_score as CHS
from tqdm import *


# Logging messages such as loss,loading,etc.


def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()

# Load neural network configuration file


def load_config(input_path):
    cp = ConfigParser.ConfigParser()
    cp.read(input_path)
    return cp


# Download / Extract mnist data


def load_mnist(val_size=0, path='/tmp/mnist/'):
    mnist = input_data.read_data_sets(
        path, validation_size=val_size)
    return mnist

# Initialize Activation dictionary


def get_act_dictionary():
    relu = tf.nn.relu
    linear = None
    sigmoid = tf.sigmoid
    softmax = tf.nn.softmax
    act_dict = {'ReLU': relu, 'Linear': linear,
                'Sigmoid': sigmoid, 'Softmax': softmax}
    return act_dict

# Checks npz keys


def check_keys(npz):
    #  keys = ['data', 'labels', 'test', 'tlab']
    keys = ['train_data', 'train_labels', 'test_data', 'test_labels']
    try:
        for k in keys:
            if k not in npz.keys():
                raise ValueError('npz structure missing key: ' + k + ', npz must' +
                                 'have keys [data, labels, test, tlab]')
    except:
        raise ValueError('npz structure missing key: ' + k + ', npz must' +
                         'have keys [data, labels, test, tlab]')


# Generic function for loading datasets


def load_data(path):
    if '.npz' not in path:
        raise TypeError("load_data currently handles only .npz structures")
    struct = np.load(path)
    check_keys(struct)
    try:
        # handle labels as integers
        #  trlab = struct['labels']
        trlab = struct['train_labels']
        trlab = trlab.reshape(trlab.shape[0], 1)
        #  telab = struct['tlab']
        telab = struct['test_labels']
        telab = telab.reshape(telab.shape[0], 1)
        #  return Dataset(struct['data'], struct['test'], trlab, telab)
        return Dataset(struct['train_data'], struct['test_data'], trlab, telab)
    except:
        #  trlab = struct['labels']
        trlab = struct['train_labels']
        #  telab = struct['tlab']
        telab = struct['test_labels']
        #  d = Dataset(struct['data'], struct['test'], trlab, telab)
        d = Dataset(struct['train_data'], struct['test_data'], trlab, telab)
        d.set_train_ones(trlab)
        d.set_test_ones(telab)
        return d

# Load evidence data


def load_evidence(inp_path, val_size=0):
    if inp_path == '':
        evidence = load_mnist(val_size=val_size)
        evidence = Dataset(evidence.train.images,
                           evidence.test.images,
                           evidence.train.labels.flatten().reshape(60000, 1),
                           evidence.test.labels.flatten().reshape(10000, 1))
    else:
        evidence = load_data(inp_path)
    return evidence

# Access full dataset permutation


def get_perm(perm_str, XX):
    if os.path.exists(perm_str):
        p = np.load(perm_str)
    else:
        p = np.random.permutation(XX.shape[0])
        np.save(perm_str, p)
    return p


# Unsupervised clustering accuracy from DEC
# (https://arxiv.org/pdf/1511.06335.pdf)


def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in xrange(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w

# Save tensor output


def save(sess, tensor_in, data_in, tensor_out, path, batch_size):
    # Calculate tensor to numpy and save
    try:
        calc = sess.run(tensor_out, feed_dict={tensor_in: data_in})
        np.save(path, calc)
    # If gpu memory is not enough for whole dataset, calculate in batches
    except:
        temp_arr = np.zeros(shape=(data_in.shape[0], tensor_out.shape[1]))
        for row in tqdm(xrange(0, data_in.shape[0], batch_size), ascii=True):
            # Get batch
            idx = slice(row, row + batch_size)
            X_batch = data_in[idx]
            savehidden = sess.run(tensor_out, feed_dict={tensor_in: X_batch})
            temp_arr[idx, :] = savehidden
            hidden = temp_arr
        np.save(path, temp_arr[:])

# Save tensor output when sure that GPU is OOM


def save_OOM(sess, tensor_in, data_in, tensor_out, path, batch_size):
    temp_arr = np.zeros(shape=(data_in.shape[0], tensor_out.shape[1]))
    for row in tqdm(xrange(0, data_in.shape[0], batch_size), ascii=True):
        # Get batch
        idx = slice(row, row + batch_size)
        X_batch = data_in[idx]
        savehidden = sess.run(tensor_out, feed_dict={tensor_in: X_batch})
        temp_arr[idx, :] = savehidden
        hidden = temp_arr
    np.save(path, temp_arr[:])

# Get tensor output


def run(sess, tensor_in, data_in, tensor_out, batch_size):
    # Calculate tensor to numpy and save
    try:
        calc = sess.run(tensor_out, feed_dict={tensor_in: data_in})
        return calc
    # If gpu memory is not enough for whole dataset, calculate in batches
    except:
        temp_arr = np.zeros(shape=(data_in.shape[0], tensor_out.shape[1]))
        for row in tqdm(xrange(0, data_in.shape[0], batch_size), ascii=True):
            # Get batch
            idx = slice(row, row + batch_size)
            X_batch = data_in[idx]
            savehidden = sess.run(tensor_out, feed_dict={tensor_in: X_batch})
            temp_arr[idx, :] = savehidden
            hidden = temp_arr
        return temp_arr[:]


# Get tensor output when sure that GPU is OOM


def run_OOM(sess, tensor_in, data_in, tensor_out, batch_size):
    temp_arr = np.zeros(shape=(data_in.shape[0], tensor_out.shape[1]))
    for row in tqdm(xrange(0, data_in.shape[0], batch_size), ascii=True):
        # Get batch
        idx = slice(row, row + batch_size)
        X_batch = data_in[idx]
        savehidden = sess.run(tensor_out, feed_dict={tensor_in: X_batch})
        temp_arr[idx, :] = savehidden
        hidden = temp_arr
    return temp_arr[:]


# Log unsupervised clustering accuracy


def log_accuracy(cp, ground_truth, dataset, log_flag, SEED=1234):
    # Init clustering hyperparameters
    n_clusters = cp.getint('Hyperparameters', 'ClusterNum')
    cluster_init = cp.getint('Hyperparameters', 'ClusterInit')
    # KMeans model
    km = KMeans(n_clusters=n_clusters, n_init=cluster_init, n_jobs=-1,
                random_state=SEED)
    if isinstance(dataset, basestring):
        pred = km.fit_predict(np.load(dataset))
    else:
        pred = km.fit_predict(dataset)
    log('--------------- {} {} ------------------------'.
        format(log_flag, cluster_acc(pred, ground_truth)[0]))

# Log normalized mutual information


def log_NMI(cp, ground_truth, dataset, log_flag, SEED=1234):
    # Init clustering hyperparameters
    n_clusters = cp.getint('Hyperparameters', 'ClusterNum')
    cluster_init = cp.getint('Hyperparameters', 'ClusterInit')
    # KMeans model
    km = KMeans(n_clusters=n_clusters, n_init=cluster_init, n_jobs=-1,
                random_state=SEED)
    if isinstance(dataset, basestring):
        pred = km.fit_predict(np.load(dataset))
    else:
        pred = km.fit_predict(dataset)
    log('--------------- {} {} ------------------------'.
        format(log_flag, NMI(ground_truth, pred)))


# Prepare membership plot


def log_MEM(cp, dataset, SEED=1234):
    # Init clustering hyperparameters
    n_clusters = cp.getint('Hyperparameters', 'ClusterNum')
    cluster_init = cp.getint('Hyperparameters', 'ClusterInit')
    # KMeans model
    km = KMeans(n_clusters=n_clusters, n_init=cluster_init, n_jobs=-1,
                random_state=SEED)
    if isinstance(dataset, basestring):
        pred = km.fit_predict(np.load(dataset))
    else:
        pred = km.fit_predict(dataset)
    return km.cluster_centers_, pred


# Log calinski harabaz score


def log_CHS(cp, raw_data, dataset, log_flag, SEED=1234):
    # Init clustering hyperparameters
    n_clusters = cp.getint('Hyperparameters', 'ClusterNum')
    cluster_init = cp.getint('Hyperparameters', 'ClusterInit')
    # KMeans model
    km = KMeans(n_clusters=n_clusters, n_init=cluster_init, n_jobs=-1,
                random_state=SEED)
    if isinstance(dataset, basestring):
        pred = km.fit_predict(np.load(dataset))
    else:
        pred = km.fit_predict(dataset)
    x = raw_data
    log('--------------- {} {} ------------------------'.
        format(log_flag, CHS(x, pred)))
