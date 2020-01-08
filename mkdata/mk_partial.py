#! /usr/bin/env python

import numpy as np
import sys

sys.path.append('..')

import utils

LEN_MNIST_TRAIN = 60000
LEN_MNIST_TEST = 10000

LEN_CIFAR_TRAIN = 50000
LEN_CIFAR_TEST = 10000

LEN_20NG_TRAIN = 14625
LEN_20NG_TEST = 3657

LEN_REUTERS_TRAIN = 87239
LEN_REUTERS_TEST = 9694

REUTERS_COUNT = 100000
REUTERS_LABELS = 10

VGG_16_FEATURES = 4096

GNEWSVEC = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'

REU_URLS = [
    'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz',
    'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz',
    'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz',
    'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz',
    'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz',
    'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz'
]


REU_DAT_gz = ['lyrl2004_tokens_test_pt0.dat.gz',
              'lyrl2004_tokens_test_pt1.dat.gz',
              'lyrl2004_tokens_test_pt2.dat.gz',
              'lyrl2004_tokens_test_pt3.dat.gz',
              'lyrl2004_tokens_train.dat.gz',
              'rcv1-v2.topics.qrels.gz']

def create_partial(tr_labels, te_labels, rate):
    # var init
    rate = float(rate)
    total_num = len(tr_labels) + len(te_labels)
    total_lab = np.concatenate((tr_labels, te_labels))
    num_keep = int(total_num * rate)
    # Init list with "invalid" labels, 999 represents missing label
    placeholder_lst = np.array([999] * total_num)
    # Keep only rate percent worth of labels / evidence
    np.random.seed(1234)
    random_idc = np.random.permutation(total_num)
    random_idc = random_idc[:num_keep]
    placeholder_lst[random_idc] = total_lab[random_idc]
    # Split train/test
    tr_labels = placeholder_lst[:len(tr_labels)]
    te_labels = placeholder_lst[-1*len(te_labels):]
    return tr_labels, te_labels

def create_partial2(tr_data, tr_labels, te_data, te_labels, rate):
    # var init
    rate = float(rate)
    total_num = len(tr_labels) + len(te_labels)
    num_keep = int(total_num * rate)
    total_lab = np.concatenate((tr_labels, te_labels))
    total_data = np.concatenate((tr_data, te_data))
    # Keep only rate percent worth of labels / evidence
    np.random.seed(1234)
    random_idc = np.random.permutation(total_num)
    random_idc = random_idc[:num_keep]

    unlab_data = [i for c,i in enumerate(total_data) if c not in random_idc]
    unlab_labels = [i for c,i in enumerate(total_lab) if c not in random_idc]

    lab_data = total_data[random_idc]
    lab_labels = total_lab[random_idc]

    return unlab_data, unlab_labels, lab_data, lab_labels


def skip_class(tr_data, tr_labels, te_data, te_labels, num=1):
    # Init
    total_lab = np.concatenate((tr_labels, te_labels))
    total_data = np.concatenate((tr_data, te_data))

    # Skip # number of evidence "classes"
    unique_labels = np.unique(total_lab)
    np.random.seed(1234)
    keep_idc = np.random.permutation(len(unique_labels))
    keep = unique_labels[keep_idc[:len(unique_labels)-int(num)]]

    unlab_perm = [c for c, i in enumerate(total_lab) if i not in keep]
    lab_perm = [c for c, i in enumerate(total_lab) if i in keep]

    return total_data[unlab_perm], total_lab[unlab_perm], total_data[lab_perm], total_lab[lab_perm]




# Mnist real (w:3) = x mod 3 (relation)


def mnist_real3(out='./', prefix='MNIST_', rate=None, skip=None):
    mnist = utils.load_mnist(val_size=0)
    tr_labels = []
    for i in mnist.train.labels:
        tr_labels.append(i % 3)
    te_labels = []
    for i in mnist.test.labels:
        te_labels.append(i % 3)
    if rate:
        tr_data, tr_labels, te_data, te_labels = create_partial2(mnist.train.images, tr_labels,
                mnist.test.images, te_labels, rate)
        np.savez_compressed(out + prefix + 'real3_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    elif skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(mnist.train.images, tr_labels,
                mnist.test.images, te_labels, skip)
        np.savez_compressed(out + prefix + 'real3_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    else:
        np.savez_compressed(out + prefix + 'real3.npz',
                            train_data=np.zeros(shape=(1, 1)),
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=np.zeros(shape=(1, 1))
                            )

# Mnist real (w:4) = hash(x) mod 4 (relation)


def mnist_real4(out='./', prefix='MNIST_', rate=None, skip=None):
    mnist = utils.load_mnist(val_size=0)
    tr_labels = []
    for i in mnist.train.labels:
        tr_labels.append(hash(i) % 4)
    te_labels = []
    for i in mnist.test.labels:
        te_labels.append(hash(i) % 4)
    if rate:
        tr_data, tr_labels, te_data, te_labels = create_partial2(mnist.train.images, tr_labels,
                mnist.test.images, te_labels, rate)
        np.savez_compressed(out + prefix + 'real4_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    elif skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(mnist.train.images, tr_labels,
                mnist.test.images, te_labels, skip)
        np.savez_compressed(out + prefix + 'real4_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    else:
        np.savez_compressed(out + prefix + 'real4.npz',
                            train_data=np.zeros(shape=(1, 1)),
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=np.zeros(shape=(1, 1))
                            )


# Mnist real (w:4) = hash(x) mod 4 (relation)


def mnist_real10(out='./', prefix='MNIST_', rate=None, skip=None):
    mnist = utils.load_mnist(val_size=0)
    tr_labels = mnist.train.labels
    te_labels = mnist.test.labels
    if skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(mnist.train.images, tr_labels,
                mnist.test.images, te_labels, skip)
        np.savez_compressed(out + prefix + 'real10_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    else:
        tr_data, tr_labels, te_data, te_labels = create_partial2(mnist.train.images, tr_labels,
                    mnist.test.images, te_labels, rate)
        np.savez_compressed(out + prefix + 'real10_'+str(rate)+'.npz',
                                train_data=tr_data,
                                train_labels=tr_labels, test_labels=te_labels,
                                test_data=te_data
                            )

# Random evidence (w:3) = white noise


def mnist_rand3(out='./', prefix='MNIST_'):
    np.savez_compressed(out + prefix + 'rand3.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=np.random.uniform(
                            size=(LEN_MNIST_TRAIN, 3)),
                        test_labels=np.random.uniform(
                            size=(LEN_MNIST_TEST, 3)),
                        test_data=np.zeros(shape=(1, 1))
                        )

# Random evidence (w:10) = white noise


def mnist_rand10(out='./', prefix='MNIST_'):
    np.savez_compressed(out + prefix + 'rand10.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=np.random.uniform(
                            size=(LEN_MNIST_TRAIN, 10)),
                        test_labels=np.random.uniform(
                            size=(LEN_MNIST_TEST, 10)),
                        test_data=np.zeros(shape=(1, 1))
                        )

# Real evidence (w:3) in random order


def mnist_rorder3(out='./', prefix='MNIST_'):
    mnist = utils.load_mnist(val_size=0)
    tr_labels = []
    for i in mnist.train.labels:
        tr_labels.append(i % 3)
    te_labels = []
    for i in mnist.test.labels:
        te_labels.append(i % 3)

    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    np.random.shuffle(tr_labels)
    np.random.shuffle(te_labels)

    np.savez_compressed(out + prefix + 'rorder3.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=tr_labels, test_labels=te_labels,
                        test_data=np.zeros(shape=(1, 1))
                        )

# Real evidence (w:10) in random order


def mnist_rorder10(out='./', prefix='MNIST_'):
    mnist = utils.load_mnist(val_size=0)
    tr_labels = mnist.train.labels
    te_labels = mnist.test.labels

    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    np.random.shuffle(tr_labels)
    np.random.shuffle(te_labels)

    np.savez_compressed(out + prefix + 'rorder10.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=tr_labels, test_labels=te_labels,
                        test_data=np.zeros(shape=(1, 1))
                        )

# Extracting vgg 16 features
# from https://github.com/XifengGuo/DEC-keras


def extract_vgg16_features(x):
    from keras.preprocessing.image import img_to_array, array_to_img
    from keras.applications.vgg16 import preprocess_input, VGG16
    from keras.models import Model
    im_h = 224
    model = VGG16(include_top=True, weights='imagenet',
                  input_shape=(im_h, im_h, 3))

    feature_model = Model(model.input, model.get_layer('fc1').output)
    x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,
                                                                       im_h))) for im in x])
    x = preprocess_input(x)
    features = feature_model.predict(x)

    return features


def mk_cifar10_vgg(out='./', prefix='CIFAR10_'):
    from keras.datasets import cifar10
    LEN_CIFAR_FULL = LEN_CIFAR_TRAIN + LEN_CIFAR_TEST
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y)).reshape((LEN_CIFAR_FULL,))

    # extract features
    features = np.zeros((LEN_CIFAR_FULL, VGG_16_FEATURES))
    BATCHES = 6
    SLICE = LEN_CIFAR_FULL / BATCHES
    for i in range(BATCHES):
        idx = range(i * SLICE, (i + 1) * SLICE)
        features[idx] = extract_vgg16_features(x[idx])

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)

    np.savez_compressed(out + prefix + 'vgg.npz',
                        train_data=features[:LEN_CIFAR_TRAIN],
                        test_data=features[LEN_CIFAR_TRAIN:],
                        train_labels=y[:LEN_CIFAR_TRAIN],
                        test_labels=y[LEN_CIFAR_TRAIN:])

# Real evidence (w:3): Vehicles, pets, wild animals


def cifar10_real3(out='./', prefix='CIFAR10_', rate=None, skip=None):
    #  from keras.datasets import cifar10
    #  (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    cif = np.load('CIFAR10_vgg.npz')
    train_x = cif['train_data']
    train_y = cif['train_labels']
    test_x = cif['test_data']
    test_y = cif['test_labels']

    #  (TRAIN) Vehicles:0, Pets:1, Wild animals:2
    tr_labels = []
    for i in train_y:
        if i in [3, 5, 7]:
            tr_labels.append(0)
        elif i in [0, 1, 8, 9]:
            tr_labels.append(1)
        else:
            tr_labels.append(2)
    # (TEST) Vehicles:0, Pets:1, Wild animals:2
    te_labels = []
    for i in test_y:
        if i in [3, 5, 7]:
            te_labels.append(0)
        elif i in [0, 1, 8, 9]:
            te_labels.append(1)
        else:
            te_labels.append(2)
    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    if rate:
        tr_data, tr_labels, te_data, te_labels = create_partial2(train_x, tr_labels,
                test_x, te_labels, rate)
        np.savez_compressed(out + prefix + 'real3_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    elif skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(train_x, tr_labels,
                test_x, te_labels, skip)
        np.savez_compressed(out + prefix + 'real3_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    else:
        np.savez_compressed(out + prefix + 'real3.npz',
                            train_data=np.zeros(shape=(1, 1)),
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=np.zeros(shape=(1, 1))
                            )


# Real evidence (w:4): Vehicles, breaking down animals in more categories


def cifar10_real4(out='./', prefix='CIFAR10_', rate=None, skip=None):
    #  from keras.datasets import cifar10
    #  (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    cif = np.load('CIFAR10_vgg.npz')
    train_x = cif['train_data']
    train_y = cif['train_labels']
    test_x = cif['test_data']
    test_y = cif['test_labels']

    #  (TRAIN) Vehicles:0, Household four legged pets (dog/cat):1, Outdoors
    #  four legged animals (deer/horse): 2, Wild (frog/bird): 3
    tr_labels = []
    for i in train_y:
        if i in [0, 1, 8, 9]:
            tr_labels.append(0)
        elif i in [3, 5]:
            tr_labels.append(1)
        elif i in [4, 7]:
            tr_labels.append(2)
        else:
            tr_labels.append(3)
    #  (TEST) Vehicles:0, Household four legged pets (dog/cat):1, Outdoors
    #  four legged animals (deer/horse): 2, Wild (frog/bird): 3
    te_labels = []
    for i in test_y:
        if i in [0, 1, 8, 9]:
            te_labels.append(0)
        elif i in [3, 5]:
            te_labels.append(1)
        elif i in [4, 7]:
            te_labels.append(2)
        else:
            te_labels.append(3)
    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    if rate:
        tr_data, tr_labels, te_data, te_labels = create_partial2(train_x, tr_labels,
                test_x, te_labels, rate)
        np.savez_compressed(out + prefix + 'real4_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    elif skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(train_x, tr_labels,
                test_x, te_labels, skip)
        np.savez_compressed(out + prefix + 'real4_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    else:
        np.savez_compressed(out + prefix + 'real4.npz',
                            train_data=np.zeros(shape=(1, 1)),
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=np.zeros(shape=(1, 1))
                            )

# Real evidence (w:5): breakdown vehicles and animals in more categories


def cifar10_real5(out='./', prefix='CIFAR10_', rate=None, skip=None):
    #  from keras.datasets import cifar10
    #  (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    cif = np.load('CIFAR10_vgg.npz')
    train_x = cif['train_data']
    train_y = cif['train_labels']
    test_x = cif['test_data']
    test_y = cif['test_labels']


    # (TRAIN) 0: car/truck, 1: dog,deer, 2:cat,horse, 3: bird,frog,
    # 4: airplane, ship
    tr_labels = []
    for i in train_y:
        if i in [9, 1]:
            tr_labels.append(0)
        elif i in [5, 4]:
            tr_labels.append(1)
        elif i in [3, 7]:
            tr_labels.append(2)
        elif i in [2, 6]:
            tr_labels.append(3)
        else:
            tr_labels.append(4)
    # (TEST) 0: car/truck, 1: dog,deer, 2:cat,horse, 3: bird,frog,
    # 4: airplane, ship
    te_labels = []
    for i in test_y:
        if i in [9, 1]:
            te_labels.append(0)
        elif i in [5, 4]:
            te_labels.append(1)
        elif i in [3, 7]:
            te_labels.append(2)
        elif i in [2, 6]:
            te_labels.append(3)
        else:
            te_labels.append(4)
    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    if rate:
        tr_data, tr_labels, te_data, te_labels = create_partial2(train_x, tr_labels,
                test_x, te_labels, rate)
        np.savez_compressed(out + prefix + 'real5_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    elif skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(train_x, tr_labels,
                test_x, te_labels, skip)
        np.savez_compressed(out + prefix + 'real5_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    else:
        np.savez_compressed(out + prefix + 'real5.npz',
                            train_data=np.zeros(shape=(1, 1)),
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=np.zeros(shape=(1, 1))
                            )



def cifar10_real10(out='./', prefix='CIFAR10_', rate=None, skip=None):
    cif = np.load('CIFAR10_vgg.npz')
    train_x = cif['train_data']
    train_y = cif['train_labels']
    test_x = cif['test_data']
    test_y = cif['test_labels']

    if rate:
        tr_data, tr_labels, te_data, te_labels = create_partial2(train_x, train_y,
                test_x, test_y, rate)
        np.savez_compressed(out + prefix + 'real10_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    elif skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(train_x, train_y,
                test_x, test_y, skip)
        np.savez_compressed(out + prefix + 'real10_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )

# Random evidence (w:3) = white noise


def cifar10_rand3(out='./', prefix='CIFAR10_'):
    np.savez_compressed(out + prefix + 'rand3.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=np.random.uniform(
                            size=(LEN_CIFAR_TRAIN, 3)),
                        test_labels=np.random.uniform(
                            size=(LEN_CIFAR_TEST, 3)),
                        test_data=np.zeros(shape=(1, 1))
                        )

# Random evidence (w:10) = white noise


def cifar10_rand10(out='./', prefix='CIFAR10_'):
    np.savez_compressed(out + prefix + 'rand10.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=np.random.uniform(
                            size=(LEN_CIFAR_TRAIN, 10)),
                        test_labels=np.random.uniform(
                            size=(LEN_CIFAR_TEST, 10)),
                        test_data=np.zeros(shape=(1, 1))
                        )

# Random evidence (w:10) = white noise


def cifar10_rand5(out='./', prefix='CIFAR10_'):
    np.savez_compressed(out + prefix + 'rand5.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=np.random.uniform(
                            size=(LEN_CIFAR_TRAIN, 5)),
                        test_labels=np.random.uniform(
                            size=(LEN_CIFAR_TEST, 5)),
                        test_data=np.zeros(shape=(1, 1))
                        )

# Real evidence (w:3) in random order


def cifar10_rorder3(out='./', prefix='CIFAR10_'):
    try:
        cifar10_real3 = np.load(out + prefix + 'real3.npz')
    except:
        print out + prefix + 'real3.npz, run ./mk.py cifar10_real3 first'
        exit()
    tr_labels = cifar10_real3['train_labels']
    te_labels = cifar10_real3['test_labels']

    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    np.random.shuffle(tr_labels)
    np.random.shuffle(te_labels)

    np.savez_compressed(out + prefix + 'rorder3.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=tr_labels, test_labels=te_labels,
                        test_data=np.zeros(shape=(1, 1))
                        )

# Real evidence (w:10) in random order


def cifar10_rorder10(out='./', prefix='CIFAR10_'):
    from keras.datasets import cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    tr_labels = np.array(train_y)
    te_labels = np.array(test_y)

    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    np.random.shuffle(tr_labels)
    np.random.shuffle(te_labels)

    np.savez_compressed(out + prefix + 'rorder10.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=tr_labels, test_labels=te_labels,
                        test_data=np.zeros(shape=(1, 1))
                        )


def filter_docs(corpus, texts, labels, condition_on_doc):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    #  print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels)


def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    return np.mean(word2vec_model[doc], axis=0)


def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)

# Make average word2vec features for 20newsgroups
# https://github.com/sdimi/average-word2vec


def mk_20newsgroups_avgw2v(out='./', prefix='20NG_'):
    import gensim
    from gensim import utils
    from sklearn.datasets import fetch_20newsgroups
    from nltk import download
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    import os
    import wget

    if os.path.exists('GoogleNews-vectors-negative300.bin.gz'):
        pass
    else:
        wget.download(GNEWSVEC)

    model = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin.gz', binary=True)

    print 'Preprocessing.........'

    # Preprocessing
    download('punkt')
    download('stopwords')
    stop_words = stopwords.words('english')

    def corpus_preprocess(text):
        text = text.lower()
        doc = word_tokenize(text)
        doc = [word for word in doc if word not in stop_words]
        # restricts string to alphabetic characters only
        doc = [word for word in doc if word.isalpha()]
        return doc

    ng20 = fetch_20newsgroups(subset='all',
                              remove=('headers', 'footers', 'quotes'))

    texts, y = ng20.data, ng20.target
    corpus = [corpus_preprocess(text) for text in texts]

    corpus, texts, y = filter_docs(corpus, texts, y,
                                   lambda doc: has_vector_representation(model, doc))

    x = []
    for doc in corpus:
        x.append(document_vector(model, doc))

    X = np.array(x)

    np.savez_compressed(out + prefix + 'w2v.npz',
                        train_data=X[:LEN_20NG_TRAIN],
                        test_data=X[LEN_20NG_TRAIN:],
                        train_labels=y[:LEN_20NG_TRAIN],
                        test_labels=y[LEN_20NG_TRAIN:])


# Real evidence (w: 5): root section clustering (comp,rec,sci,talk,misc)


def ng20_real5(out='./', prefix='20NG_', rate=None, skip=None):
    try:
        ng20 = np.load(out + prefix + 'w2v.npz')
    except:
        print out + prefix + 'w2v.npz not found, run  ./mk.py mk_20newsgroups_avgw2v'
        exit()
    train_labels = ng20['train_labels']
    test_labels = ng20['test_labels']
    train_x = ng20['train_data']
    test_x = ng20['test_data']

    tr_labels = []

    for i in train_labels:
        # COMP category
        if i in [1, 2, 3, 4, 5]:
            tr_labels.append(0)
        # REC category
        elif i in [7, 8, 9, 10]:
            tr_labels.append(1)
        # SCI category
        elif i in [11, 12, 13, 14]:
            tr_labels.append(2)
        # TALK category
        elif i in [16, 17, 18, 19]:
            tr_labels.append(3)
        # MISC category
        else:
            tr_labels.append(4)

    te_labels = []

    for i in test_labels:
        # COMP category
        if i in [1, 2, 3, 4, 5]:
            te_labels.append(0)
        # REC category
        elif i in [7, 8, 9, 10]:
            te_labels.append(1)
        # SCI category
        elif i in [11, 12, 13, 14]:
            te_labels.append(2)
        # TALK category
        elif i in [16, 17, 18, 19]:
            te_labels.append(3)
        # MISC category
        else:
            te_labels.append(4)

    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)

    if rate:
        tr_data, tr_labels, te_data, te_labels = create_partial2(train_x, tr_labels,
                test_x, te_labels, rate)
        np.savez_compressed(out + prefix + 'real5_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    elif skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(train_x, tr_labels,
                test_x, te_labels, skip)
        np.savez_compressed(out + prefix + 'real5_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    else:
        np.savez_compressed(out + prefix + 'real5.npz',
                            train_data=np.zeros(shape=(1, 1)),
                            train_labels=tr_labels,
                            test_data=np.zeros(shape=(1, 1)),
                            test_labels=te_labels)


# Real evidence (w: 6): Categories: sports, politics, religion, vehicles
#                                   systems, sci


def ng20_real6(out='./', prefix='20NG_', rate=None, skip=None):
    try:
        ng20 = np.load(out + prefix + 'w2v.npz')
    except:
        print out + prefix + 'w2v.npz not found, run  ./mk.py mk_20newsgroups_avgw2v'
        exit()
    train_labels = ng20['train_labels']
    test_labels = ng20['test_labels']
    train_x = ng20['train_data']
    test_x = ng20['test_data']

    tr_labels = []

    for i in train_labels:
        # Sports
        if i in [9, 10]:
            tr_labels.append(0)
        # Politics
        elif i in [16, 17, 18]:
            tr_labels.append(1)
        # Religion
        elif i in [15, 19, 0]:
            tr_labels.append(2)
        # Vehicles
        elif i in [7, 8, 6]:
            tr_labels.append(3)
        # Systems
        elif i in [3, 4, 5, 2]:
            tr_labels.append(4)
        # SCI
        else:
            tr_labels.append(5)

    te_labels = []

    for i in test_labels:
        # Sports
        if i in [9, 10]:
            te_labels.append(0)
        # Politics
        elif i in [16, 17, 18]:
            te_labels.append(1)
        # Religion
        elif i in [15, 19, 0]:
            te_labels.append(2)
        # Vehicles
        elif i in [7, 8, 6]:
            te_labels.append(3)
        # Systems
        elif i in [3, 4, 5, 2]:
            te_labels.append(4)
        # SCI
        else:
            te_labels.append(5)

    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)

    if rate:
        tr_data, tr_labels, te_data, te_labels = create_partial2(train_x, tr_labels,
                test_x, te_labels, rate)
        np.savez_compressed(out + prefix + 'real6_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    elif skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(train_x, tr_labels,
                test_x, te_labels, skip)
        np.savez_compressed(out + prefix + 'real6_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    else:
        np.savez_compressed(out + prefix + 'real6.npz',
                            train_data=np.zeros(shape=(1, 1)),
                            train_labels=tr_labels,
                            test_data=np.zeros(shape=(1, 1)),
                            test_labels=te_labels)

def ng20_real20(out='./', prefix='20NG_', rate=None, skip=None):
    try:
        ng20 = np.load(out + prefix + 'w2v.npz')
    except:
        print out + prefix + 'w2v.npz not found, run  ./mk.py mk_20newsgroups_avgw2v'
        exit()
    train_labels = ng20['train_labels']
    test_labels = ng20['test_labels']
    train_x = ng20['train_data']
    test_x = ng20['test_data']

    if rate:
        tr_data, tr_labels, te_data, te_labels = create_partial2(train_x, train_labels,
                test_x, test_labels, rate)
        np.savez_compressed(out + prefix + 'real20_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    elif skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(train_x, train_labels,
                test_x, test_labels, skip)
        np.savez_compressed(out + prefix + 'real20_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )


# Random evidence (w:3) = white noise


def ng20_rand3(out='./', prefix='20NG_'):
    np.savez_compressed(out + prefix + 'rand3.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=np.random.uniform(
                            size=(LEN_20NG_TRAIN, 3)),
                        test_labels=np.random.uniform(
                            size=(LEN_20NG_TEST, 3)),
                        test_data=np.zeros(shape=(1, 1))
                        )

# Random evidence (w:10) = white noise


def ng20_rand10(out='./', prefix='20NG_'):
    np.savez_compressed(out + prefix + 'rand10.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=np.random.uniform(
                            size=(LEN_20NG_TRAIN, 10)),
                        test_labels=np.random.uniform(
                            size=(LEN_20NG_TEST, 10)),
                        test_data=np.zeros(shape=(1, 1))
                        )


# Real evidence (w:5) in random order


def ng20_rorder5(out='./', prefix='20NG_'):
    try:
        ng20_real5 = np.load(out + prefix + 'real5.npz')
    except:
        print out + prefix + 'real5.npz not found, run  ./mk.py ng20_real5'
        exit()

    train_labels = ng20_real5['train_labels']
    test_labels = ng20_real5['test_labels']

    np.random.shuffle(train_labels)
    np.random.shuffle(test_labels)

    np.savez_compressed(out + prefix + 'rorder5.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=train_labels,
                        test_data=np.zeros(shape=(1, 1)),
                        test_labels=test_labels)


# Real evidence (w:20) in random order


def ng20_rorder20(out='./', prefix='20NG_'):
    try:
        ng20 = np.load(out + prefix + 'w2v.npz')
    except:
        print out + prefix + 'w2v.npz not found, run  ./mk.py mk_20newsgroups_avgw2v'
        exit()

    train_labels = ng20['train_labels']
    test_labels = ng20['test_labels']

    np.random.shuffle(train_labels)
    np.random.shuffle(test_labels)

    np.savez_compressed(out + prefix + 'rorder20.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=train_labels,
                        test_data=np.zeros(shape=(1, 1)),
                        test_labels=test_labels)

# Make reuters data from
# https://github.com/XifengGuo/DEC-keras


def mk_reuters_data(data_dir='./', rate=None, skip=None):
    from os.path import join
    from sklearn.feature_extraction.text import CountVectorizer
    import os
    import wget

    for c, i in enumerate(REU_URLS):
        if os.path.exists(REU_DAT_gz[c].split('.gz')[0]):
            continue
        else:
            wget.download(i)
            os.system('gunzip ' + REU_DAT_gz[c])

    print 'Generating TF-IDF features....'

    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]
    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    ids = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                            ids.append(did)
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)

    X = x.todense()
    y = np.asarray(y)
    ids = np.asarray(ids)

    np.random.seed(1234)
    p = np.random.permutation(X.shape[0])
    np.save('document_vectors.npy', X[p])
    np.save('document_labels.npy', y[p])
    np.save('document_ids.npy', ids[p])
    np.save('perm.npy', p)
    mk_reuters_sublabels(rate=rate, skip=skip)

    return data, target, ids



def mk_reuters_sublabels(data_dir='./', rate=None, skip=None):
    from os.path import join
    import pickle
    from collections import defaultdict
    d = {}
    llst = []
    prev = None
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            llst.append(cat)
            if prev != did and not(prev is None):
                d[did] = llst
                llst = []
            prev = did

    ids = np.load('document_ids.npy')
    multi = []

    for i in ids:
        tmp = d[i]
        for t in tmp:
            if t in ['CCAT', 'GCAT', 'MCAT', 'ECAT']:
                tmp.remove(t)
        multi.append(tmp)

    keep = ['C15', 'C151', 'GPOL', 'GSPO', 'GDIP', 'E51',
            'M11', 'M14', 'E21', 'E41']

    multi2 = defaultdict(list)

    for k in keep:
        multi2[k] = []

    for c, m in enumerate(multi):
        if len(list(set(keep) & set(m))) == 1:
            lab = list(set(keep) & set(m))[0]
            if len(multi2[lab]) == (REUTERS_COUNT / REUTERS_LABELS):
                pass
            else:
                multi2[lab].append(ids[c])

    for m in multi2:
        print m, len(multi2[m])

    with open('l10.pkl', 'wb') as pout:
        pickle.dump(multi2, pout, protocol=pickle.HIGHEST_PROTOCOL)

    mk_reuters_subset(rate=rate, skip=skip)


def mk_reuters_subset(out='./', prefix='REU_', rate=None, skip=None):
    import pickle
    np.random.seed(1234)

    CAT = ['C', 'G', 'M', 'E']

    ids = np.load('document_ids.npy')
    x = np.load('document_vectors.npy')

    with open('l10.pkl', 'rb') as pin:
        l10 = pickle.load(pin)

    idc = []
    labels10 = []
    labels4 = []
    for c, l in enumerate(l10):
        lst = l10[l]
        for ll in lst:
            idc.append(np.where(ll == ids))
            labels10.append(c)
            labels4.append(CAT.index(l[0]))

    p = np.random.permutation(len(idc))
    dat = x[idc].reshape(len(idc), x.shape[1])

    # REAL 10 evidence
    labels10 = np.array(labels10)

    # REAL 4 evidence
    labels4 = np.array(labels4)

    dat = dat[p]
    labels10 = labels10[p]
    labels4 = labels4[p]

    if rate:
        tr_data, tr_labels, te_data, te_labels = create_partial2(dat[:LEN_REUTERS_TRAIN],
                 labels10[:LEN_REUTERS_TRAIN], dat[LEN_REUTERS_TRAIN:], labels10[LEN_REUTERS_TRAIN:], 
                 rate)
        np.savez_compressed(out + prefix + 'real10_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
        tr_data, tr_labels, te_data, te_labels = create_partial2(dat[:LEN_REUTERS_TRAIN],
                 labels4[:LEN_REUTERS_TRAIN], dat[LEN_REUTERS_TRAIN:], labels4[LEN_REUTERS_TRAIN:], 
                 rate)
        np.savez_compressed(out + prefix + 'real4_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    elif skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(dat[:LEN_REUTERS_TRAIN],
                                                 labels10[:LEN_REUTERS_TRAIN],
                                                 dat[LEN_REUTERS_TRAIN:],
                                                 labels10[LEN_REUTERS_TRAIN:],
                                                 skip)
        np.savez_compressed(out + prefix + 'real10_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
        tr_data, tr_labels, te_data, te_labels = skip_class(dat[:LEN_REUTERS_TRAIN],
                                                 labels4[:LEN_REUTERS_TRAIN],
                                                 dat[LEN_REUTERS_TRAIN:],
                                                 labels4[LEN_REUTERS_TRAIN:],
                                                 skip)
        np.savez_compressed(out + prefix + 'real4_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )

    else:
        np.savez_compressed(out + prefix + 'real10.npz',
                            train_data=dat[:LEN_REUTERS_TRAIN],
                            test_data=dat[LEN_REUTERS_TRAIN:],
                            train_labels=labels10[:LEN_REUTERS_TRAIN],
                            test_labels=labels10[LEN_REUTERS_TRAIN:])

        np.savez_compressed(out + prefix + 'real4.npz',
                            train_data=dat[:LEN_REUTERS_TRAIN], 
                            test_data=dat[LEN_REUTERS_TRAIN:],
                            train_labels=labels4[:LEN_REUTERS_TRAIN],
                            test_labels=labels4[LEN_REUTERS_TRAIN:])

# Real evidence (w:5), label number mod 5

def reuters_real5(out='./', prefix='REU_', rate=None, skip=None):
    try:
        rw2v = np.load(out + prefix + 'real10.npz')
    except:
        print out + prefix + '_real10.npz not found, run  ./mk.py mk_reuters_avgw2v'
        exit()

    train_labels = rw2v['train_labels']
    test_labels = rw2v['test_labels']
    train_x = rw2v['train_data']
    test_x = rw2v['test_data']


    tr_labels = []

    for i in train_labels:
        tr_labels.append(i%5)

    te_labels = []

    for i in test_labels:
        te_labels.append(i%5)

    if rate:
        tr_data, tr_labels, te_data, te_labels = create_partial2(train_x, tr_labels,
                test_x, te_labels, rate)
        np.savez_compressed(out + prefix + 'real5_'+str(rate)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    elif skip:
        tr_data, tr_labels, te_data, te_labels = skip_class(train_x, tr_labels,
                test_x, te_labels, skip)
        np.savez_compressed(out + prefix + 'real5_skip'+str(skip)+'.npz',
                            train_data=tr_data,
                            train_labels=tr_labels, test_labels=te_labels,
                            test_data=te_data
                            )
    else:
        np.savez_compressed(out + prefix + 'real5.npz',
                            train_data=np.zeros(shape=(1, 1)), 
                            test_data=np.zeros(shape=(1, 1)),
                            train_labels=tr_labels,
                            test_labels=te_labels)


# Random evidence (w:3) = white noise


def reuters_rand3(out='./', prefix='REU_'):
    np.savez_compressed(out + prefix + 'rand3.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=np.random.uniform(
                            size=(LEN_REUTERS_TRAIN, 3)),
                        test_labels=np.random.uniform(
                            size=(LEN_REUTERS_TEST, 3)),
                        test_data=np.zeros(shape=(1, 1))
                        )

# Random evidence (w:10) = white noise


def reuters_rand10(out='./', prefix='REU_'):
    np.savez_compressed(out + prefix + 'rand10.npz',
                        train_data=np.zeros(shape=(1, 1)),
                        train_labels=np.random.uniform(
                            size=(LEN_REUTERS_TRAIN, 10)),
                        test_labels=np.random.uniform(
                            size=(LEN_REUTERS_TEST, 10)),
                        test_data=np.zeros(shape=(1, 1))
                        )

# Real evidence (w:4) in random order

def reuters_rorder4(out='./', prefix='REU_'):
    try:
        rw2v = np.load(out + prefix + 'real4.npz')
    except:
        print out + prefix + '_real4.npz not found, run  ./mk.py mk_reuters_avgw2v'
        exit()

    train_labels = rw2v['train_labels']
    test_labels = rw2v['test_labels']

    np.random.shuffle(train_labels)
    np.random.shuffle(test_labels)

    np.savez_compressed(out + prefix + 'rorder4.npz',
                        train_data=np.zeros(shape=(1, 1)), 
                        test_data=np.zeros(shape=(1, 1)),
                        train_labels=train_labels,
                        test_labels=test_labels)

# Real evidence (w:10) in random order

def reuters_rorder10(out='./', prefix='REU_'):
    try:
        rw2v = np.load(out + prefix + 'real10.npz')
    except:
        print out + prefix + '_real10.npz not found, run  ./mk.py mk_reuters_avgw2v'
        exit()

    train_labels = rw2v['train_labels']
    test_labels = rw2v['test_labels']

    np.random.shuffle(train_labels)
    np.random.shuffle(test_labels)

    np.savez_compressed(out + prefix + 'rorder10.npz',
                        train_data=np.zeros(shape=(1, 1)), 
                        test_data=np.zeros(shape=(1, 1)),
                        train_labels=train_labels,
                        test_labels=test_labels)

# Make all evidence for MNIST dataset

def mnist():
    mnist_real3(rate=0.1)
    mnist_real4(rate=0.1)
    mnist_real10(rate=0.1)
    mnist_real3(rate=0.3)
    mnist_real4(rate=0.3)
    mnist_real10(rate=0.3)
    mnist_real3(skip=1)
    mnist_real4(skip=1)
    mnist_real10(skip=1)
    mnist_real3(skip=2)
    mnist_real4(skip=2)
    mnist_real10(skip=2)

    #  if rate:
        #  mnist_real10(rate=rate)
    #  mnist_rand3()
    #  mnist_rand10()
    #  mnist_rorder3()
    #  mnist_rorder10()

# Make all evidence for CIFAR-10 dataset

def cifar():
    mk_cifar10_vgg()

    cifar10_real3(rate=0.1)
    cifar10_real4(rate=0.1)
    cifar10_real5(rate=0.1)
    cifar10_real10(rate=0.1)

    cifar10_real3(rate=0.3)
    cifar10_real4(rate=0.3)
    cifar10_real5(rate=0.3)
    cifar10_real10(rate=0.3)

    cifar10_real3(skip=1)
    cifar10_real4(skip=1)
    cifar10_real5(skip=1)
    cifar10_real10(skip=1)

    cifar10_real3(skip=2)
    cifar10_real4(skip=2)
    cifar10_real5(skip=2)
    cifar10_real10(skip=2)

# Make all evidence for 20 Newsgroups dataset

def ng20(rate=None):
    mk_20newsgroups_avgw2v()
    ng20_real5(rate=0.1)
    ng20_real6(rate=0.1)
    ng20_real20(rate=0.1)

    ng20_real5(rate=0.3)
    ng20_real6(rate=0.3)
    ng20_real20(rate=0.3)

    ng20_real5(skip=1)
    ng20_real6(skip=1)
    ng20_real20(skip=1)

    ng20_real5(skip=2)
    ng20_real6(skip=2)
    ng20_real20(skip=2)

    #  ng20_rand3()
    #  ng20_rand10()
    #  ng20_rorder5()
    #  ng20_rorder20()

# Make all evidence for 20 Newsgroups dataset

def reuters(rate=None):
    #  mk_reuters_data(rate=0.1)
    mk_reuters_subset()
    mk_reuters_data(rate=0.1)
    mk_reuters_subset(rate=0.3)
    mk_reuters_subset(skip=1)
    mk_reuters_subset(skip=2)

    reuters_real5(rate=0.1)
    reuters_real5(rate=0.3)
    reuters_real5(skip=1)
    reuters_real5(skip=2)

    #  reuters_rand3()
    #  reuters_rand10()
    #  reuters_rorder4()
    #  reuters_rorder10()

if __name__ == "__main__":
    try:
        funcs = {}
        for key, value in locals().items():
            if callable(value) and value.__module__ == __name__:
                funcs[key] = value
        try:
            funcs[sys.argv[1]](rate=sys.argv[2])
        except IndexError:
            funcs[sys.argv[1]]()
    except KeyError:
        print 'Did not find make dataset function named: ' + sys.argv[1]
