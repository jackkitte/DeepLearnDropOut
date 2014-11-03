import numpy as np
import scipy.io as sio
import cPickle
import gzip
import os
import sys

import theano
import theano.tensor as T

def _shared_dataset(data_x, data_y):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def _shared_dataset_y_matrix(data_y1, data_y2):

    shared_y1 = theano.shared(np.asarray(data_y1,
                                           dtype=theano.config.floatX))
    shared_y2 = theano.shared(np.asarray(data_y2,
                                           dtype=theano.config.floatX))

    return shared_y1, shared_y2

def load_mnist(path):
    mnist = np.load(path)
    train_set_x = mnist['train_data']
    train_set_y = mnist['train_labels']
    test_set_x = mnist['test_data']
    test_set_y = mnist['test_labels']

    train_set_x, train_set_y = _shared_dataset((train_set_x, train_set_y))
    test_set_x, test_set_y = _shared_dataset((test_set_x, test_set_y))
    valid_set_x, valid_set_y = test_set_x, test_set_y
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_initial_params_data(dataset):

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], "data", dataset)
        if os.path.isfile(new_path) or data_file == "VisionHogFeatures.mat":
            dataset = new_path

    print '... loading data'

    mat = sio.loadmat(dataset)
    mats = mat.values()
    epochs = mats[0]
    param1 = theano.shared(np.asarray(mats[2][0,0], dtype=theano.config.floatX))
    param2 = theano.shared(np.asarray(mats[2][0,1], dtype=theano.config.floatX))
    params = [param1, param2]

    return params


def load_umontreal_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], "data", dataset)
        if os.path.isfile(new_path) or data_file == "VisionHogFeatures.mat":
            dataset = new_path

    print '... loading data'

    mat = sio.loadmat(dataset)
    testY, testX, a, b, c, trainX, trainY = mat.values()
    testX = np.array(testX, dtype=np.float32)
    trainX = np.array(trainX, dtype=np.float32)
    testY_matrix = np.array(testY, dtype=np.float32)
    trainY_matrix = np.array(trainY, dtype=np.float32)
    testY = np.nonzero(testY)[1]
    trainY = np.nonzero(trainY)[1]
    n_test = len(testX)
    n_train = len(trainX)
    shuffle_test = np.arange(n_test)
    shuffle_train = np.arange(n_train)
    np.random.seed(1)
    np.random.shuffle(shuffle_test)
    np.random.shuffle(shuffle_train)
    testX = testX[shuffle_test]
    testY = testY[shuffle_test]
    testY_matrix = testY_matrix[shuffle_test]
    trainY_matrix = trainY_matrix[shuffle_train]
    trainX = trainX[shuffle_train]
    trainY = trainY[shuffle_train]

    test_set_x, test_set_y = _shared_dataset(testX, testY)
    train_set_x, train_set_y = _shared_dataset(trainX, trainY)
    test_set_matrixY, train_set_matrixY = _shared_dataset_y_matrix(testY_matrix, trainY_matrix)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y), (test_set_matrixY, train_set_matrixY)]
    return rval

