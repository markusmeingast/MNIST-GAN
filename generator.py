"""
Generator functions to supply training data batches for the different cases.
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np

from keras.datasets.mnist import load_data as load_data_mnist
from keras.datasets.fashion_mnist import load_data as load_data_fmnist
from emnist_loader import load_data as load_data_emnist

def image_generator(DATASET='mnist', BATCH_SIZE=128, CAT_SHP=10):

    ##### CLASSIC MNIST DATASET
    if DATASET == 'mnist':

        ##### LOAD DATA
        (X_train, y_train), (X_test, y_test) = load_data_mnist()

        ##### COMBINE TEST AND TRAINING DATA
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

    ##### FASHION MNIST DATASET
    elif DATASET == 'fmnist':

        ##### LOAD DATA
        (X_train, y_train), (X_test, y_test) = load_data_fmnist()

        ##### COMBINE TEST AND TRAINING DATA
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

    ##### EXTENDED MNIST DATASET
    elif DATASET == 'emnist':

        ##### LOAD DATA
        (X_train, y_train), (X_test, y_test) = load_data_emnist()

        ##### COMBINE TEST AND TRAINING DATA
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        ##### BALANCE THE DATASET
        minlen = 999999
        cats = len(np.unique(y))

        for i in range(cats):
            idx = np.where(y == i)[0]
            minlen = min(len(idx), minlen)

        X_new = np.zeros((minlen*cats, 28, 28), dtype=int)
        y_new = np.zeros((minlen*cats,), dtype=int)

        for i in range(cats):
            idx = np.where(y == i)[0]
            X_new[i*minlen:(i+1)*minlen] = X[idx[0:minlen]]
            y_new[i*minlen:(i+1)*minlen] = y[idx[0:minlen]]

        X = X_new
        y = y_new

    ##### FUTURE DATASETS TO BE INCLUDED HERE
    else:
        print('Dataset not recognized')
        exit()

    ##### SHUFFLE DATA
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    ##### TESTING
    assert y.min() == 0, "Class labels are expected to start at 0"
    assert X.shape[1:] == (28, 28), f"Expecting images of shape 28 x 28 x 1, got {X.shape[1:]}"

    ##### CALCULATE NUMBER OF BATCHES
    BATCHES = int(len(X)/BATCH_SIZE)

    ##### FIRST YIELD NUMBER OF BATCHES
    yield BATCHES

    ##### LOOP THROUGH BATCHES
    while True:

        ##### GENERATE REAL BATCH DATA
        X_real = X[:BATCH_SIZE, :, :, np.newaxis]/127.5-1.0
        y_real = y[:BATCH_SIZE]
        y_real_oh = np.zeros((BATCH_SIZE, CAT_SHP), dtype=int)
        y_real_oh[np.arange(BATCH_SIZE), y_real] = 1
        w_real = np.ones((BATCH_SIZE, 1), dtype=int)

        ##### REMOVE PROCESSED BATCH
        X = np.delete(X, range(BATCH_SIZE), axis=0)
        y = np.delete(y, range(BATCH_SIZE), axis=0)

        yield X_real, y_real_oh, w_real
