"""
Script to load e-mnist data in a comparable way to the tf.keras dataset loaders
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
from scipy.io import loadmat

def load_data():
    ##### LOAD REAL EMNIST LETTERS DATA
    data = loadmat('matlab/emnist-letters.mat')
    #data = loadmat('drive/My Drive/Data/emnist-letters.mat')
    X_train = data['dataset'][0][0][0][0][0][0]
    y_train = data['dataset'][0][0][0][0][0][1]
    X_test = data['dataset'][0][0][1][0][0][0]
    y_test = data['dataset'][0][0][1][0][0][1]

    X_train = X_train.reshape(-1,28,28).transpose(0,2,1)
    X_test = X_test.reshape(-1,28,28).transpose(0,2,1)
    if y_train.min() == 1:
      y_train = y_train-1
      y_test = y_test-1

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    return (X_train, y_train), (X_test, y_test)
