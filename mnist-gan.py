"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import InputLayer, Conv2D, LeakyReLU, Dropout, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as mp

################################################################################
# %% CONSTANTS
################################################################################

EPOCHS = 3
BATCH_SIZE = 128
BATCHES = 545
IMG_SHP = (28, 28, 1)
LAT_SHP = 100

################################################################################
# %% BUILD DESCRIMINATOR MODEL (FOR TRAINING)
################################################################################

d_model = Sequential()
d_model.add(InputLayer(input_shape=IMG_SHP))
d_model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
d_model.add(LeakyReLU(alpha=0.2))
d_model.add(Dropout(0.4))
d_model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
d_model.add(LeakyReLU(alpha=0.2))
d_model.add(Dropout(0.4))
d_model.add(Flatten())
d_model.add(Dense(11, activation='softmax'))
d_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

################################################################################
# %% BUILD GENERATOR MODEL (FOR PREDICTING)
################################################################################

g_model = Sequential()
g_model.add(InputLayer(input_shape=LAT_SHP))
g_model.add(Dense(7*7*128))
g_model.add(LeakyReLU(alpha=0.2))
g_model.add(Reshape((7, 7, 128)))
g_model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
g_model.add(LeakyReLU(alpha=0.2))
g_model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
g_model.add(LeakyReLU(alpha=0.2))
g_model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))

################################################################################
# %% COMBINED MODEL (FOR TRAINING GENERATOR)
################################################################################

d_fix_model = clone_model(d_model)
d_fix_model.trainable = False
gan_model = Sequential()
gan_model.add(g_model)
gan_model.add(d_fix_model)
gan_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metric=['accuracy'])

loss = []

################################################################################
# %% RUN THROUGH EPOCHS
################################################################################

for epoch in range(EPOCHS):

    print(f'Epoch: {epoch+1}')

    ############################################################################
    # GENERATE REAL DATA
    ############################################################################

    ##### LOAD REAL DATA
    (X_train, y_train), (X_test, y_test) = load_data()

    ##### COMBINE TRAIN AND TEST
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    ##### SHUFFLE DATA
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    ################################################################################
    # RUN THROUGH BATCHES
    ################################################################################

    for batch in range(BATCHES):

        ##### GENERATE REAL BATCH DATA
        X_real = X[:BATCH_SIZE, :, :, np.newaxis]
        y_real = y[:BATCH_SIZE]
        y_real_oh = np.zeros((BATCH_SIZE, 11), dtype=int)
        y_real_oh[np.arange(BATCH_SIZE), y_real] = 1

        ##### REMOVE PROCESSED BATCH
        X = np.delete(X, range(BATCH_SIZE), axis=0)
        y = np.delete(y, range(BATCH_SIZE), axis=0)

        ############################################################################
        # GENERATE FAKE DATA
        ############################################################################

        ##### GENERATE RANDOM DIGITS
        idx = np.random.randint(0, high=10, size=BATCH_SIZE, dtype='int')

        ##### INIT EMPTY INPUT ARRAY
        X_in_p = np.zeros((BATCH_SIZE, 10), dtype=int)

        ##### ONE HOT DIGITS TO INPUT
        X_in_p[np.arange(BATCH_SIZE), idx] = 1

        ##### FILL REST WITH RANDOM NUMBERS
        X_in_r = np.random.randn(BATCH_SIZE, LAT_SHP-10)
        X_in = np.concatenate((X_in_p, X_in_r), axis=1)

        ##### PREDICT IMAGE FROM RANDOM INPUT
        X_fake = g_model.predict(X_in)

        ##### SET TARGET TO FAKE
        y_fake_oh = np.zeros((BATCH_SIZE, 11), dtype=int)
        y_fake_oh[np.arange(BATCH_SIZE), 10] = 1

        ############################################################################
        # TRAIN DESCRIMINATOR
        ############################################################################

        X_batch = np.concatenate((X_real, X_fake), axis=0)
        y_batch = np.concatenate((y_real_oh, y_fake_oh), axis=0)
        d_loss, d_acc = d_model.train_on_batch(X_batch, y_batch)

        ############################################################################
        # GENERATE RANDOM/PARAMETERIZED GENERATOR INPUT
        ############################################################################

        ##### GENERATE RANDOM DIGITS
        idx = np.random.randint(0, high=10, size=2*BATCH_SIZE, dtype='int')

        ##### INIT EMPTY INPUT AND TARGET ARRAYS
        X_in_p = np.zeros((2*BATCH_SIZE, 10), dtype=int)
        y_gan_oh = np.zeros((2*BATCH_SIZE, 11), dtype=int)

        ##### ONE HOT DIGITS TO INPUT AND OUTPUT
        X_in_p[np.arange(2*BATCH_SIZE), idx] = 1
        y_gan_oh[np.arange(2*BATCH_SIZE), idx] = 1

        ##### FILL REST WITH RANDOM NUMBERS
        X_in_r = np.random.randn(2*BATCH_SIZE, LAT_SHP-10)
        X_in = np.concatenate((X_in_p, X_in_r), axis=1)

        ############################################################################
        # TRAIN GENERATOR ON DESCRIMINATOR (FIXED) ERROR
        ############################################################################

        ##### GET WEIGHTS FROM LAST TRAINING STEP
        d_fix_model.set_weights(d_model.get_weights())
        g_loss = gan_model.train_on_batch(X_in, y_gan_oh)

        print(f'd_loss: {d_loss}, g_loss: {g_loss}')

        loss.append([d_loss, g_loss])

    print(f'Accuracy: {d_acc}')
