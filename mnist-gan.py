"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
from tensorflow.keras.models import Sequential, clone_model, Model
from tensorflow.keras.layers import InputLayer, Conv2D, LeakyReLU, Dropout, Flatten, Dense, Reshape, Conv2DTranspose, Input, concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as mp

##### WHICH DATA LOADER TO USE
dataset = 'mnist'

if dataset == 'mnist':
    from keras.datasets.mnist import load_data
elif dataset == 'fmnist':
    from keras.datasets.fashion_mnist import load_data
elif dataset == 'emnist':
    from matlab.loader import load_data

################################################################################
# %% CONSTANTS
################################################################################

EPOCHS = 100
BATCH_SIZE = 128
IMG_SHP = (28, 28, 1)
LAT_SHP = 100

################################################################################
# %% DEFINE DISCRIMINATOR MODEL (FOR TRAINING)
################################################################################

def build_descriminator(IMG_SHP, CAT_SHP):
    input1 = Input(IMG_SHP)
    layer1 = Conv2D(64, (3,3), strides=(2, 2), padding='same')(input1)
    layer2 = LeakyReLU(alpha=0.2)(layer1)
    layer3 = Dropout(0.4)(layer2)
    layer4 = Conv2D(64, (3,3), strides=(2, 2), padding='same')(layer3)
    layer5 = LeakyReLU(alpha=0.2)(layer4)
    layer6 = Dropout(0.4)(layer5)
    layer7 = Flatten()(layer6)
    ##### PREDICT TRUE/FALSE
    out1 = Dense(1, activation='sigmoid')(layer7)
    ##### PRECIT DIGIT
    out2 = Dense(CAT_SHP, activation='softmax')(layer7)
    model = Model(inputs = input1, outputs = [out1, out2])
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model

################################################################################
# %% BUILD GENERATOR MODEL (FOR PREDICTING)
################################################################################

def build_generator(LAT_SHP, CAT_SHP):
    input1 = Input(CAT_SHP)
    input2 = Input(LAT_SHP-CAT_SHP)
    inputs = concatenate([input1, input2], axis=-1)
    layer1 = Dense(7*7*128)(inputs)
    layer2 = LeakyReLU(alpha=0.2)(layer1)
    layer3 = Reshape((7, 7, 128))(layer2)
    layer4 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(layer3)
    layer5 = LeakyReLU(alpha=0.2)(layer4)
    layer6 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(layer5)
    layer7 = LeakyReLU(alpha=0.2)(layer6)
    output = Conv2D(1, (7,7), activation='sigmoid', padding='same')(layer7)
    model = Model(inputs = [input1, input2], outputs = output)
    return model

################################################################################
# %% COMBINED MODEL (FOR TRAINING GENERATOR)
################################################################################

def build_gan(d_model, g_model):
    d_model.trainable = False
    input1 = Input(CAT_SHP)
    input2 = Input(LAT_SHP-CAT_SHP)
    layer1 = g_model([input1, input2])
    output = d_model(layer1)
    model = Model(inputs = [input1, input2], outputs = output)
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model

################################################################################
# %% INIT ARRAYS FOR POSTPROCESSING
################################################################################

loss = []

################################################################################
# %% RUN THROUGH EPOCHS
################################################################################

for epoch in range(EPOCHS):

    print(f'Epoch: {epoch+1}')

    ############################################################################
    # GENERATE REAL DATA
    ############################################################################

    ##### LOAD REAL MNIST/FMNIST DATA
    (X_train, y_train), (X_test, y_test) = load_data()

    ##### COMBINE TRAIN AND TEST
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    """
    ##### LIMIT TO FIRST CAT_SHP SYMBOLS (TO REDUCE CLASSES FOR TESTING)
    idx = np.where(y<10)[0]
    X = X[idx]
    y = y[idx]
    """

    ##### SHUFFLE DATA
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    assert y.min() == 0, "Class labels are expected to start at 0"

    ##### CALCULATE NUMBER OF BATCHES
    BATCHES = int(len(X)/BATCH_SIZE)

    ##### GET NUMBER OF CLASSES
    CAT_SHP = int(y.max()+1)

    ##### BUILD MODELS
    d_model = build_descriminator(IMG_SHP, CAT_SHP)
    g_model = build_generator(LAT_SHP, CAT_SHP)
    gan_model = build_gan(d_model, g_model)

    ################################################################################
    # RUN THROUGH BATCHES
    ################################################################################

    for batch in range(BATCHES):

        ##### GENERATE REAL BATCH DATA
        X_real = X[:BATCH_SIZE, :, :, np.newaxis]/255
        y_real = y[:BATCH_SIZE]
        y_real_oh = np.zeros((BATCH_SIZE, CAT_SHP), dtype=int)
        y_real_oh[np.arange(BATCH_SIZE), y_real] = 1
        z_real = np.zeros((BATCH_SIZE, 1), dtype=int)

        ##### REMOVE PROCESSED BATCH
        X = np.delete(X, range(BATCH_SIZE), axis=0)
        y = np.delete(y, range(BATCH_SIZE), axis=0)

        ############################################################################
        # GENERATE FAKE DATA
        ############################################################################

        ##### GENERATE RANDOM DIGITS
        idx = np.random.randint(0, high=CAT_SHP, size=BATCH_SIZE, dtype='int')

        ##### INIT EMPTY INPUT ARRAY
        X_in_p = np.zeros((BATCH_SIZE, CAT_SHP), dtype=float)

        ##### ONE HOT DIGITS TO INPUT
        X_in_p[np.arange(BATCH_SIZE), idx] = 1.0

        ##### FILL REST WITH RANDOM NUMBERS
        X_in_r = np.random.randn(BATCH_SIZE, LAT_SHP-CAT_SHP)

        ##### PREDICT IMAGE FROM RANDOM INPUT
        X_fake = g_model.predict([X_in_p, X_in_r])

        ##### SET TARGET TO FAKE
        y_fake_oh = np.zeros((BATCH_SIZE, CAT_SHP), dtype=int)
        y_fake_oh[np.arange(BATCH_SIZE), idx] = 1
        z_fake = np.ones((BATCH_SIZE, 1), dtype=int)

        ############################################################################
        # TRAIN DISCRIMINATOR
        ############################################################################

        X_batch = np.concatenate((X_real, X_fake), axis=0)
        y_batch = np.concatenate((y_real_oh, y_fake_oh), axis=0)
        z_batch = np.concatenate((z_real, z_fake), axis=0)

        d_loss, d_acc, _ = d_model.train_on_batch(X_batch, [z_batch, y_batch])

        ############################################################################
        # GENERATE RANDOM/PARAMETERIZED GENERATOR INPUT
        ############################################################################

        ##### GENERATE RANDOM DIGITS
        idx = np.random.randint(0, high=CAT_SHP, size=2*BATCH_SIZE, dtype='int')

        ##### INIT EMPTY INPUT AND TARGET ARRAYS
        X_in_p = np.zeros((2*BATCH_SIZE, CAT_SHP), dtype=float)

        ##### ONE HOT DIGITS TO INPUT AND OUTPUT
        X_in_p[np.arange(2*BATCH_SIZE), idx] = 1.0

        ##### FILL REST WITH RANDOM NUMBERS
        X_in_r = np.random.randn(2*BATCH_SIZE, LAT_SHP-CAT_SHP)

        ##### BUILD OUTPUT ARRAY
        y_gan_oh = np.zeros((2*BATCH_SIZE, CAT_SHP), dtype=int)
        y_gan_oh[np.arange(2*BATCH_SIZE), idx] = 1
        z_gan = np.zeros((2*BATCH_SIZE, 1), dtype=int)

        ############################################################################
        # TRAIN GENERATOR ON DISCRIMINATOR (FIXED) ERROR
        ############################################################################

        ##### GET WEIGHTS FROM LAST TRAINING STEP
        g_loss, g_acc, _ = gan_model.train_on_batch([X_in_p, X_in_r], [z_gan, y_gan_oh])

        print(f'd_loss: {d_loss:.3f}, g_loss: {g_loss:.3f}')

        loss.append([d_loss, g_loss])

    print(f'Accuracy: Discriminator {d_acc:.3f} Generator {g_acc:.3f}')

############################################################################
# %% PLOT LOSS CURVES
############################################################################

mp.semilogy(loss)
mp.show()

############################################################################
# %% SAVE MODELS
############################################################################

g_model.save('gen_model.h5')
d_model.save('dis_model.h5')
gan_model.save('gan_model.h5')

############################################################################
# %% TEST ON INPUT STRING
############################################################################

input_string = [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9]
X_in_p = np.zeros((len(input_string), CAT_SHP))
X_in_p[np.arange(len(input_string)), input_string] = 1
X_in_r = np.random.randn(len(input_string), LAT_SHP-CAT_SHP)
img = g_model.predict([X_in_p, X_in_r])
out = img.reshape(28, len(input_string)*28)
mp.imshow(out, cmap='gray_r')
