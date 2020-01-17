"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
from tensorflow.keras.models import Sequential, clone_model, Model
from tensorflow.keras.layers import InputLayer, Conv2D, LeakyReLU, Dropout, Flatten, Dense, Reshape, Conv2DTranspose, Input, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as mp
import generator

################################################################################
# %% CONSTANTS
################################################################################

DATASET = 'mnist'
EPOCHS = 5
BATCH_SIZE = 128
IMG_SHP = (28, 28, 1)
CAT_SHP = 62
LAT_SHP = 100

################################################################################
# %% DEFINE DISCRIMINATOR MODEL (FOR TRAINING)
################################################################################

def build_descriminator(IMG_SHP, CAT_SHP):

    ##### INPUT IMAGE
    input = Input(IMG_SHP)

    ##### CONV2D LAYER
    net = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(input)
    net = LeakyReLU()(net)
    net = Dropout(0.4)(net)

    ##### CONV2D LAYER
    net = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(net)
    net = LeakyReLU()(net)
    net = Dropout(0.4)(net)

    ##### TO DENSE
    net = Flatten()(net)

    ##### PREDICT TRUE/FALSE
    out1 = Dense(1, activation='sigmoid')(net)

    ##### PRECIT DIGIT
    out2 = Dense(CAT_SHP, activation='softmax')(net)

    ##### BUILD MODEL AND COMPILE
    model = Model(inputs = input, outputs = [out1, out2])
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=Adam(lr=0.0002, beta_1=0.5))

    return model

################################################################################
# %% BUILD GENERATOR MODEL (FOR PREDICTING)
################################################################################

def build_generator(LAT_SHP, CAT_SHP):

    ##### INPUT DIGIT ONE-HOT
    input1 = Input(CAT_SHP)

    ##### INPUT RANDOM LATENT NOISE
    input2 = Input(LAT_SHP-CAT_SHP)

    ##### COMBINE RANDOM AND ONE-HOT INPUTS
    inputs = concatenate([input1, input2], axis=-1)

    ##### DENSE LAYER
    net = Dense(7*7*64)(inputs)
    net = LeakyReLU()(net)
    net = Dropout(0.4)(net)

    ##### TO CONV2D
    net = Reshape((7, 7, 64))(net)

    ##### CONV2D.T LAYER
    net = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(net)
    net = LeakyReLU()(net)
    net = Dropout(0.4)(net)

    ##### CONV2D.T LAYER
    net = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(net)
    net = LeakyReLU()(net)
    net = Dropout(0.4)(net)

    ##### OUTPUT IMAGE
    output = Conv2D(1, (7, 7), activation='tanh', padding='same')(net)

    ##### BUILD MODEL (COMPILATION IN GAN MODEL)
    model = Model(inputs = [input1, input2], outputs = output)

    return model

################################################################################
# %% COMBINED MODEL (FOR TRAINING GENERATOR)
################################################################################

def build_gan(d_model, g_model):

    ##### ONLY GENERATOR SHOULD BE TRAINED BASED ON DISCRIMINATOR OUTPUT
    d_model.trainable = False

    ##### NUMBER OF CLASSES TO COVER
    input1 = Input(CAT_SHP)

    ##### FILL REMAINDER WITH RANDOM NOISE
    input2 = Input(LAT_SHP-CAT_SHP)

    ##### PREDICT IMAGE
    layer1 = g_model([input1, input2])

    ##### DISCRIMINATE IMAGE
    output = d_model(layer1)

    ##### BUILD MODEL AND COMPILE
    model = Model(inputs = [input1, input2], outputs = output)
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=Adam(lr=0.0002, beta_1=0.5))

    return model

################################################################################
# %% INIT ARRAYS FOR POSTPROCESSING
################################################################################

loss = []

################################################################################
# %% BUILD MODELS
################################################################################

d_model = build_descriminator(IMG_SHP, CAT_SHP)
g_model = build_generator(LAT_SHP, CAT_SHP)
gan_model = build_gan(d_model, g_model)

################################################################################
# %% RUN THROUGH EPOCHS
################################################################################

for epoch in range(EPOCHS):

    print(f'Epoch: {epoch+1}')

    ##### INIT GENERATOR
    real_gen = generator.image_generator('mnist', BATCH_SIZE, CAT_SHP)

    ##### GET NUMBER OF BATCHES
    BATCHES = next(real_gen)

    ################################################################################
    # RUN THROUGH BATCHES
    ################################################################################

    for batch in range(BATCHES):

        ##### GET NEXT BATCH FROM REAL IMAGE GENERATOR
        X_real, y_real_oh, z_real = next(real_gen)

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

        if batch%50==0:
            print(f'd_loss: {d_loss:.3f}, g_loss: {g_loss:.3f}')

        loss.append([d_loss, g_loss])

    print(f'Accuracy: Discriminator {d_acc:.3f} Generator {g_acc:.3f}')

############################################################################
# %% PLOT LOSS CURVES
############################################################################

fig = mp.figure(figsize=(10,8))
mp.loglog(loss)
mp.xlabel('batch')
mp.ylabel('loss')
mp.legend(['d_loss', 'g_loss'])
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

fig = mp.figure(figsize=(CAT_SHP, 1))
input_string = np.arange(CAT_SHP)
input_string = [0,1,2,10,11,12,36,37,38,39]
X_in_p = np.zeros((len(input_string), CAT_SHP))
X_in_p[np.arange(len(input_string)), input_string] = 1
X_in_r = np.random.randn(len(input_string), LAT_SHP-CAT_SHP)
img = g_model.predict([X_in_p, X_in_r])
out = img.reshape(28, len(input_string)*28)
mp.imshow(out, cmap='gray_r')
