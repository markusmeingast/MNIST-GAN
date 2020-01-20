"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

from gan import AEACGAN
from tensorflow.keras.models import load_model
import generator
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as mp

################################################################################
# %% CONSTANTS
################################################################################

DATASET = 'mnist'
IMG_SHP = (28, 28, 1)
CLS_SHP = 10
LNV_SHP = 100
EPOCHS = 100
BATCH_SIZE = 128

################################################################################
# %% BUILD MODELS
################################################################################

##### INIT GAN
gan = AEACGAN(IMG_SHP, CLS_SHP, LNV_SHP)

##### BUILD ENCODER AND GENERATOR
e_model = gan.build_encoder()
g_model = gan.build_generator()
d_model = gan.build_discriminator()

##### BUILD AUTOENCODER
ae_model = gan.build_autoencoder(e_model, g_model)

##### BUILD GAN
acgan_model = gan.build_acgan(g_model, d_model)

################################################################################
# %% INIT HISTORY
################################################################################

loss = []

################################################################################
# %% LOOP THROUGH EPOCHS
################################################################################

for epoch in range(EPOCHS):

    print(f'Epoch: {epoch+1}')

    ##### INIT GENERATOR
    real_gen = generator.image_generator(DATASET, BATCH_SIZE, CLS_SHP)

    ##### GET NUMBER OF BATCHES
    BATCHES = next(real_gen)

    ############################################################################
    # RUN THROUGH BATCHES
    ############################################################################

    for batch in range(BATCHES):

        ############################################################################
        # DISCRIMINATOR TRAINING
        ############################################################################

        ##### GET NEXT BATCH FROM REAL IMAGE GENERATOR
        X_real, y_real, w_real = next(real_gen)

        ##### GENERATE RANDOM DIGITS
        idx = np.random.randint(0, high=CLS_SHP, size=BATCH_SIZE, dtype='int')

        ##### ONE-HOT-ENCODE NUMBERS
        y_fake = np.zeros((BATCH_SIZE, CLS_SHP), dtype=float)
        y_fake[np.arange(BATCH_SIZE), idx] = 1

        ##### GENERATE LATENT NOISE VECTOR
        z_fake = np.random.randn(BATCH_SIZE, LNV_SHP)

        ##### PREDICT IMAGE FROM RANDOM INPUT
        X_fake = g_model.predict([y_fake, z_fake])

        ##### SET BINARY CLASS TO FAKE
        w_fake = 0.1*np.ones((BATCH_SIZE, 1), dtype=int)

        ##### CONCAT REAL AND FAKE DATA
        X_batch = np.concatenate((X_real, X_fake), axis=0)
        y_batch = np.concatenate((y_real, y_fake), axis=0)
        w_batch = np.concatenate((w_real, w_fake), axis=0)

        ##### TRAIN!
        d1_loss, d2_loss, d3_loss = d_model.train_on_batch(X_batch, [w_batch, y_batch])

        ############################################################################
        # GENERATOR TRAINING
        ############################################################################

        g1_loss, g2_loss, g3_loss = acgan_model.train_on_batch([y_fake, z_fake], [w_real, y_fake])

        ############################################################################
        # ENCODE TRAINING
        ############################################################################

        e1_loss = ae_model.train_on_batch(X_real, X_real)

        loss.append([d1_loss, d2_loss, d3_loss, g1_loss, g2_loss, g3_loss, e1_loss])

        if batch%50 == 0:
            print(loss[-1])


############################################################################
# %% PLOT AUTOENCODER RESULTS
############################################################################

idx = np.random.randint(low=0, high=BATCH_SIZE)
mp.subplot(1,2,1)
mp.imshow(X_real[idx, :, :, 0], cmap='gray_r')
mp.axis('off')
y_pred, z_pred = e_model.predict(X_real)
X_pred = g_model.predict([y_pred, z_pred])
mp.subplot(1,2,2)
mp.imshow(X_pred[idx, :, :, 0], cmap='gray_r')
mp.axis('off')

############################################################################
# %% PLOT LOSS CURVES
############################################################################

fig = mp.figure(figsize=(10,8))
mp.semilogy(np.array(loss)[:, [1, 2, 4, 5, 6]])
mp.xlabel('batch')
mp.ylabel('loss')
mp.legend(['d_bin_loss', 'd_cat_loss', 'g_bin_loss',  'g_cat_loss', 'e_msq_loss'])
mp.show()

############################################################################
# %% TEST ON INPUT STRING
############################################################################

fig = mp.figure(figsize=(CLS_SHP, 1))
#input_string = np.arange(CLS_SHP)

input_string = 3*np.ones((CLS_SHP,)).astype(int)

y = np.zeros((len(input_string), CLS_SHP))
y[np.arange(len(input_string)), input_string] = 1
z = np.random.randn(len(input_string), LNV_SHP)
img = g_model.predict([y, z])
img = img[:, :, :, 0]
img = img.transpose(1,2,0)
out = img.reshape(28, len(input_string)*28, order='F')
out = np.concatenate((out[:,:len(input_string)*28//2], out[:,len(input_string)*28//2:]))
mp.imshow(out, cmap='gray_r')
ax = mp.gca()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

############################################################################
# %% SAVE MODELS
############################################################################

g_model.save('gen_model.h5')
d_model.save('dis_model.h5')
acgan_model.save('gan_model.h5')
e_model.save('enc_model.h5')
ae_model.save('ae_model.h5')
