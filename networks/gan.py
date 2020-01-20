"""


# PARAMETERS:
  * CLS_SHP        - number of classes to predict
  * LNV_SHP        - length of latent noise vector

"""




################################################################################
# %% IMPORT PACKAGES
################################################################################

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input, GaussianNoise, Conv2D, Dense, Flatten
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.layers import Conv2DTranspose, concatenate, Dropout
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.utils import plot_model

################################################################################
# %% BUILD AEACGAN CLASS ([A]UTO-[E]NCODER-[A]UXILLARY-[C]LASSIFIER-GAN)
################################################################################

class AEACGAN():

    def __init__(self, IMG_SHP=(28, 28, 1), CLS_SHP=10, LNV_SHP=100, depth=64):

        ##### IMAGE SHAPE
        self.IMG_SHP = (28, 28, 1)

        ##### NUMBER OF CLASSES TO PREDICT
        self.CLS_SHP = CLS_SHP

        ##### LENGTH OF LATENT NOISE VECTOR
        self.LNV_SHP = LNV_SHP

        ##### DEPTH OF CONV LAYERS
        self.depth = depth

        ##### KERNEL INIT
        self.init = RandomNormal(stddev=0.02)

    def  __repr__(self):
        ...

    def build_encoder(self):
        """
        Input:
            * image of IMG_SHP
        Output:
            * vector of CLS_SHP with one-hot-encoded class
            * vector of LNV_SHP predicting latent noise vector
        """

        ##### INPUT IMAGE
        X_in = Input(self.IMG_SHP)

        ##### ADD NOISE TO IMAGE
        net = GaussianNoise(0.05)(X_in)

        ##### CONV2D LAYER WITH STRIDE 2
        net = Conv2D(self.depth, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)(net)
        #net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = Dropout(0.4)(net)

        ##### CONV2D LAYER WITH STRIDE 2
        net = Conv2D(self.depth, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)(net)
        #net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = Dropout(0.4)(net)

        ##### TO DENSE
        net = Flatten()(net)

        ##### DENSE LAYER
        net = Dense(self.depth)(net)
        #net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = Dropout(0.4)(net)

        ##### OUTPUT1: ONE-HOT-VECTOR OF CLASS
        y_out = Dense(self.CLS_SHP, activation='softmax')(net)

        ##### OUTPUT2: LATENT NOISE PREDICTION
        z_out = Dense(self.LNV_SHP, activation='sigmoid')(net)

        ##### BUILD AND RETURN MODEL
        model = Model(inputs = X_in, outputs = [y_out, z_out])
        return model

    def build_generator(self):
        """
        Input:
            * vector of CLS_SHP with one-hot-encoded class
            * vector of LNV_SHP predicting latent noise vector
        Output:
            * image of IMG_SHP
        """

        ##### INPUT1: ONE-HOT-VECTOR OF CLASS
        y_in = Input(self.CLS_SHP)

        ##### INPUT2: LATENT NOISE VECTOR
        z_in = Input(self.LNV_SHP)

        ##### COMBINE
        net = concatenate([y_in, z_in], axis=-1)

        ##### DENSE LAYER
        net = Dense(7*7*self.depth)(net)
        #net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = Dropout(0.4)(net)

        ##### TO CONV2D
        net = Reshape((7, 7, self.depth))(net)

        ##### CONV2D TRANSPOSE WITH STRIDE 2
        net = Conv2DTranspose(self.depth, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)(net)
        #net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = Dropout(0.4)(net)

        ##### CONV2D TRANSPOSE WITH STRIDE 2
        net = Conv2DTranspose(self.depth, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)(net)
        #net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = Dropout(0.4)(net)

        ##### OUTPUT IMAGE
        X_out = Conv2D(1, (7, 7), activation='tanh', padding='same', kernel_initializer=self.init)(net)

        ##### BUILD AND RETURN MODEL
        model = Model(inputs = [y_in, z_in], outputs = X_out)
        return model

    def build_discriminator(self):
        """
        Input:
            * image of IMG_SHP
        Output:
            * binary real/fake classification
            * vector of CLS_SHP with one-hot-encoded class
        """

        ##### INPUT IMAGE
        X_in = Input(self.IMG_SHP)

        ##### ADD NOISE TO IMAGE
        net = GaussianNoise(0.05)(X_in)

        ##### CONV2D LAYER WITH STRIDE 2
        net = Conv2D(self.depth, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)(net)
        #net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = Dropout(0.4)(net)

        ##### CONV2D LAYER WITH STRIDE 2
        net = Conv2D(self.depth, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)(net)
        #net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = Dropout(0.4)(net)

        ##### TO DENSE
        net = Flatten()(net)

        ##### DENSE LAYER
        net = Dense(self.depth)(net)
        #net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = Dropout(0.4)(net)

        ##### OUTPUT1: BINARY CLASSIFICATION
        w_out = Dense(1, activation='sigmoid')(net)

        ##### OUTPUT2: ONE-HOT-VECTOR OF CLASS
        y_out = Dense(self.CLS_SHP, activation='softmax')(net)

        ##### BUILD, COMPILE AND RETURN MODEL
        model = Model(inputs = X_in, outputs = [w_out, y_out])
        model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=Adam(lr=0.0002, beta_1=0.5))
        return model

    def build_autoencoder(self, e_model, g_model):
        """
        Input:
            * image of IMG_SHP
        Output:
            * image of IMG_SHP
        """

        ##### INPUT IMAGE
        X_in = Input(self.IMG_SHP)

        ##### INTERMEDIATE OUTPUT (ONE-HOT-VECTOR AND LATENT NOISE)
        y, z = e_model(X_in)

        ##### GENERATOR OUTPUT
        X_out = g_model([y, z])

        ##### BUILD, COMPILE AND RETURN MODEL
        model = Model(inputs = X_in, outputs = X_out)
        model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return model

    def build_acgan(self, g_model, d_model):
        """
        Input:
            * vector of CLS_SHP with one-hot-encoded class
            * vector of LNV_SHP latent noise vector
        Output:
            * binary real/fake classification
            * vector of CLS_SHP with one-hot-encoded class

        No discriminator training!
        """

        ##### FREEZE DISCRIMINATOR
        d_model.trainable = False

        ##### INPUT1: ONE-HOT-VECTOR OF CLASS
        y_in = Input(self.CLS_SHP)

        ##### INPUT2: LATENT NOISE VECTOR
        z_in = Input(self.LNV_SHP)

        ##### GENERATE IMAGE
        X = g_model([y_in, z_in])

        ##### DISCRIMINATE
        w_out, y_out = d_model(X)

        ##### BUILD, COMPILE AND RETURN MODEL
        model = Model(inputs = [y_in, z_in], outputs = [w_out, y_out])
        model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=Adam(lr=0.0002, beta_1=0.5))
        return model
