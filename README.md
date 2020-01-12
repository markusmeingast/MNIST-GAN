# E/F/MNIST-GAN

This project explores the possibility of using a GAN network to not only generate realistic imitation images for a specific data source, but also define features this imitation should have. Although other methods exist to realize this type of model, a AC-GAN (Auxiliary Classifier GAN) has been implemented.

## GAN

Generative-Adversarial-Networks are a method of semi-supervised learning.
Within a GAN network two neural networks work against each other:

1. a discriminator NN, which tries to determine if a randomly given sample is
real or not
2. a generator NN, which tries to create images that pass the discriminators test

Through each training cycle the discriminator is given real and fake data,
in order to distinguish between them. The generator is then trained on a random
latent input to produce images that the discriminator is tested on.

## Auxiliary Classifier GAN Model

The model utilizes LeakyReLU activation functions and a dropout of 40% per CNN layer. Especially within the discriminator, a batch normalization decreases accuracy convergence significantly.

Generator Model      | Discriminator Model
:-------------------:|:-------------------:
![](img/g_model.png) | ![](img/d_model.png)


Discriminator:

    Model: "d_model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 28, 28, 1)]  0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 14, 14, 64)   1088        input_1[0][0]                    
    __________________________________________________________________________________________________
    leaky_re_lu (LeakyReLU)         (None, 14, 14, 64)   0           conv2d[0][0]                     
    __________________________________________________________________________________________________
    dropout (Dropout)               (None, 14, 14, 64)   0           leaky_re_lu[0][0]                
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 7, 7, 64)     65600       dropout[0][0]                    
    __________________________________________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)       (None, 7, 7, 64)     0           conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 7, 7, 64)     0           leaky_re_lu_1[0][0]              
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 3136)         0           dropout_1[0][0]                  
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 1)            3137        flatten[0][0]                    
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 10)           31370       flatten[0][0]                    
    ==================================================================================================
    Total params: 101,195
    Trainable params: 101,195
    Non-trainable params: 0
    __________________________________________________________________________________________________

Generator:

    Model: "g_model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            [(None, 10)]         0                                            
    __________________________________________________________________________________________________
    input_3 (InputLayer)            [(None, 90)]         0                                            
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 100)          0           input_2[0][0]                    
                                                                     input_3[0][0]                    
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 3136)         316736      concatenate[0][0]                
    __________________________________________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)       (None, 3136)         0           dense_2[0][0]                    
    __________________________________________________________________________________________________
    dropout_2 (Dropout)             (None, 3136)         0           leaky_re_lu_2[0][0]              
    __________________________________________________________________________________________________
    reshape (Reshape)               (None, 7, 7, 64)     0           dropout_2[0][0]                  
    __________________________________________________________________________________________________
    conv2d_transpose (Conv2DTranspo (None, 14, 14, 64)   65600       reshape[0][0]                    
    __________________________________________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)       (None, 14, 14, 64)   0           conv2d_transpose[0][0]           
    __________________________________________________________________________________________________
    dropout_3 (Dropout)             (None, 14, 14, 64)   0           leaky_re_lu_3[0][0]              
    __________________________________________________________________________________________________
    conv2d_transpose_1 (Conv2DTrans (None, 28, 28, 64)   65600       dropout_3[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_4 (LeakyReLU)       (None, 28, 28, 64)   0           conv2d_transpose_1[0][0]         
    __________________________________________________________________________________________________
    dropout_4 (Dropout)             (None, 28, 28, 64)   0           leaky_re_lu_4[0][0]              
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 28, 28, 1)    3137        dropout_4[0][0]                  
    ==================================================================================================
    Total params: 451,073
    Trainable params: 451,073
    Non-trainable params: 0
    __________________________________________________________________________________________________

## MNIST Dataset

One of the most popular and academic entrances into deep learning involves the
MNIST dataset. The data contains balanced 70.000 samples of hand-written digits
that can be used as a basis for CNN classification methods. The data is widely
available and requires no noteworthy cleaning or preprocessing to be worked with.

![loss history](img/mnist-loss.png "loss history for 30 epochs")

real data:

![real data](img/mnist-real.png)

generated data after 5/30/50 epochs:

![5 epochs](img/mnist-5.png "5 epochs of training")
![30 epochs](img/mnist-30.png "30 epochs of training")
![50 epochs](img/mnist-50.png "50 epochs of training")
![100 epochs](img/mnist-100.png "100 epochs of training")

## Fashion-MNIST

Zalando-Research provides a dataset that is highly similar in structure to the MNIST-Digit dataset. Instead of 10 classes containg digits, the F-MNIST dataset contains classes of clothing (e.g. boots, shirts, pants, etc.).

Due to the input shape and classification labels being identical to the MNIST set, the model can be used 1:1 without any modifications apartfrom adjusting for number of total samples.

![loss history](img/fmnist-loss.png "loss history for 30 epochs")

real data:

![real data](img/fmnist-real.png)

generated data after 5/30/100 epochs:

![5 Epochs](img/fmnist-5.png "5 epochs of training")
![30 epochs](img/fmnist-30.png "30 epochs of training")
![50 epochs](img/fmnist-50.png "50 epochs of training")
![100 Epochs](img/fmnist-100.png "100 epochs of training")

## Extended-MNIST

The E-MNIST dataset is the original dataset the MNIST-digits set was originally derived from. There are multiple versions of the dataset available that contain digits, letters in lower and upper case. The full dataset, `emnist-byclass` contains a total of 62 classes (10 digits, 26 lower-case letter and 26 upper-case letters). This dataset is not balanced though with a lowest to highest number of sample ratio of ...

![loss history](img/emnist-loss.png "loss history for 30 epochs")

real data (random order):
![real data](img/emnist-real.png)

generated data after 5/30/100 epochs:
![5 Epochs](img/emnist-5.png "5 epochs of training")
![30 epochs](img/emnist-30.png "30 epochs of training")
![100 Epochs](img/emnist-100.png "100 epochs of training")

## References
