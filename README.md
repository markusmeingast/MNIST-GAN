# E/F/MNIST - GAN

## Nomenclature

X     images
y     one-hot-vector classification
z     noise vector
w     real/fake classification

## Assumptions

* Real samples get label 1, fake get label 0
* ~~One-sided feature smoothing (real = 0.9, fake = 0.0)~~
* Latent noise vector in interval [0,1]
* Images normalized to [-1,1] (tanh)

* Generator, discriminator and encoder to use LeakyReLU and Dropout
