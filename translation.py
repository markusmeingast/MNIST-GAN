"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as mp
import string

################################################################################
# %% CONSTANTS
################################################################################

CLS_SHP = 62
LNV_SHP = 100

################################################################################
# %% BUILD DICTIONARY
################################################################################

char_to_idx = {}
for i in range(10):
    char_to_idx[str(i)] = i
for i in range(26):
    char_to_idx[string.ascii_uppercase[i]] = i+10
for i in range(26):
    char_to_idx[string.ascii_lowercase[i]] = i+36

################################################################################
# %% LOAD MODEL
################################################################################

gen_model = load_model('models/gen_model_emnist.h5')

############################################################################
# %% TEST ON INPUT STRING
############################################################################

fig = mp.figure(figsize=(CLS_SHP, 1))
#input_string = np.arange(CLS_SHP)

input_string = 3*np.ones((CLS_SHP,)).astype(int)

y = np.zeros((len(input_string), CLS_SHP))
y[np.arange(len(input_string)), input_string] = 1
z = np.random.randn(len(input_string), LNV_SHP)
img = gen_model.predict([y, z])
img = img[:, :, :, 0]
img = img.transpose(1,2,0)
out = img.reshape(28, len(input_string)*28, order='F')
out = np.concatenate((out[:,:len(input_string)*28//2], out[:,len(input_string)*28//2:]))
mp.imshow(out, cmap='gray_r')
ax = mp.gca()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

############################################################################
# %% TEST ON INPUT STRING
############################################################################

fig = mp.figure(figsize=(6, 2))
input_string = 'HelloWorld'
idx = []
for char in input_string:
    print(char, char_to_idx[char])
    idx.append(char_to_idx[char])

y = np.zeros((len(input_string), CLS_SHP))
y[np.arange(len(input_string)), idx] = 1
z = np.random.randn(len(input_string), LNV_SHP)
img = gen_model.predict([y, z])
img = img[:, :, :, 0]
img = img.transpose(1,2,0)
out = img.reshape(28, len(input_string)*28, order='F')
out = np.concatenate((out[:,:len(input_string)*28//2], out[:,len(input_string)*28//2:]))
mp.imshow(out, cmap='gray_r')
ax = mp.gca()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
mp.tight_layout()


mp.savefig('asd.png')
