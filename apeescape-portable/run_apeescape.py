import os
import tensorflow.keras as keras
from keras import models
from keras.utils import pad_sequences 
import sys
import numpy as np

thirty_two_kb = 1024 * 32

model = keras.models.load_model('ape_escape.h5')
ggrs = []
for filename in sys.argv[1:]:
  with open(filename, 'rb') as ggrfile:
    hexed_file = ggrfile.read().hex(' ').split(' ')
    ggrs.append(hexed_file[136:])
    
sequences = sequences = [[int(n, base=16) for n in sequence] for sequence in ggrs]
output = model(pad_sequences(sequences, maxlen = thirty_two_kb))
print(output)

