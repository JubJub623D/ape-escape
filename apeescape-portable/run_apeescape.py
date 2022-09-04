import tensorflow.keras as keras
from keras import models
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences 
import sys
import fileinput
import numpy as np

thirty_two_kb = 2 * 1024 * 32

model = keras.models.load_model('densemodel.h5')
file_list = fileinput.input(mode='rb')
ggrs = [ggrfile.hex(' ')[408:] for ggrfile in file_list]
print(ggrs)
'''tkn = Tokenizer(10000)
sequences = pad_sequences(tkn.texts_to_sequences(ggrs), thirty_two_kb)

output = model(sequences)
'''
