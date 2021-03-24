from keras.utils import np_utils
from keras.layers import Input, Dense, Flatten, Reshape
from keras import layers
from keras.models import Model
from keras.optimizers import RMSprop

import tensorflow as tf

from music21 import converter, instrument, note, chord, stream
import pickle
import os
import numpy as np

from . import get_songs

songs, durations = get_songs('..music/training_songs')

# === Helper functions
def list_to_dict(x):
    unique = sorted(set([item for item in x]))
    Dict = dict((i, j) for j, i in enumerate(x))
    return Dict

def construct_melody(x, dictionary, sequence_length):
  result = []
  for i in range(0, len(x) - sequence_length):
    sequence_window = x[i:i + sequence_length]
    result.append([dictionary[char] for char in sequence_window])
  return result
# ===

# === One-hot encode notes

def get_TrainingData(notes, durations, sequence_length):

    print('Desired sequence length is', sequence_length)

    note_to_int = list_to_dict(notes)
    duration_to_int = list_to_dict(durations)
    print('Note dictionary, \n',note_to_int)
    print('Duration dictionary, \n',duration_to_int)

    Notes_ = construct_melody(notes, note_to_int, sequence_length)
    durations_ = construct_melody(durations, duration_to_int, sequence_length)

    Notes_ = np_utils.to_categorical(Notes_).transpose(0,2,1)
    Notes_ = np.array(Notes_, np.float)

    return note_to_int, Notes_, durations_

int_to_note, Notes_, durations_ = get_TrainingData(notes, durations, sequence_length)

""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)


""" Hyperparameters """
sequence_length = 60
latent_dim = 2
input_sample = Notes_.shape[0]
input_notes = Notes_.shape[1]
input_dim = input_notes * sequence_length
num_epochs = 10

def get_autoencoder(input_sample, input_notes, input_dim, latent_dim):

    ##Encoder
    EncInput = Input(shape= (input_dim))
    Enc = Dense(latent_dim, activation = 'tanh')(EncInput)
    encode = Model(EncInput, Enc)

    ##Decoder
    DecInput = Input(shape= (latent_dim))
    Dec = Dense(input_dim, activation = 'sigmoid')(DecInput)
    decode = Model(DecInput, Dec)

    ##Autoencoder
    autoencoder = Model(EncInput, decode(Enc))
    autoencoder.compile(loss = 'binary_crossentropy', optimizer=RMSprop(learning_rate=0.00013))

    return autoencoder, decode

autoencoder = get_autoencoder(input_sample, input_notes, input_dim, latent_dim)
autoencoder.fit(Notes_.reshape(input_sample, input_dim), Notes_.reshape(input_sample, input_dim), epochs=num_epochs)

def get_key(val, dictionary):
    for key, value in dictionary.items():
         if val == value:
             return key

    return "key doesn't exist"


def decode(int_to_note, decoder):

    ##Computer's melody
    ComputersMelody = decode(np.random.normal(size=(1, latent_dim))).numpy().reshape(input_notes, sequence_length).argmax(0)
    MelodyNotes = [get_key(c, IntToNote) for c in ComputersMelody]


    return MelodyNotes

MelodyNotes = decode(int_to_note, decode)

def give_song(MelodyNotes):
    s_obj = stream.Stream()
    s_obj.append(instrument.Piano())

    count = 0
    for x in MelodyNotes:
      if x == "key doesn't exist":
        count += 1
        continue;
      else:
        if '.' in x:
          s_obj.append(chord.Chord(x.replace('.', ' ')))
        else:
          s_obj.append(note.Note(x))

    print('erased', count, 'notes')
    s_obj.write('midi', fp='/content/drive/My Drive/notebooks/music'+'/generated.mid')
    print('generated')
    break;
