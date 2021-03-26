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
  """
  Converts list to dict or list of lists to dict
  """
  try: #if JUST a list as input
    unique = sorted(set([item for item in x]))
    print(len(unique),' unique notes put into dictionary')
    Dict = {}
    for i in range(0, len(unique)):
      Dict[unique[i]] = i
    #Dict = dict((i, j) for j, i in enumerate(x))
    return Dict

  except: #if LIST input contains LIST (if input is a list of lists...)
    new_x = []
    for i, j in enumerate(x):
      if isinstance(j, list):
        for k in j:
          new_x.append(k)
      else:
        new_x.append(j)

    return list_to_dict(new_x)

def construct_melody(x, dictionary, sequence_length):
  """
  Constructs "windows" for melody,
  segments of notes (of len sequence_length) to use as input,
  since we encode each note using a dictionary.

  If your collab crashes a lot, this is probably due to a bad dictionary.
  The autoencoder should be able to handle longer sequences.
  """
  print('Constructing segments of notes as melodies to use as input, since we encode each note using a dictionary (note_to_int)')
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
    #duration_to_int = list_to_dict(durations)
    print('Note dictionary, \n',note_to_int)
    #print('Duration dictionary, \n',duration_to_int)

    Notes_ = construct_melody(notes, note_to_int, sequence_length)
    #durations_ = construct_melody(durations, duration_to_int, sequence_length)

    Notes_ = np_utils.to_categorical(Notes_).transpose(0,2,1)
    Notes_ = np.array(Notes_, np.float)
    print('notes->Notes_; one-hot encoded; turned into a numpy array')

    return note_to_int, Notes_#, durations_

sequence_length = 60 #how long your output is
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
    print(Enc)
    encode = Model(EncInput, Enc)

    ##Decoder
    DecInput = Input(shape= (latent_dim))
    Dec = Dense(input_dim, activation = 'sigmoid')(DecInput)
    print(Dec)
    decode = Model(DecInput, Dec)

    ##Autoencoder
    autoencoder = Model(EncInput, decode(Enc))
    autoencoder.compile(loss = 'binary_crossentropy', optimizer=RMSprop(learning_rate=0.000013))

    return autoencoder, decode

autoencoder, Decode_r = get_autoencoder(input_sample, input_notes, input_dim, latent_dim)
fit_on = Notes_.reshape(input_sample, input_dim)
autoencoder.fit(fit_on, fit_on, epochs=num_epochs)

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
