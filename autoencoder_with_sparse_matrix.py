from load_songs import *
import sys

# === load data

path = "/Users/liamisaacs/Desktop/github repositories/kitchensoundscapes/music"
melody, durations = get_notes(path)


# === correct random fractions that music21 causes

durations = [x.replace('1/3', '0.25') for x in durations]
durations = [x.replace('2/3', '0.75') for x in durations]
durations = [x.replace('4/3', '1') for x in durations]
durations = [x.replace('5/3', '1.25') for x in durations]
durations = [x.replace('7/3', '2.25') for x in durations]
durations = [x.replace('8/3', '2.5') for x in durations]
durations = [x.replace('10/3', '3.25') for x in durations]
durations = [x.replace('13/3', '4.25') for x in durations]
durations = [x.replace('23/3', '7') for x in durations]

# === create string of unique notes

def note_string(melody):
    #define notes as a string list of each note in melody
    notes = []
    for i, j in enumerate(melody):
      if isinstance(j, list):
        for k in j:
          notes.append(k)
      else:
        notes.append(j)

    def list_to_dict(x):
        unique = sorted(set([item for item in x]))
        print(len(unique),' unique notes put into dictionary')
        Dict = dict((i, j) for j, i in enumerate(x))
        return Dict

    notes_d = list_to_dict(notes)
    return notes_d

notes_d = note_string(melody)

#initialize sparse matrix
l = []

for i in notes_d.keys():
  l.append([i])
print('Each note has been given a row in matrix l')

# === sparse matrix

def sparse_matrix(l, melody, durations, notes_d):
    """
    Returns a sparse matrix from a melody/duration input.

    Melody can be ['A#3', 'C4'], durations can be ['0.5', '0.25']

    Then sparse matrix will be:

    [['A#3', 1 1 0],
     ['C4', 0 0 1]]

    The rows are notes (there are 88 unique notes), and the columns are taken every instance in time (it's hard to explain).
    Basically, 0 means the note is not being played and 1 means it is.

    We remove the A#3 and C4 but preserve the order of the notes in "order_"

    So order_ is ['A#3', 'C4'] and l is [[1, 1, 0], [0, 0, 1]]

    We do this because chords, rests and overlap all need to be captured.

    A chord is more than one note occurring simultaneously. In the data it looks like ['E3', ['C3, E4']] in "melody" variable.

    To account for this, we use our "__freeze_time" boolean. If it's a chord, we just "freeze" (we do not increment) the t variable,
    so it's like time is not passing. In that instance, we add all of the notes, one by one, and to the computer it just seems
    like we played them all simultaneously.
    """

    print('Melody length is ', len(melody))
    stops, t = {}, 0

    for index, n in enumerate(melody):

        if isinstance(n, list):
            __freeze_time = True
        else:
            __freeze_time = False

        d = dict((k, v) for k, v in stops.items() if v < t)

        if __freeze_time == True:
            for note in n:
                for i, j in enumerate(l):
                    dur = durations[index]
                    m = int(float(dur) / 0.25)

                    if note != j[0] and j[0] not in n:
                        for x in range(m):
                            l[i].append(0)
                    else:

                        for x in range(m):
                            l[i].append(1)
                        stops[note] = m

        elif __freeze_time == False:
            for i, j in enumerate(l):
                dur = durations[index]
                m = int(float(dur) / 0.25)

                if n != j[0]:
                    for x in range(m):
                        l[i].append(0)
                else:

                    for x in range(m):
                        l[i].append(1)
                    stops[n] = m
        t +=1


    #prep for sparse matrix by cutting off first note
    for i in range(len(l)):
        l[i][0] = notes_d[l[i][0]]

    #might need to preserve note order for later..
    # order_ = []
    # for i in range(len(l)):
    #     order_.append(l[i][0])
    print(l)
    return l

l = sparse_matrix(l, melody, durations, notes_d)

from scipy.sparse import csr_matrix, csc_matrix

#print(np.array(l))
sparse = csc_matrix(l)
print('Sparse matrix of form m x n where m=note and n=duration stored in l')
print("Memory utilised (bytes): ", sys.getsizeof(sparse))
print('Order of notes preserved in "order_" variable')

from keras.utils import np_utils
from keras.layers import Input, Dense, Flatten, Reshape
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import RMSprop

import tensorflow as tf

""" Trying to run an autoencoder on sparse matrix... """

""" Hyperparameters """
#Notes_ = Notes_[1:2000]
Notes_ = sparse
sequence_length = 1

latent_dim = 2
input_sample = Notes_.shape[0]
print('input_sample (datasize) is the # of unique notes (# of rows in sparse matrix), ', input_sample)
input_notes = Notes_.shape[1]
input_dim = input_notes * sequence_length
print('input_dim (# of cols in matrix) are', input_dim)

epochs = 10

""" Model (cross-entropy? autoencoder) """

##Encoder
#EncInput = Input(shape = (number_of_samples))
EncInput = Input(shape= (input_dim))
Enc = Dense(2, activation = 'tanh')(EncInput)
print(Enc)
encode = Model(EncInput, Enc)

##Decoder
DecInput = Input(shape= (latent_dim))
Dec = Dense(input_dim, activation = 'sigmoid')(DecInput)
print(Dec)
decode = Model(DecInput, Dec)

##Autoencoder
autoencoder = Model(EncInput, decode(Enc))

#Try different learning rates, batch sizes, dropout layer (?), add another layer (if more data)?
autoencoder.compile(loss = 'binary_crossentropy', optimizer=RMSprop(learning_rate=0.00013))

# ==== autoencoder

#Activation functions

def sigmoid(x):
    if not isinstance(x, np.ndarray):
        print("Wrong parameter of sigmoid function")
        return False
    sigm = 1.0 / (1 + np.exp(-x))
    return sigm

# def tanh(x):
#     return np.tanh(x)

def ReLU(x):
    # Recifier Nonlinearities
    if not isinstance(x, np.ndarray):
        print("Wrong parameter of ReLU funciton")
        return False
    relu = x.copy()  # copy(), Too Important!
    relu[relu < 0] = 0
    return relu

# === still a work in progress
# === the output is not currently 0s and 1s
