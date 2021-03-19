from music21 import converter, instrument, note, chord
import pickle
import os
import numpy as np

#Just for clear outputs, you can replace pp.print with print if you want
import pprint
pp = pprint.PrettyPrinter(indent=4)

def get_notes(path):

    """ Get all the notes and chords from the midi files in the ./music directory
        Store in 'notes' array. This is just the letter note (A, B, C, C#) NO FLATS followed by octive.

        Get all the notes' and chords' durations (in terms of quarter notes) from the midi files in the ./music directory
        Store in 'durations' array. This means a "0.25" duration is one quarter note, a "4.0" duration is one whole note.

        We uses str(i.whatever) a lot to convert from music21 pitch objects into letters.
    """

    notes = []
    durations = []
    for root, subfolders, file in os.walk(path):
        file = str(root)+"/"+str(file[0])
        print("Parsing", file)

        midi = converter.parse(file)
        need_parse = midi.flat.notes

        for i in need_parse:
            if isinstance(i, note.Note):
                #What about a 2D array? [[note, duration], [note, duration]]
                #notes.append([str(i.pitch), str(i.duration.quarterLength)])
                notes.append(str(i.pitch))
                durations.append(str(i.duration.quarterLength))
            elif isinstance(i, chord.Chord):
                #2D array:
                #notes.append(['.'.join(str(n) for n in i.pitches), str(i.duration.quarterLength)])
                notes.append('.'.join(str(n) for n in i.pitches))
                durations.append(str(i.duration.quarterLength))

    with open('/Users/liamisaacs/Desktop/github repositories/SelfPiano/notes/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    #Uncomment for clarity
    #print('\n Notes array is \n')
    #pp.pprint(notes)
    #print('\n Durations array is \n')
    #pp.pprint(durations)
    return notes, durations

notes_from_here_please = "/Users/liamisaacs/Desktop/github repositories/SelfPiano/music"
notes, durations = get_notes(notes_from_here_please)

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network

        We will identify a sorted set of pitchnames, and use that to create a note-to-int dictionary.
        We can use this to represent our pitches as a string of numbers.

        After that, we can define that as a network_input and a network_output.
        We then reshape into a LSTM-compatible format.

    """
    sequence_length = 32

    # get all pitch names
    pitchnames = sorted(set([item for item in notes]))
    #pp.pprint(pitchnames)

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    #If you can run tensorflow (not on Mac M1):
    #network_output = np_utils.to_categorical(network_output)

    #Otherwise..
    def one_hot(a, num_classes):
        a = np.array(a)
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    network_output = one_hot(network_output, len(network_output))
    #pp.pprint(network_output)

    return (network_input, network_output)

network_input, network_output = prepare_sequences(notes, 20)
