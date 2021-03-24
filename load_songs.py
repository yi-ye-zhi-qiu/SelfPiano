from music21 import converter, instrument, note, chord
import os

def get_notes(path):

    melody = []
    notes = []
    durations = []
    for filename in os.listdir(path):
      file = path + "/" + filename
      print("Parsing", file)

      midi = converter.parse(file)
      need_parse = midi.flat.notes

      for i in need_parse:
          if isinstance(i, note.Note):
              melody.append(str(i.pitch))
              durations.append(str(i.duration.quarterLength))
          elif isinstance(i, chord.Chord):
              melody.append([str(n) for n in i.pitches])
              durations.append(str(i.duration.quarterLength))

    return melody, durations
