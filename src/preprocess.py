"""
Preprocessing module for MIDI data. MIDI files must be transformed to an
appropriate format for consumption by a neural network for training.
"""

import numpy as np

from music21 import chord, converter, instrument, note
from sklearn.preprocessing import OneHotEncoder


def parse_midi(file: str) -> list:
    """
    Parse a midi file given the relative path to the file

    Args:
      - file: the relative path to the file
    Returns: A list of notes from the midi file
    """
    notes: list = []
    midi = converter.parse(file)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)

    # check to see if MIDI file has multiple parts
    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if type(element) == note.Note:
            notes.append(str(element.pitch))
        elif type(element) == chord.Chord:
            notes.append(str(".".join(str(n) for n in element.normalOrder)))
    return notes


def one_hot_notes(params: list) -> OneHotEncoder:
    """
    Fit a one hot encoder using the list of all of the notes

    Args:
      - params: A list of all of the notes in the positions they correspond to
    Returns: A fitted one hot encoder
    """
    encoder = OneHotEncoder(categories=params)

    # TODO should we fit encoder to `params`?
    return encoder


def one_hot_transform(encoder: OneHotEncoder, notes: list) -> np.array:
    """
    Given a trained one hot encoder and a list of notes to transform,
    one-hot-encode the notes.

    Args:
      - encoder: A trained/fitted one hot encoder
      - notes: a list of notes to transform
    Returns: A transformed array
    """
    return encoder.transform(notes)
