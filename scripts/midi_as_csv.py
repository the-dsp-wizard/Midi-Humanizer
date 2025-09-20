import mido
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def autocorrelate(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

params = np.array([])

files = ["maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_08_Track08_wav.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_10_Track10_wav.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_02_Track02_wav.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_05_R1_2004_02-03_ORIG_MID--AUDIO_05_R1_2004_06_Track06_wav.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_07_R1_2004_01_ORIG_MID--AUDIO_07_R1_2004_02_Track02_wav.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_07_R1_2004_01_ORIG_MID--AUDIO_07_R1_2004_04_Track04_wav.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_07_R1_2004_01_ORIG_MID--AUDIO_07_R1_2004_06_Track06_wav.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_07_R1_2004_01_ORIG_MID--AUDIO_07_R1_2004_12_Track12_wav.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_03_Track03_wav--1.midi",
    "maestro-v1.0.0/2004/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_07_Track07_wav.midi"
]

for i in range(len(files)):
    mid = mido.MidiFile(files[i])

    timing = np.array([])
    velocity = np.array([])

    integrator = 0

    # Print messages from all tracks
    for n, track in enumerate(mid.tracks):
        #print(f'Track {i}: {track.name}')
                
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                integrator += msg.time / mid.ticks_per_beat
                timing = np.append(timing, integrator)
                velocity = np.append(velocity, msg.velocity)

    timing -= timing[0]
    diff = (timing - np.round(timing * 4) * 0.25)

    velocity /= 127

    X = np.append(0, np.diff(np.round(timing * 4) * 0.25) )
    Y = diff

    array = [1, 2, 3, 4, 5]

    with open(f'dataset/ex{i}_in.csv', 'w') as f:
        for item in X:
            f.write(f"{item}, \n")
    
    print(f"Writing to ex{i}_in.csv")
    
    with open(f'dataset/ex{i}_out.csv', 'w') as f:
        for item in Y:
            f.write(f"{item}, \n")
