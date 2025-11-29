import mido
import numpy as np
import sys
from scipy.io import wavfile

def envelope(fs, release):
    return np.append(np.linspace(0, 1, int(0.01 * fs) ), np.linspace(1, 0, int(release * fs - 0.01 * fs) ) )

def wave(fs, freq, length):
    return np.arctan(np.sin(np.linspace(0, length / fs, length) * freq * 2 * np.pi) ) * 0.1

def synth(fs, release_time, file):
    mid = file

    timing = np.array([])
    freq = np.array([])

    integrator = 0
    tempo = 0

    for n, track in enumerate(mid.tracks):
        for msg in track:
            integrator += msg.time / mid.ticks_per_beat * tempo * 1e-6

            if msg.type == 'set_tempo':
                tempo = msg.tempo
            if msg.type == 'note_on' and msg.velocity > 0:
                integrator += msg.time / mid.ticks_per_beat * tempo * 1e-6
                timing = np.append(timing, integrator)
                freq = np.append(freq, 440 * np.power(2, (msg.note - 60)  / 12) )

    sound = np.zeros(int(fs * timing[len(timing) - 1] + int(fs * release_time) ) )
 
    for i in range(len(timing) ):
        start = int(fs * timing[i])
        end = start + int(release_time * fs)
        length =  int(release_time * fs)
        sound[start:end] += wave(fs, freq[i], length) * envelope(fs, release_time)

    return sound