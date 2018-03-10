# -*- coding: utf-8 -*-
"""
"""

import os
import numpy as np
import pylab as pl
import soundfile as sf

def _fixup_wave(wave, samplerate):
    THRESHOLD = 1
    correction = list(map(
        lambda x: -2.0 if ((x[1]-x[0]) > THRESHOLD) else 2.0 if ((x[1]-x[0]) < -THRESHOLD) else 0,
        zip([wave[0]] + wave, wave)
        ))
    wave = np.add(wave, np.cumsum(correction))
    peak = max(np.max(wave), -np.min(wave))
    print(peak)
    # wave = np.divide(wave, 1.01 * peak)
    return wave

def fixup_wave(wave, samplerate):
    
    # Step 1: Remove wrap by step size
    WRAP = 2
    correction = [0] * len(wave)
    for (i,s) in enumerate(wave[0:-1]):
        d_upper = abs(wave[i+1] + WRAP - wave[i])
        d_mid = abs(wave[i+1] - wave[i])
        d_lower = abs(wave[i+1] - WRAP - wave[i])
        correction[i+1] = WRAP if (d_upper < d_mid) else -WRAP if (d_lower < d_mid) else 0
        
    wave = np.add(wave, np.cumsum(correction))
    
    # Step 2: Clean using sample prediction
    DELTA = 1.5
    correction = [0] * len(wave)
    delta = [0] * len(wave)
    for i in range(2, len(wave)):
        p = wave[i-1] + (wave[i-1] - wave[i-2])
        delta[i] = wave[i] - p
        if (delta[i] > DELTA):
            wave = [(y-WRAP if x>=i else y) for (x,y) in enumerate(wave)]
        elif (delta[i] < -DELTA):
            wave = [(y+WRAP if x>=i else y) for (x,y) in enumerate(wave)]

    return wave

wavefile = '..\\recordings\\Taurids 2017_11_11 22-17-34\\Taurids 2017_11_11 22-17-34 - 3h25m02.500s.wav'
sourcename = os.path.splitext(os.path.split(wavefile)[1])[0]      # Extract the name of the file

wave, samplerate = sf.read(wavefile)
wave = wave.tolist()
#wave = wave[int(samplerate * 0.7800):int(samplerate * 0.7840)]
result = fixup_wave(wave, samplerate)

t = [i/samplerate for i in range(0, len(wave))]

pl.subplot(3,1,1)
pl.plot(t, wave)
pl.title(sourcename)
pl.subplot(3,1,2)
pl.plot(t, wave, t, result)
# pl.xlim([1.04, 1.05])

peak = max(np.max(result), -np.min(result))
result = np.divide(result, 1.01 * peak)

pl.subplot(3,1,3)
pl.plot(t, result)

sf.write(wavefile + " - fixed.wav", result, samplerate, subtype='DOUBLE')
