# -*- coding: utf-8 -*-
"""
"""

import sys, os, math
import numpy as np
import padasip as pa
import pylab as pl
import scipy.signal as sp
import soundfile as sf
import itertools
from timeit import default_timer as timer

def hhmmss(t):
    s = t;
    m = int(s // 60); s = s - m*60
    h = int(m // 60); m = m - h*60
    return '{:d}h{:02d}m{:06.3f}s'.format(h,m,s)
    
# See https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
def ranges(i):
    for a, b in itertools.groupby(enumerate(i), lambda xy: xy[1] - xy[0]):
        b = list(b)
        yield b[0][1], b[-1][1]
        
def power(samples):
    p = np.cumsum(np.square(samples))
    return p / len(samples)

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

def goertzel(samples, sample_rate, *freqs):
    """
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.

    `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.

    The function returns 2 arrays, one containing the actual frequencies calculated,
    the second the coefficients `(real part, imag part, power)` for each of those frequencies.
    For simple spectral analysis, the power is usually enough.

    Example of usage :
        
        freqs, results = goertzel(some_samples, 44100, (400, 500), (1000, 1100))
    """
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.floor(f_end / f_step + 1))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    freqs = []
    results = []
    for k in sorted(bins):

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation for each sample using iteration
        """
        d1, d2 = 0.0, 0.0
        for n in range(0, window_size):
            y  = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y
        """

        # Doing the calculation on the whole buffer using lfilter (4x faster)
        a = [1.0, -w_real, 1.0]
        b = [1.0, 0.0,     0.0]
        d = sp.lfilter(b, a, samples)
        d1, d2 = d[-1], d[-2]
        
        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2**2 + d1**2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
        
    return freqs, results, f_step

def filter_signal(source, samplerate):
    ORDER = 5
    # filtering
    x = pa.input_from_history(source, ORDER)[:-1]
    source = source[ORDER:]
    f = pa.filters.FilterRLS(mu=0.99, n=ORDER)
    y, e, w = f.run(source, x)

    pl.subplot(3,1,1)
    pl.plot(source)
    pl.subplot(3,1,2)
    pl.plot(y)
    pl.subplot(3,1,3)
    pl.plot(e)

    return y

wavefile = sys.argv[1]
sourcepath = os.path.split(wavefile)[0]   # Extract the path to the file
sourcename = os.path.splitext(os.path.split(wavefile)[1])[0]      # Extract the name of the file
outputpath = os.path.join(sourcepath, sourcename)
os.makedirs(outputpath, exist_ok=True)
with sf.SoundFile(wavefile, 'r') as f:

    TARGET_FREQUENCY = 1066.0   # Hz
    REF_FREQUENCY = 1400.0      # Hz
    BLOCK_DURATION = 0.1        # seconds
    BLOCK_OVERLAP = 0.0         # seconds
    
    print("Processing:", wavefile)
    print("Output to:", outputpath)
    print("Sample rate:", f.samplerate)

    BLOCK_DURATION_N = int(BLOCK_DURATION * f.samplerate)   # samples
    BLOCK_OVERLAP_N = int(BLOCK_OVERLAP * f.samplerate)   # samples

    # Run Goertzel filters across target and reference frequency bands
    freqs = []
    f_step = 0
    num_samples = 0
    signal_power = []
    meteor_power = []
    noise_power = []
    start = timer()
    for block in f.blocks(blocksize=BLOCK_DURATION_N, overlap=BLOCK_OVERLAP_N) :
        if len(block) != BLOCK_DURATION_N: break
        num_samples = num_samples + BLOCK_DURATION_N
        signal_power.append(power(block))
        freqs, g_results, f_step = goertzel(block, f.samplerate, (TARGET_FREQUENCY, TARGET_FREQUENCY), (REF_FREQUENCY, REF_FREQUENCY))
        meteor_power.append(g_results[0][2])
        noise_power.append(g_results[1][2])

    finish = timer()
    print('Time taken: {} for {} samples ({:.3g} MS/s)'.format(hhmmss(finish-start), num_samples, num_samples/(1E6*(finish-start))))
    print("Frequencies:", freqs)
    print("Frequency resolution:", f_step)
    
    # Finalise measures and plot power sequences
    t = [i*(BLOCK_DURATION-BLOCK_OVERLAP) for i in range(0, len(meteor_power))]
    power_ratio = np.divide(meteor_power, noise_power)
    meteor_std = np.std(meteor_power)
    noise_std = np.std(noise_power)
    
    """
    pl.subplot(1,4,1)
    pl.plot(t, signal_power)
    pl.subplot(1,4,2)
    pl.plot(t, meteor_power)
    pl.ylim([0,1000])
    pl.subplot(1,4,3)
    pl.plot(t, noise_power)
    pl.ylim([0,1000])    
    pl.subplot(1,4,4)
    pl.plot(t, power_ratio)
    pl.ylim([0,1000])    
    """

    # Find high energy events from power ratio
    '''
    DETECTION_THRESHOLD = 1000.0
    detections = [(t[i],v) for (i,v) in enumerate(power_ratio) if v > DETECTION_THRESHOLD]
    print("Events detected from power ratio:")
    for (x,y) in detections:
        print('{}: {}'.format(hhmmss(x),y))
    '''
    
    # Find high energy events from deviation
    DETECTION_THRESHOLD = 1.0
    detections = [(i, t[i], j/meteor_std) for (i,j) in enumerate(meteor_power) if (j/meteor_std) > DETECTION_THRESHOLD]
    # print("Events detected from statistical deviation:")
    # for (i,j,k) in detections:
    #    print('{}: {:>5.1f}'.format(hhmmss(j),k))  # start_time, significance
    
    # RLE detection samples
    detections_i = list(ranges([i for (i,j,k) in detections]))
    detections_final = [(t[i],t[j]-t[i]+BLOCK_DURATION,np.max(meteor_power[i:j+1])/meteor_std,i,j) for (i,j) in detections_i]
    with open(os.path.join(outputpath, sourcename + ".txt"), "w") as tf:
        tf.write('start_time, duration, significance\n')
        for (i,j,k,l,m) in detections_final:
            # print('  {}: {:>3.1f} {:>5.1f}'.format(hhmmss(i),j,k))  # start_time, duration, significance
            tf.write('  {}: {:>3.1f} {:>5.1f}\n'.format(hhmmss(i),j,k))  # start_time, duration, significance
        
    # Plot detections
    DETECTION_THRESHOLD = 1.0
    DETECTION_CONTEXT = 1.0     # seconds
    DETECTION_CONTEXT_N = int(DETECTION_CONTEXT / BLOCK_DURATION)   # blocks
    print("Events detected from statistical deviation:")
    for d in detections_final:
        if d[2] > DETECTION_THRESHOLD:
            print('  {}: {:>3.1f} {:>5.1f}'.format(hhmmss(d[0]),d[1],d[2]))  # start_time, duration, significance
            '''
            start_i = d[3]-DETECTION_CONTEXT_N 
            stop_i = d[4]+1+DETECTION_CONTEXT_N
            pl.plot(t[start_i:stop_i],meteor_power[start_i:stop_i])
            pl.show()
            '''
    
    # Plot Spectrogram for each detection, plot to screen and to file
    DETECTION_THRESHOLD = 2.0
    SPECTROGRAM_NFFT = 1024
    SPECTROGRAM_STEP = 128
    print("Events detected from statistical deviation:")
    for d in detections_final:
        if d[2] > DETECTION_THRESHOLD:
            print('  {}: {:>3.1f} {:>5.1f}'.format(hhmmss(d[0]),d[1],d[2]))  # start_time, duration, significance
            start_i = d[3]-DETECTION_CONTEXT_N 
            stop_i = d[4]+1+DETECTION_CONTEXT_N
            wave = sf.read(wavefile, start=start_i*BLOCK_DURATION_N, stop=stop_i*BLOCK_DURATION_N)[0]
            wave = fixup_wave(wave, f.samplerate)
            # Apply noise cancellation to signal
            wave = filter_signal(wave, f.samplerate)
            # Generate plot of wave, spectrogram, and power and save as image file
            pl.subplot(3,1,1)
            pl.plot(wave)
            pl.title('start={}, dur={:.1f}, sig={:.1f}'.format(hhmmss(d[0]),d[1],d[2]))  # start_time, duration, significance
            pl.xticks([])
            pl.subplot(3,1,2)
            pl.specgram(wave,SPECTROGRAM_NFFT,f.samplerate,noverlap=SPECTROGRAM_NFFT-SPECTROGRAM_STEP,cmap=pl.get_cmap('inferno'))
            pl.xticks([])
            pl.ylim([750,1750])    
            pl.subplot(3,1,3)
            pl.plot(t[start_i:stop_i],meteor_power[start_i:stop_i])
            pl.savefig(os.path.join(outputpath, sourcename + " - " + hhmmss(d[0]) + ".png"))
            pl.show()
            #pl.close()
            # Generate wav file for detection
            # Renormalise for WAV format (+/-1. limits)
            peak = max(np.max(wave), -np.min(wave))
            wave = np.divide(wave, 1.01 * peak)
            sf.write(os.path.join(outputpath, sourcename + " - " + hhmmss(d[0]) + ".wav"), wave, f.samplerate)
            
            