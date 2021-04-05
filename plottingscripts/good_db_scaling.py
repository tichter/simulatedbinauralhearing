import numpy as np
import scipy
import scipy.signal

fs = 48000
duration = 0.085
n_samples = int(duration*fs)

f_min = 4000
f_max = 8000
butter_order = 6
cutoff = [f_min,f_max]
sos = scipy.signal.butter(butter_order,cutoff,'bandpass',output='sos',fs=fs)

snd_t = np.random.normal(size = n_samples)
snd_t = scipy.signal.sosfilt(sos,snd_t)
