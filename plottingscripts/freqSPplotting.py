import sys
sys.path.insert(0,"D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\utils")

from freqSoundProcessor import FreqSoundProcessor
from hrtfloader import HRTFLoader
import numpy as np
import matplotlib.pyplot as plt

add_ele = False
doublepolar = False
dbl = 120
hrtf_n = 1

edhrtfloader = HRTFLoader(ED=True,add_ele=add_ele,doublepolar=doublepolar)

fsp = FreqSoundProcessor(db=dbl,add_ele=add_ele,doublepolar=doublepolar,edhrtfloader=edhrtfloader)

fsp.generateNoise()
noise_t = fsp.snd_t
noise_f = fsp.snd_f.copy()
freqs = fsp.freqs

print(len(noise_f))
print(len(freqs))

fsp.bandpassfilter(f_min=1000,f_max=4000)
band_t = fsp.snd_t
band_f = fsp.snd_f

fsp.setdBSPL(db=dbl)
db_f = fsp.snd_f

fsp.calcSound()
fsp_data = fsp.data
freqs = fsp.return_freqs
in_array = fsp_data[hrtf_n]
print(freqs)

print(in_array.shape)
print(in_array[:1015])
plt.plot(freqs,in_array[:1015])
plt.plot(freqs,in_array[1015:])
plt.title("NN-Input dB:{}".format(dbl))
plt.xlabel("Frequency")
plt.ylabel("dB SPL")
plt.show()


"""
plt.plot(noise_t)
plt.plot(band_t)
plt.title("Time")
plt.show()

plt.plot(freqs,noise_f)
plt.plot(freqs,band_f)
plt.title("Frequency")
plt.show()
"""
