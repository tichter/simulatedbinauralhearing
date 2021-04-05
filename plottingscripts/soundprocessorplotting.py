import sys
sys.path.insert(0,"D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\utils")

from freqSoundProcessor import FreqSoundProcessor
from hrtfloader import HRTFLoader
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'figure.autolayout': True})
"""
Generate noise
bandpass filter -> in frequency domain with raised cosine filter
set dB
convolve
convert to dB
"""

add_ele = False
doublepolar = False
hrtf_n = 9

intensity = np.random.randint(40,91)
intensity = 90
band_start = np.random.randint(100,6001)
band_start = 2000
octaves = np.random.randint(2,5)
octaves = 4

edhrtfloader = HRTFLoader(ED=True,add_ele=add_ele,doublepolar=doublepolar)

sp = FreqSoundProcessor(db=intensity,add_ele=add_ele,doublepolar=doublepolar,edhrtfloader=edhrtfloader)

sp.generateNoise()
noise_t = sp.snd_t
noise_f = sp.snd_f

sp.bandpassfilter(f_min=band_start,f_max=min(band_start*octaves,12000))
bandpass_t = sp.snd_t
bandpass_f = sp.snd_f

sp.setdBSPL()
db_t = sp.snd_t
db_f = sp.snd_f

sp.calcSound()
sp_data = sp.data

in_array = sp_data[hrtf_n]
sp_label = sp.labels[hrtf_n]

freqs = sp.freqs
return_freqs = sp.return_freqs

print("Noise plot")
plt.plot(np.linspace(0,0.085,len(noise_t)),noise_t,'black')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.ylim([-4,4])
plt.title("Gaussian White Noise")
plt.show()
plt.clf()

print("Bandpass plot")
plt.semilogx(freqs[:np.power(2,11)],bandpass_f[:np.power(2,11)],'black')
plt.xlabel("Frequency (kHz)")
plt.ylabel("Amplitude")
plt.ylim([-200,200])
plt.title("Bandpassfiltered \n f_min: {} with {} octaves".format(band_start, octaves-1))
plt.show()
plt.clf()

print("DB plot")
plt.semilogx(freqs[:np.power(2,11)],db_f[:np.power(2,11)],'black')
plt.xlabel("Frequency (kHz)")
plt.ylabel("Amplitude (dBSPL)")
plt.ylim([-200,200])
plt.title("{} dB SPL adjusted".format(intensity))
plt.show()
plt.clf()

print("Result plot")
plt.semilogx(return_freqs,in_array[:1015],color='blue')
plt.semilogx(return_freqs,in_array[1015:],color='red')
plt.legend(["Left Ear","Right Ear"])
plt.xlabel("Frequency (kHz)")
plt.ylabel("Scaled Amplitude\n(dBSPL/100)")
plt.title("NN-Input Azimuth: {}".format(sp_label*180))
plt.show()
