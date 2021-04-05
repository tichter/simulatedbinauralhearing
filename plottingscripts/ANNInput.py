import sys
sys.path.insert(0,"D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\utils")
from freqSoundProcessor import FreqSoundProcessor

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})
#matplotlib.rcParams.update({'figure.autolayout': True})

f_min = 1000
octaves = 5
fsp = FreqSoundProcessor(db=60,add_ele=True,doublepolar=True)
fsp.generateNoise()
fsp.bandpassfilter(f_min=f_min,f_max=f_min*octaves)
fsp.setdBSPL()
fsp.calcSound(single=False)

data = fsp.data
labels = fsp.labels
labels = [l*180 for l in labels]
freqs = fsp.return_freqs

d1 = np.array(data)[np.array(labels)==-14.999999999999996].flatten()

print(d1.shape)

fsp = FreqSoundProcessor(db=80,add_ele=True,doublepolar=True)
fsp.generateNoise()
fsp.bandpassfilter(f_min=100,f_max=12000)
fsp.setdBSPL()
fsp.calcSound(single=False)

data = fsp.data
labels = fsp.labels
labels = [l*180 for l in labels]
freqs = fsp.return_freqs

d2 = np.array(data)[np.array(labels)==90.0].flatten()

dl1 = d1[:1015]
dr1 = d1[1015:]
ild1 = (dr1 - dl1)*100

dl2 = d2[:1015]
dr2 = d2[1015:]
ild2 = (dr2 - dl2)*100

ax1 = plt.subplot(311)
plt.semilogx(freqs,dl1,'blue')#'aqua')
plt.semilogx(freqs,dr1,'red')#'salmon')
plt.legend(["Left Ear","Right Ear"])#,"ILD"])
#plt.xlabel("Frequency")
plt.ylabel("Amplitude\n(dB SPL/100)")
plt.ylim([-0.1,1.1])
#plt.hlines(0.0,0.0,freqs[-1])
#plt.title("Neural Network Input -15 degree")
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.subplot(312,sharex = ax1)
plt.semilogx(freqs,dl2,'blue')
plt.semilogx(freqs,dr2,'red')
#plt.legend(["Left Ear","Right Ear"])#,"ILD"])
#plt.xlabel("Frequency")
plt.ylabel("Amplitude\n(dB SPL/100)")
plt.ylim([-0.1,1.1])
#plt.hlines(0.0,0.0,freqs[-1])
#plt.title("Neural Network Input 90 degree")

ax3 = plt.subplot(313)#,sharex = ax2)
plt.semilogx(freqs,ild1,'grey')
plt.semilogx(freqs,ild2,'black')
plt.legend(["-15 deg","90 deg"])#,"ILD"])
plt.xlabel("Frequency (Hz)")
plt.ylabel("ILD (dB SPL)")

f1 = matplotlib.ticker.ScalarFormatter()
f1.set_scientific(False)
ax3.xaxis.set_major_formatter(f1)
#plt.hlines(0.0,0.0,freqs[-1])
#plt.title("ILDs")
#plt.ylim([-25,55])
plt.subplots_adjust(hspace=.0)

plt.show()
