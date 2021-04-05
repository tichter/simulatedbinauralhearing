import sys
sys.path.insert(0,"D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\utils")
from hrtfloader import HRTFLoader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

plt.rcParams.update({'font.size': 14})

def toDB(sfreq):
    unique = int(np.ceil((len(sfreq)+1)/2.0))
    sfreq = sfreq[0:unique]
    sfreq = np.abs(sfreq)/float(len(sfreq))
    sfreq = sfreq**2
    sfreq = sfreq+20e-6
    return 20*np.log10(sfreq/20e-6)

fs = 48000
n_fft=np.power(2,12)

hl = HRTFLoader(ED=True)
hrtf,labels = hl.getAzimuthFrontal()
hrtfs = np.array(hrtf)

print(hrtfs.shape)
print(labels)

diff_f = []
freq = np.fft.rfftfreq(n=n_fft,d=1.0/fs)

cut_idx = (freq<12000)*(freq>100)
freq_cut = freq[cut_idx]

for i in range(25):
    fft_l = np.abs(np.fft.fft(hrtfs[i,0,:],n=n_fft))
    fft_r = np.abs(np.fft.fft(hrtfs[i,1,:],n=n_fft))


    #fft_l = toDB(fft_l)
    #fft_r = toDB(fft_r)

    diff = fft_r - fft_l
    print(diff.shape)

    unique = int(np.ceil((len(diff)+1)/2.0))
    diff = diff[0:unique]
    diff_f.append(diff[cut_idx])

print(np.array(diff_f).shape)
sorted_labels = sorted(labels)
sorted_diff = [x for _, x in sorted(zip(labels,diff_f), key=lambda pair: pair[0])]

plt.set_cmap("plasma")
color = iter(cm.plasma(np.linspace(0,1,25)))
for i,diff in enumerate(sorted_diff):
    plt.semilogx(freq_cut,diff,c=next(color))

plt.legend([str(i) for i in sorted_labels])
plt.title("OlHead HRTF ILDs Azimuth")
plt.xlabel("Frequency (Hz)")
plt.ylabel("ILD (dB)")
plt.show()
