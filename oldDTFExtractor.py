import sys
sys.path.insert(0,"D:\\1_Uni\\0_Master\\5_CI-Thesis\\01Deliverable\\Code\\utils")

from hrtfloader import HRTFLoader
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})
import numpy as np

input_size = 2030
hidden_size = 20
test_set_size = 10
db_levels = [45,60,75,90,100]
plot_dtf = True
plot_dtf_ild = False
plot_spatial = False
bbhplp = False
db_spatialtuning = 60
h20_n1 = 16
h20_n2 = 18

freqs = np.fft.rfftfreq(n=np.power(2,12),d=1.0/48000)
cut_idx = (freqs<12000) * (freqs>100)

hrtf = HRTFLoader(ED=True,add_ele=True)
#hrtf = HRTFLoader(ED=True)

print(np.array(hrtf.returnHRTFs).shape)
m_hrtf_l = np.sqrt(np.mean(np.array(hrtf.returnHRTFs[:][0][:])**2,axis=0))
m_hrtf_r = np.sqrt(np.mean(np.array(hrtf.returnHRTFs[:][1][:])**2,axis=0))

m_hrtf_l = np.mean(np.array(hrtf.returnHRTFs[:][0][:]),axis=0)
m_hrtf_r = np.mean(np.array(hrtf.returnHRTFs[:][1][:]),axis=0)

m_hrtf = np.array([m_hrtf_l,m_hrtf_r])

h90L = hrtf.returnHRTFs[13]-m_hrtf
h90R = hrtf.returnHRTFs[12]-m_hrtf

h90LF = np.fft.rfft(h90L,n=np.power(2,12))
h90LF = np.abs(h90LF)
h90LF = h90LF[:,cut_idx]

h90RF = np.fft.rfft(h90R,n=np.power(2,12))
h90RF = np.abs(h90RF)
h90RF = h90RF[:,cut_idx]

p0 = 20e-6
p = np.sqrt(np.mean(np.array(h90LF[0])**2))
current_dBSPL = 20*np.log10(p/p0)
new_pressure = np.power(10,60/20)*p0

h90LF = h90LF * (new_pressure)
h90RF = h90RF * (new_pressure)

if plot_dtf:
    NUM_COLORS = len(hrtf.returnHRTFs)
    F = freqs
    cut_idx = (F<12000) * (F>100)
    F = F[cut_idx]

    LH = []
    RH = []
    for h in hrtf.returnHRTFs:
        h = h - m_hrtf
        hF = np.fft.rfft(h,n=np.power(2,12))
        hF = np.abs(hF)
        hF = hF[:,cut_idx]

        """
        p0 = 20e-6
        p_0 = np.sqrt(np.mean(np.array(hF[0])**2))
        p_1 = np.sqrt(np.mean(np.array(hF[1])**2))

        new_pressure = np.power(10,60/20)*p0
        new_pressure = np.power(10,60/20)*p0

        hF[0] = hF[0] * (new_pressure)
        hF[1] = hF[1] * (new_pressure)
        """

        LH.append(hF[0])
        RH.append(hF[1])

    labels_fl = [float(x) for x in hrtf.returnLabels]
    labels_ele = [float(x) for x in hrtf.returnLabelsEle]


    #LH = [x for _,x in sorted(zip(labels_fl,LH))]
    #RH = [x for _,x in sorted(zip(labels_fl,RH))]

    #labels_fl = sorted(labels_fl)


    # temp
    import scipy.io as io
    DTFdict = {}
    print(np.array(labels_fl).shape)
    DTFdict['left'] = np.array(LH)
    DTFdict['right'] = np.array(RH)
    DTFdict['labels'] = np.array(labels_fl)
    DTFdict['labels_ele'] = np.array(labels_ele)
    DTFdict['frequency'] = np.array(F)
    io.savemat('OlHeaD-DTFs.mat',DTFdict)


    if plot_dtf_ild:
        cm = plt.get_cmap('plasma')
        fig, ax1 = plt.subplots(1,1)
        ax1.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        for i in range(len(LH)):
            ax1.semilogx(F,LH[i]-RH[i],label=labels_fl[i])
        ax1.legend(loc='upper left')#, bbox_to_anchor=(1.2, 1.0))
        ax1.set_title("OlHead DTFs ILD")
        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("Amplitude")
        plt.show()
    else:
        cm = plt.get_cmap('plasma')
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        ax2.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        for i in range(len(LH)):
            ax2.semilogx(F,RH[i],label=labels_fl[i])
            ax1.semilogx(F,LH[i],label=labels_fl[i])
        ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax1.set_title("OlHead DTFs Left Ear")
        ax2.set_title("OlHead DTFs Right Ear")
        ax2.set_xlabel("Frequency")
        ax1.set_ylabel("Amplitude")
        ax2.set_ylabel("Amplitude")
        plt.show()
