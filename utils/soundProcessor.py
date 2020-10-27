import numpy as np
import scipy
import scipy.signal
import librosa
from tqdm import tqdm

from hrtfloader import HRTFLoader

class SoundProcessor():
    def __init__(self,duration=0.085,db=60,f_min=100,f_max=12000,n_fft=np.power(2,12),fs=48000,targetlabeltype="ONE",add_ele=False,doublepolar=False,edhrtfloader=None):
        print("Soundprcessor init")
        self.n_fft = n_fft
        self.fs = fs
        self.fs_hrtf = 48000

        #self.dataL = [] #TEMP
        #self.dataR = [] #TEMP

        self.data = None
        self.lables = None
        self.targetlabeltype = targetlabeltype
        self.currentlabeltype = None

        self.duration = duration
        self.n_samples = int(self.duration*self.fs)
        self.db = db
        self.f_min = f_min
        self.f_max = f_max
        self.f_min_cut = 100
        self.f_max_cut = 12000

        self.snd_t = None # original sound time
        self.snd_f = None # original sound frequency
        self.freqs = None
        self.return_freqs = None

        self.snd_fil_t = None # filtered sound time
        self.snd_fil_f = None # filtered sound frequency

        if edhrtfloader is None:
            self.edhrtfloader = HRTFLoader(ED=True,add_ele=add_ele,doublepolar=doublepolar)
        else:
            self.edhrtfloader = edhrtfloader

        self.currentlabeltype = "DEG"

    def generateNoise(self):
        """
        Generates gaussian noise, bandpass filters it using the butter filter, and sets the dB
        """
        #self.snd_t = np.random.uniform(-1,1,n_samples) # sound in time domain
        self.snd_t = np.random.normal(size=self.n_samples) # gaussian sound in time domain

        self.snd_f = np.fft.rfft(self.snd_t,n=self.n_fft) # sound in frequency domain
        self.freqs = np.fft.rfftfreq(n=self.n_fft,d=1.0/self.fs) # frequency labels

        """
        import matplotlib.pyplot as plt

        plt.plot(np.linspace(0,self.n_samples/self.fs,self.n_samples),self.snd_t)
        plt.xlabel("Time in s")
        plt.ylabel("Amplitude")
        plt.title("Gaussian White Noise Time Domain")
        plt.show()
        """

    def bandpassfilter(self,f_min=None,f_max=None):
        """
        Butterworth Bandpass filters the sound and adjusts the dB level
        """
        if f_min is not None and f_max is not None:
            self.f_min = f_min
            self.f_max = f_max
            print("SP, SET f_min: {} f_max: {}".format(self.f_min,self.f_max))

        butter_order = 6
        cutoff = [self.f_min,self.f_max]
        sos = scipy.signal.butter(butter_order,cutoff,'bandpass',output='sos',fs=self.fs)
        self.snd_fil_t = scipy.signal.sosfilt(sos,self.snd_t)
        self.snd_fil_f = np.fft.rfft(self.snd_fil_t,n=self.n_fft)

    def setdBSPL(self,db=None):
        if db is not None:
            self.db = db

        P0 = 20e-6 # reference pressure

        p = np.sqrt(np.mean(self.snd_fil_t**2)) # sound pressure is RMS of sound
        current_dBSPL = 20*np.log10(p/P0)
        new_pressure = np.power(10,self.db/20) * P0

        self.snd_fil_t = self.snd_fil_t * (new_pressure/p)
        self.snd_fil_f = np.fft.rfft(self.snd_fil_t,n=self.n_fft) # sound in frequency domain

        #debug
        p_new = np.sqrt(np.mean(self.snd_fil_t**2))
        new_dBSPL = 20*np.log10(p_new/P0)

        print("SP, dB SET: {}".format(new_dBSPL))

    def calcSound(self):
        n_hrtfs = len(self.edhrtfloader)

        self.data = []
        self.labels = []
        self.currentlabeltype = "DEG"

        #for i in tqdm(range(n_hrtfs)):
        for i in range(n_hrtfs):
            in_left = None
            in_right = None

            hrtf,label = self.edhrtfloader[i]
            in_left = self.placeNH(channel="L",hrtf=hrtf)
            in_right = self.placeNH(channel="R",hrtf=hrtf)

            # cut to appropriate frequency range
            cut_idx = (self.freqs<self.f_max_cut) * (self.freqs>self.f_min_cut)
            self.cut_idx = cut_idx # TEMP: for additional graphics
            self.return_freqs = self.freqs[cut_idx]
            combined = in_left[:,cut_idx] + in_right[:,cut_idx]

            self.data.append(np.concatenate((combined[0],combined[1])))
            self.labels.append(label)

        self.setLabelType(self.targetlabeltype)

    def placeNH(self,channel,hrtf):
        result = np.zeros((2,len(self.freqs)),dtype='f')
        if channel is "L":
            l_channel_t = self.resampleConv(self.snd_fil_t,self.fs,hrtf[0,:],self.fs_hrtf)
            #self.dataL.append(l_channel_t) #TEMP
            #l_channel_t = np.convolve(self.snd_fil_t,hrtf[0,:])
            l_channel_f = np.fft.rfft(l_channel_t,n=self.n_fft)
            l_channel_f = np.abs(l_channel_f)
            l_channel_db = (20*np.log10(l_channel_f/20e-6))/120 # normalize

            result[0,:] = l_channel_db
        elif channel is "R":
            r_channel_t = self.resampleConv(self.snd_fil_t,self.fs,hrtf[1,:],self.fs_hrtf)
            #elf.dataR.append(r_channel_t) #TEMP
            #r_channel_t = np.convolve(self.snd_fil_t,hrtf[1,:])
            r_channel_f = np.fft.rfft(r_channel_t,n=self.n_fft)
            r_channel_f = np.abs(r_channel_f)
            r_channel_db = (20*np.log10(r_channel_f/20e-6))/120 # normalize

            result[1,:] = r_channel_db
        else:
            print("Place Normal Hearing, channel not recognized")

        return result

    def getData(self):
        if self.data is None:
            print("No data calculated before getting the data from the SoundProcessor.")
        return self.data,self.labels

    def setLabelType(self,target):
        """
        Changes the labels accordingly.
        """
        if self.labels is None:
            print("Please calculate data before setting the label.")
            print("This should not happen.")
        if self.currentlabeltype is "DEG":
            if target is "ONE" or target == "ONE":
                self.labels = [self.deg_to_one(l) for l in self.labels]
            elif target is "RAD" or target == "RAD":
                self.labels = [self.deg_to_rad(l) for l in self.labels]
        elif self.currentlabeltype is "ONE":
            if target is "DEG" or target == "DEG":
                self.labels = [self.one_to_deg(l) for l in self.labels]
            elif target is "RAD" or target == "RAD":
                self.labels = [self.one_to_rad(l) for l in self.labels]
        elif self.currentlabeltype is "RAD":
            if target is "DEG" or target == "DEG":
                self.labels = [self.rad_to_deg(l) for l in self.labels]
            if target is "ONE" or target == "ONE":
                self.labels = [self.rad_to_one(l) for l in self.labels]
        else:
            print("Label type not recognized.")

        self.currentlabeltype = target

    def resampleConv(self,snd,fs_snd,transfun,fs_transfun):
        res_snd = librosa.core.resample(snd,fs_snd,fs_transfun)
        conv_snd = np.convolve(snd,transfun)
        return librosa.core.resample(conv_snd,fs_transfun,fs_snd)

    def rad_to_deg(self,rad):
        return rad*(180/np.pi)

    def deg_to_rad(self,deg):
        return deg*(np.pi/180)

    def one_to_deg(self,one):
        return one*180

    def deg_to_one(self,deg):
        return deg/180

    def one_to_rad(self,one):
        return self.deg_to_rad(self.one_to_deg(one))

    def rad_to_one(self,rad):
        return self.deg_to_one(self.rad_to_deg(rad))
