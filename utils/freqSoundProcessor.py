import numpy as np
import scipy
import scipy.signal
import librosa

from hrtfloader import HRTFLoader

class FreqSoundProcessor():
    def __init__(self,duration=0.085,db=60,f_min=100,f_max=12000,n_fft=np.power(2,12),fs=48000,targetlabeltype="ONE",add_ele=False,doublepolar=False,edhrtfloader=None):
        #print("Frequency Soundprocessor init")
        self.n_fft = n_fft
        self.fs = fs
        self.fs_hrtf = 48000

        self.data = None
        self.labels = None
        self.targetlabeltype = targetlabeltype
        self.currentlabeltype = None

        self.duration = duration
        self.n_samples = int(self.duration*self.fs)
        self.db = db
        self.f_min = f_min
        self.f_max = f_max
        self.f_min_cut = 100
        self.f_max_cut = 12000

        self.snd_o = None
        self.snd_t = None
        self.snd_f = None
        self.freqs = None
        self.return_freqs = None

        if edhrtfloader is None:
            self.edhrtfloader = HRTFLoader(ED=True,add_ele=add_ele,doublepolar=doublepolar)
        else:
            self.edhrtfloader = edhrtfloader

        self.currentlabeltype = "DEG"

    def generateNoise(self):
        self.snd_t = np.random.normal(size=self.n_samples)
        self.snd_o = self.snd_t

        self.snd_f = np.fft.fft(self.snd_t,n=self.n_fft)
        self.freqs = np.fft.fftfreq(n=self.n_fft,d=1.0/self.fs)

        unique = int(np.ceil((len(self.freqs)+1)/2.0))
        self.return_freqs = self.freqs[0:unique]

    def bandpassfilter(self,f_min=None,f_max=None):
        self.snd_f = np.fft.fft(self.snd_t,n=self.n_fft)
        # inspired by
        # https://github.com/alessandro-gentilini/opencv_exercises-butterworth/issues/1
        if f_min is not None and f_max is not None:
            self.f_min = f_min
            self.f_max = f_max
            #print("SP, SET f_min: {} f_max: {}".format(self.f_min,self.f_max))

        lowbound = (np.abs(self.freqs-self.f_min)).argmin()
        highbound = (np.abs(self.freqs-self.f_max)).argmin()
        self.snd_f[:lowbound] = 0
        self.snd_f[-lowbound:] = 0
        self.snd_f[highbound:-highbound] = 0

        self.snd_t = np.fft.ifft(self.snd_f,n=self.n_fft)

    def setdBSPL(self,db=None):
        # Inspired by
        # http://samcarcagno.altervista.org/blog/basic-sound-processing-python/?doing_wp_cron=1610980634.9656710624694824218750
        if db is not None:
            self.db = db

        P0 = 20e-6
        n = len(self.snd_f)
        unique = int(np.ceil((n+1)/2.0))
        #print(unique)
        p_curr = self.snd_f[0:unique]
        p_curr = abs(p_curr)

        p_curr = p_curr/float(n)
        p_curr = p_curr**2

        if n % 2 > 0: # we've got odd number of points fft
            p_curr[1:len(p_curr)] = p_curr[1:len(p_curr)] * 2
        else:
            p_curr[1:len(p_curr) -1] = p_curr[1:len(p_curr) - 1] * 2 # we've got even number of points fft

        p_curr = np.sqrt(np.mean(p_curr))
        db_curr = 20*np.log10(p_curr/P0)
        #print(db_curr)

        new_pressure = np.power(10,self.db/20) * P0
        #print("Current P: {}, Goal P: {}".format(p_curr,new_pressure))
        self.snd_f = self.snd_f * (new_pressure/p_curr)

        #debug
        p_curr = self.snd_f[0:unique]
        p_curr = abs(p_curr)

        p_curr = p_curr/float(n)
        p_curr = p_curr**2

        if n % 2 > 0: # we've got odd number of points fft
            p_curr[1:len(p_curr)] = p_curr[1:len(p_curr)] * 2
        else:
            p_curr[1:len(p_curr) -1] = p_curr[1:len(p_curr) - 1] * 2 # we've got even number of points fft

        p_curr = np.sqrt(np.mean(p_curr))
        db_curr = 20*np.log10(p_curr/P0)
        #print(db_curr)

    def calcSound(self,single=True):
        n_hrtfs = len(self.edhrtfloader)

        self.data = []
        self.labels = []
        self.currentlabeltype = "DEG"

        if single:
            i = np.random.randint(n_hrtfs)
            in_left = None
            in_right = None

            unique = int(np.ceil((len(self.freqs)+1)/2.0))
            self.return_freqs = self.freqs[0:unique]

            hrtf,label = self.edhrtfloader[i]
            in_left = self.placeNH(channel="L",hrtf=hrtf)
            in_right = self.placeNH(channel="R",hrtf=hrtf)

            cut_idx = (self.return_freqs<self.f_max_cut) * (self.return_freqs>self.f_min_cut)
            self.return_freqs = self.return_freqs[cut_idx]
            combined = in_left[:,cut_idx] + in_right[:,cut_idx]
            self.data.append(np.concatenate((combined[0],combined[1])))
            self.labels.append(label)

        else:
            for i in range(n_hrtfs):
                in_left = None
                in_right = None

                unique = int(np.ceil((len(self.freqs)+1)/2.0))
                self.return_freqs = self.freqs[0:unique]

                hrtf,label = self.edhrtfloader[i]
                in_left = self.placeNH(channel="L",hrtf=hrtf)
                in_right = self.placeNH(channel="R",hrtf=hrtf)

                cut_idx = (self.return_freqs<self.f_max_cut) * (self.return_freqs>self.f_min_cut)
                self.return_freqs = self.return_freqs[cut_idx]
                combined = in_left[:,cut_idx] + in_right[:,cut_idx]
                self.data.append(np.concatenate((combined[0],combined[1])))
                self.labels.append(label)

        self.setLabelType(self.targetlabeltype)

        # TODO: Cutting and labels

    def placeNH(self,channel,hrtf):


        """
        import matplotlib.pyplot as plt
        plt.plot(hrtf[0,:])
        plt.plot(np.fft.fft(hrtf[0,:],n=8128))
        plt.plot(np.fft.fft(hrtf[0,:],n=self.n_fft))

        plt.plot(hrtf[1,:])
        plt.show()
        """
        result = np.zeros((2,len(self.return_freqs)),dtype='f')
        if channel is "L":
            hrtf_f = np.abs(np.fft.fft(hrtf[0,:],n=self.n_fft)) #get only the magnitudes
            l_channel_f = self.snd_f #* 10**(hrtf_f/20)

            unique = int(np.ceil((len(l_channel_f)+1)/2.0))
            l_channel_f = l_channel_f[0:unique]
            l_channel_f = np.abs(l_channel_f)/float(len(l_channel_f))
            #l_channel_f = l_channel_f**2
            l_channel_f = l_channel_f * 10**(hrtf_f[0:unique]/20) #convert dB of HRTF to pressure change
            l_channel_f = l_channel_f+20e-6
            l_channel_db = 20*np.log10(l_channel_f/20e-6)/100

            result[0,:] = l_channel_db
        elif channel is "R":
            hrtf_f = np.abs(np.fft.fft(hrtf[1,:],n=self.n_fft))
            r_channel_f = self.snd_f #* 10**(hrtf_f/20)

            unique = int(np.ceil((len(r_channel_f)+1)/2.0))
            r_channel_f = r_channel_f[0:unique]
            r_channel_f = np.abs(r_channel_f)/float(len(r_channel_f))
            #r_channel_f = r_channel_f**2
            r_channel_f = r_channel_f * 10**(hrtf_f[0:unique]/20) #TEMP
            r_channel_f = r_channel_f+20e-6
            r_channel_db = 20*np.log10(r_channel_f/20e-6)/100

            result[1,:] = r_channel_db
        else:
            print("Placing Normal Hearing, Channel not recognized")

        return result


    def getData(self):
        if self.data is None:
            print("No data calculated before getting the data from the SoundProcessor.")
        return self.data,self.labels

    def getSNDF(self):
        cut_idx = (self.return_freqs<self.f_max_cut) * (self.return_freqs>self.f_min_cut)

        unique = int(np.ceil((len(self.snd_f)+1)/2.0))
        snd_f = self.snd_f[0:unique]
        snd_f = np.abs(snd_f)/float(len(snd_f))
        snd_f = snd_f**2
        snd_f = snd_f+20e-6
        return 20*np.log10(snd_f[cut_idx]/20e-6)/100


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
