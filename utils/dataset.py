from freqSoundProcessor import FreqSoundProcessor
from hrtfloader import HRTFLoader

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

import pickle
import os
import uuid

class DataSet(Dataset):
    def __init__(self,name_add,nsounds=100000,db_levels=[],train=True,add_ele=False,doublepolar=False,single=True):
        super(DataSet,self).__init__()

        #self.curUUID = str(uuid.uuid4())
        #print("UUID is: {}".format(self.curUUID))

        self.datasetpath = "D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\datasets\\"
        self.edhrtfloader = HRTFLoader(ED=True,add_ele=add_ele,doublepolar=doublepolar)
        self.name = "{}-{}-{}-{}-{}".format(nsounds,add_ele,doublepolar,single,name_add)

        self.nsounds = nsounds
        self.train = train
        self.add_ele = add_ele
        self.doublepolar = doublepolar
        self.single = single

        self.db_levels = db_levels
        sp = FreqSoundProcessor(db=90,add_ele=self.add_ele,
                            doublepolar=self.doublepolar,edhrtfloader=self.edhrtfloader)
        sp.generateNoise()
        sp.bandpassfilter(f_min=100,f_max=12000)
        sp.setdBSPL()
        sp.calcSound(single=self.single)

        self.return_freqs = sp.return_freqs

        self.sps = []
        self.data = []
        self.labels = []

        self.intensities = []
        self.band_starts = []
        self.octaves = []

        # read created dataset or create new one
        if os.path.exists(self.datasetpath+self.name+"data.npy"):
            #print("Loader Currently Not Working")
            self.loadDataset()
        else:
            self.createDataset()
            np.save("D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\datasets\\intensities{}.npy".format(self.name),self.intensities)
            np.save("D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\datasets\\band_starts{}.npy".format(self.name),self.band_starts)
            np.save("D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\datasets\\octaves{}.npy".format(self.name),self.octaves)
            
        #self.calculateData()




    def createDataset(self):
        if self.train:
            for i in tqdm(range(self.nsounds)):
                intensity = np.random.randint(40,91)

                """
                #delete below until multiline comment
                bands = [[100,12000],[100,1000],[1000,2000],[2000,3000],[3000,4000],
                          [4000,5000],[5000,6000],[7000,8000],[8000,9000],[9000,10000],
                          [1000,5000],[5000,12000],[2000,8000]]

                band = bands[np.random.choice(len(bands))]
                band_start = band[0]
                band_end = band[1]

                """
                # this is the correct version
                band_start = np.random.randint(100,6001)
                band_end = np.random.randint(band_start+100,12000)

                self.intensities.append(intensity)
                self.band_starts.append(band_start)
                self.octaves.append(band_end)

                sp = FreqSoundProcessor(db=intensity,add_ele=self.add_ele,
                                    doublepolar=self.doublepolar,edhrtfloader=self.edhrtfloader)
                sp.generateNoise()
                if np.random.uniform() < 0.01:
                    sp.bandpassfilter(f_min=100,f_max=12000)
                else:
                    #sp.bandpassfilter(f_min=band_start,f_max=min(band_start*octaves,12000))
                    sp.bandpassfilter(f_min=band_start,f_max=band_end)
                sp.setdBSPL()
                sp.calcSound(single=self.single)
                data,labels = sp.getData()
                [self.data.append(d) for d in data]
                [self.labels.append(l) for l in labels]

                #self.sps.append(sp)
            self.writeDataset()

        else:
            for i in tqdm(range(self.nsounds)):
                for dbl in self.db_levels:
                    sp = FreqSoundProcessor(db=dbl,add_ele=self.add_ele,
                                        doublepolar=self.doublepolar,edhrtfloader=self.edhrtfloader)
                    sp.generateNoise()
                    sp.bandpassfilter()
                    sp.setdBSPL()
                    sp.calcSound(single=self.single)
                    self.sps.append(sp)

            self.calculateData()


    def calculateData(self):
        self.data = []
        self.labels = []

        for sp in self.sps:
            data,labels = sp.getData()
            [self.data.append(d) for d in data]
            [self.labels.append(l) for l in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return (torch.from_numpy(self.data[idx]).float(), torch.from_numpy(np.array(self.labels[idx])).float())

    def highpassFilter(self,f_min=3000,f_max=12000):
        for sp in self.sps:
            sp.bandpassfilter(f_min=f_min,f_max=f_max)
            sp.setdBSPL()
            sp.calcSound(single=False)

        self.calculateData()

    def lowpassFilter(self,f_min=100,f_max=1500):
        for sp in self.sps:
            sp.bandpassfilter(f_min=f_min,f_max=f_max)
            sp.setdBSPL()
            sp.calcSound(single=False)

        self.calculateData()

    def broadbandFilter(self,f_min=100,f_max=12000):
        for sp in self.sps:
            sp.bandpassfilter(f_min=f_min,f_max=f_max)
            sp.setdBSPL()
            sp.calcSound(single=False)

        self.calculateData()

    def getFreqBins(self):
        print(len(self.return_freqs))
        return self.return_freqs

    def writeDataset(self):
        print("Writing dataset...")
        print("{}".format(self.datasetpath+self.name))
        np.save("{}".format(self.datasetpath+self.name+"data.npy"),self.data)
        np.save("{}".format(self.datasetpath+self.name+"labels.npy"),self.labels)

        """
        with open(self.datasetpath+self.name,'wb') as f:
            for sp in self.sps:
                pickle.dump(sp,f)
        """
    def loadDataset(self):
        print("Dataset already created...")
        print("{}".format(self.datasetpath+self.name))
        self.data = np.load("{}".format(self.datasetpath+self.name+"data.npy"))
        self.labels = np.load("{}".format(self.datasetpath+self.name+"labels.npy"))

        """
        with open(self.datasetpath+self.name,'rb') as f:
            eof = False
            while not eof:
                try:
                    self.sps.append(pickle.load(f))
                except EOFError as error:
                    eof = True
        """
