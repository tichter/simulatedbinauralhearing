from soundProcessor import SoundProcessor
from hrtfloader import HRTFLoader

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

import pickle
import os

class DataSet(Dataset):
    def __init__(self,nsounds,db_levels=[45,60,75],train=True,randmanip=False,add_ele=False,doublepolar=False):
        super(DataSet,self).__init__()

        self.datasetpath = "D:\\1_Uni\\0_Master\\5_CI-Thesis\\01Deliverable\\Code\\datasets\\"
        self.name = "{}-{}-{}-{}-{}-{}.pkl".format(nsounds,db_levels,train,randmanip,add_ele,doublepolar)
        self.edhrtfloader = HRTFLoader(ED=True,add_ele=add_ele,doublepolar=doublepolar)


        self.nsounds = nsounds
        self.db_levels = db_levels
        self.train = train
        self.randmanip = randmanip
        self.add_ele = add_ele
        self.doublepolar = doublepolar

        self.sps = []
        self.data = []
        self.labels = []

        self.bands = [[100,12000],[100,1000],[1000,2000],[2000,3000],[3000,4000],
                      [4000,5000],[5000,6000],[7000,8000],[8000,9000],[9000,10000],
                      [1000,5000],[5000,12000],[2000,8000]]

        # read created dataset or create new one
        if os.path.exists(self.datasetpath+self.name):
            self.loadDataset()
        else:
            self.createDataset()

        self.calculateData()

    def createDataset(self):
        if self.randmanip:
            for i in tqdm(range(self.nsounds)):
                dbl = np.random.choice(self.db_levels)
                band = self.bands[np.random.choice(len(self.bands))]
                print("db: {} band: {}".format(dbl,band))
                sp = SoundProcessor(db=dbl,add_ele=self.add_ele,doublepolar=self.doublepolar,edhrtfloader=self.edhrtfloader)
                sp.generateNoise()
                sp.bandpassfilter(f_min=band[0],f_max=band[1])
                sp.setdBSPL()
                sp.calcSound()
                self.sps.append(sp)
        else:
            for i in tqdm(range(self.nsounds)):
                for dbl in self.db_levels:
                    if self.train:
                        for band in self.bands:
                            sp = SoundProcessor(db=dbl,add_ele=self.add_ele,doublepolar=self.doublepolar,edhrtfloader=self.edhrtfloader)
                            sp.generateNoise()
                            sp.bandpassfilter(f_min=band[0],f_max=band[1])
                            sp.setdBSPL()
                            sp.calcSound()
                            self.sps.append(sp)
                    else:
                        sp = SoundProcessor(db=dbl,add_ele=False,doublepolar=self.doublepolar,edhrtfloader=self.edhrtfloader)
                        sp.generateNoise()
                        sp.bandpassfilter()
                        sp.setdBSPL()
                        sp.calcSound()
                        self.sps.append(sp)
        self.writeDataset()


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return (torch.from_numpy(self.data[idx]).float(), torch.from_numpy(np.array(self.labels[idx])).float())

    def calculateData(self):
        self.data = []
        self.labels = []

        for sp in self.sps:
            data,labels = sp.getData()
            [self.data.append(d) for d in data]
            [self.labels.append(l) for l in labels]

    def highpassFilter(self,f_min=3000,f_max=12000):
        for sp in self.sps:
            sp.bandpassfilter(f_min=f_min,f_max=f_max)
            sp.setdBSPL()
            sp.calcSound()

        self.calculateData()

    def lowpassFilter(self,f_min=100,f_max=1500):
        for sp in self.sps:
            sp.bandpassfilter(f_min=f_min,f_max=f_max)
            sp.setdBSPL()
            sp.calcSound()

        self.calculateData()

    def broadbandFilter(self,f_min=100,f_max=12000):
        for sp in self.sps:
            sp.bandpassfilter(f_min=f_min,f_max=f_max)
            sp.setdBSPL()
            sp.calcSound()

        self.calculateData()

    def invertChannels(self,f_min=100,f_max=12000):
        self.data = []
        self.labels = []
        for sp in self.sps:
            sp.bandpassfilter(f_min=f_min,f_max=f_max)
            sp.setdBSPL()
            sp.calcSound()

            data,labels = sp.getData()

            # swap channels
            data = [np.concatenate([d[int(len(d)/2):],d[:int(len(d)/2)]]) for d in data]
            [self.data.append(d) for d in data]
            [self.labels.append(l) for l in labels]

    def getFreqBins(self):
        print(len(self.sps[0].return_freqs))
        return self.sps[0].return_freqs

    def writeDataset(self):
        print("Writing dataset...")
        print("{}".format(self.datasetpath+self.name))
        with open(self.datasetpath+self.name,'wb') as f:
            for sp in self.sps:
                pickle.dump(sp,f)

    def loadDataset(self):
        print("Dataset already created...")
        print("{}".format(self.datasetpath+self.name))
        with open(self.datasetpath+self.name,'rb') as f:
            eof = False
            while not eof:
                try:
                    self.sps.append(pickle.load(f))
                except EOFError as error:
                    eof = True
