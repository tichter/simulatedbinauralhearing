import sys
sys.path.insert(0,"D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\utils")
from dataset import DataSet
from networks import BinauralNeuralNetwork
from networks import L3BinauralNeuralNetwork
from networks import L3SplitBinauralNeuralNetwork
from freqSoundProcessor import FreqSoundProcessor

import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch.autograd import Variable

from sklearn.linear_model import LinearRegression

from scipy.ndimage import gaussian_filter1d
import scipy.io as io

class testNetwork():
    def __init__(self,modelpath,testsize,inputsize,db_levels,add_ele,doublepolar,hiddensize=None,hidden_size_L1=None,hidden_size_L2=None,L3=False,Split=False):
        self.testsize = testsize
        self.inputsize = inputsize
        self.hiddensize = hiddensize
        self.hidden_size_L1 = hidden_size_L1
        self.hidden_size_L2 = hidden_size_L2
        self.db_levels = db_levels
        self.L3 = L3
        self.Split = Split
        self.bandpassname = None
        self.add_ele = add_ele
        self.doublepolar = doublepolar

        self.network_labeltype = "ONE"

        self.testset = DataSet("test",nsounds=testsize,db_levels=db_levels,train=False,add_ele=add_ele,doublepolar=doublepolar,single=False)
        self.testloader = torch.utils.data.DataLoader(self.testset,batch_size=1,shuffle=False)

        if L3:
            if Split:
                self.model = L3SplitBinauralNeuralNetwork(input_size=self.inputsize,hidden_size_L1=hidden_size_L1,hidden_size_L2=hidden_size_L2).cuda()
            else:
                self.model = L3BinauralNeuralNetwork(input_size=self.inputsize,hidden_size_L1=hidden_size_L1,hidden_size_L2=hidden_size_L2).cuda()
        else:
            self.model = BinauralNeuralNetwork(input_size=self.inputsize,hidden_size=self.hiddensize).cuda()

        self.model.float()
        self.model.load_state_dict(torch.load(modelpath))

    def plotLosses(self):
        if self.L3:
            if self.Split:
                if self.add_ele:
                    loss_name = "models/losses/L3SplitbinauralNN-Final-ELE-Rand-{}-{}-{}-{}-{}.npy".format(self.inputsize,self.hidden_size_L1,self.hidden_size_L2,self.doublepolar,"{}")
                    loss_vali = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                    loss_train = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                    plotname = "LossLateFusionEle.png"
                    plottitle = "Late Fusion Loss"
                else:
                    loss_name = "models/losses/L3SplitbinauralNN-Final-Rand-{}-{}-{}-{}-{}.npy".format(self.inputsize,self.hidden_size_L1,self.hidden_size_L2,self.doublepolar,"{}")
                    loss_vali = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                    loss_train = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                    plotname = "LossLateFusion.png"
                    plottitle = "Late Fusion Loss"
            else:
                if self.add_ele:
                    loss_name = "models/losses/L3binauralNN-Final-ELE-Rand-{}-{}-{}-{}-{}.npy".format(self.inputsize,self.hidden_size_L1,self.hidden_size_L2,self.doublepolar,"{}")
                    loss_vali = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                    loss_train = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                    plotname = "LossEarlyFusionEle.png"
                    plottitle = "Early Fusion Loss"
                else:
                    loss_name = "models/losses/L3binauralNN-Final-Rand-{}-{}-{}-{}-{}.npy".format(self.inputsize,self.hidden_size_L1,self.hidden_size_L2,self.doublepolar,"{}")
                    loss_vali = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                    loss_train = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                    plotname = "LossEarlyFusion.png"
                    plottitle = "Early Fusion Loss"
        else:
            if self.add_ele:
                loss_name = "models/losses/binauralNN-Final-ELE-Rand-{}-{}-{}-{}.npy".format(self.inputsize,self.hiddensize,self.doublepolar,"{}")
                loss_vali = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                loss_train = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                plotname = "LossH{}Ele.png".format(self.hiddensize)
                plottitle = "{} Hidden Units".format(self.hiddensize)
            else:
                loss_name = "models/losses/binauralNN-Final-Rand-{}-{}-{}-{}.npy".format(self.inputsize,self.hiddensize,self.doublepolar,"{}")
                loss_vali = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                loss_train = np.load(loss_name.format("Lvalid"),allow_pickle=True)
                plotname = "LossH{}.png".format(self.hiddensize)
                plottitle = "{} Hidden Units".format(self.hiddensize)

        print(loss_train)
        print(loss_vali)

        plt.plot(loss_train)
        plt.plot(loss_vali)
        plt.legend(["Train Loss","Validation Loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(plottitle)
        plt.ylim(0,0.03)
        plt.savefig("plots\\testing\\{}".format(plotname))
        #plt.show()
        plt.clf()

    def testBroadBand(self,dbsplit=False):
        print("Testing Broadband")
        self.testset.broadbandFilter()
        self.testloader = torch.utils.data.DataLoader(self.testset,batch_size=1,shuffle=False)

        if self.bandpassname is None:
            plotname = self.hiddensize
        else:
            plotname = self.bandpassname

        if dbsplit:
            self.evalTestLoaderDBsplit("Broadband {}".format(plotname))
        else:
            self.evalTestLoader("Broadband {}".format(plotname))

    def testHighPass(self,dbsplit=False):
        print("Testing Highpass")
        self.testset.highpassFilter()
        self.testloader = torch.utils.data.DataLoader(self.testset,batch_size=1,shuffle=False)

        if self.bandpassname is None:
            plotname = self.hiddensize
        else:
            plotname = self.bandpassname

        if dbsplit:
            self.evalTestLoaderDBsplit("Highpass {}".format(plotname))
        else:
            self.evalTestLoader("Highpass {}".format(plotname))

    def testLowPass(self,dbsplit=False):
        print("Testing Lowpass")
        self.testset.lowpassFilter()
        self.testloader = torch.utils.data.DataLoader(self.testset,batch_size=1,shuffle=False)

        if self.bandpassname is None:
            plotname = self.hiddensize
        else:
            plotname = self.bandpassname

        if dbsplit:
            self.evalTestLoaderDBsplit("Lowpass {}".format(plotname))
        else:
            self.evalTestLoader("Lowpass {}".format(plotname))

    def evalTestLoader(self,plotname):
        target_angle = []
        pred_angle = []

        for i,data in enumerate(self.testloader):
            x = Variable(data[0]).cuda()
            output = self.model(x)

            for t_angle in data[1]:
                target_angle.append(t_angle)
            for p_angle in output.detach().cpu():
                pred_angle.append(p_angle)

        for i in range(len(target_angle)):
            plt.plot(self.getScaling(np.array(target_angle[i]),"DEG"),self.getScaling(np.array(pred_angle[i]),"DEG"),color='black',marker='.')

        target_angle = self.getScaling(np.array(target_angle),"DEG")
        pred_angle = self.getScaling(np.array(pred_angle),"DEG")

        print(np.array(target_angle).shape)
        print(np.array(pred_angle).shape)

        corr = scipy.stats.pearsonr(target_angle,pred_angle)
        print(corr)
        #reg = LinearRegression().fit(np.reshape(target_angle,(-1,1)),np.reshape(pred_angle,(-1,1)))

        #print(reg.score(np.reshape(target_angle,(-1,1)),np.reshape(pred_angle,(-1,1))))
        p = np.polyfit(target_angle,pred_angle,1)
        print(p)
        yfit = np.polyval(p,target_angle)
        error = np.mean(np.sqrt((target_angle-pred_angle)**2))
        print(error)
        plt.plot([-90,90],[-90,90],'black',label="Ideal")
        plt.plot(target_angle,yfit,'green',label="Regression")
        #plt.plot(target_angle,yfit-pred_angle,'red')
        plt.axes().set_aspect('equal')
        plt.xlabel("Stimulus Location (deg)")
        plt.ylabel("Response Location (deg)")
        plt.xlim(-95,95)
        plt.ylim(-95,95)
        plt.xticks(np.arange(-90,91,30))
        plt.yticks(np.arange(-90,91,30))
        plt.title("Testset {} Slope: {:.3f} Error: {:.3f}".format(plotname,p[0],error))
        plt.legend()
        plt.savefig("plots\\testing\\{}.png".format(plotname))
        plt.clf()

    def evalTestLoaderDBsplit(self,plotname):
        """
        Only works if one testset has size 1
        """
        print("#######")
        print("Not Person correlation")
        print("np polyfit like evalTestLoader")
        print("Must be fixed!!!")
        print("#######")

        db_levels = self.testset.db_levels

        target_angle = []
        pred_angle = []
        temp_tangle = []
        temp_pangle = []

        for i,data in enumerate(self.testloader):
            x = Variable(data[0]).cuda()
            output = self.model(x)

            for t_angle in data[1]:
                temp_tangle.append(t_angle)
            for p_angle in output.detach().cpu():
                temp_pangle.append(p_angle)

            if (i+1)%int(len(self.testset)/len(self.db_levels)) is 0 or i is len(self.testloader)-1:
                target_angle.append(temp_tangle)
                pred_angle.append(temp_pangle)
                temp_tangle = []
                temp_pangle = []

        for i in range(len(target_angle)):
            plt.plot(self.getScaling(np.array(target_angle[i]),"DEG"),self.getScaling(np.array(pred_angle[i]),"DEG"),'x')

        corr = scipy.stats.pearsonr(np.array(target_angle).flatten(),np.array(pred_angle).flatten())
        plt.plot([-90,90],[-90,90],'black')
        plt.xlabel("Stimulus Location (deg)")
        plt.ylabel("Response Location (deg)")
        plt.xlim(-95,95)
        plt.ylim(-95,95)
        plt.xticks(np.arange(-90,91,15))
        plt.yticks(np.arange(-90,91,15))
        plt.title("Testing {} Pearsonr: {:3f}".format(plotname,corr[0]))
        plt.legend(db_levels)
        plt.savefig("plots\\testing\\{}-dbsplit.png".format(plotname))
        plt.clf()

    def calcHiddenActivations(self,h_layer):
        self.hiddenActivations = []
        self.hiddenAngles = []

        n_hrtfs = int(len(self.testset)/len(self.db_levels))


        if (h_layer is 1 or h_layer is 2) and self.Split:
            curr_hidden_size = int(self.hidden_size_L1/2)
        elif h_layer is 3 and self.Split:
            curr_hidden_size = self.hidden_size_L2
        elif h_layer is 1 and self.L3:
            curr_hidden_size = self.hidden_size_L1
        elif h_layer is 2 and self.L3:
            curr_hidden_size = self.hidden_size_L2
        elif h_layer is 1 and not self.L3:
            curr_hidden_size = self.hiddensize

        # extract the data
        for i,data in enumerate(self.testloader):
            x = Variable(data[0]).cuda()
            output = self.model(x)
            #self.averageHiddenActivations.append(self.model.getAvgHiddenActivations())
            if self.L3:
                self.hiddenActivations.append(self.model.getHiddenActivations(h_layer).flatten())
            else:
                self.hiddenActivations.append(self.model.getHiddenActivations().flatten())
            #print(data[1])
            self.hiddenAngles.append(self.getScaling(data[1]))
            #print(self.hiddenAngles)
        self.hiddenActivations = np.array(self.hiddenActivations)

        #swap left and right

        self.hiddenActivations = np.concatenate([self.hiddenActivations[int(len(self.hiddenActivations)/2)+1:],self.hiddenActivations[:int(len(self.hiddenActivations)/2)+1]])
        self.hiddenAngles = np.concatenate([self.hiddenAngles[int(len(self.hiddenAngles)/2)+1:],self.hiddenAngles[:int(len(self.hiddenAngles)/2)+1]])

        # calculate score for each activation array, according to the direction.
        # Each neuron has an activation array per direction. Sort them, such that the neurons that activate high
        # on the far left come first
        weights = np.pad(np.linspace(1,0,num=8,dtype='f'),(0,n_hrtfs-8),'constant',constant_values=0)
        scores = np.matmul(self.hiddenActivations.T,weights).tolist()
        self.hiddenorder = scores
        self.hiddenActivations = np.array([x for _,x in sorted(zip(scores,self.hiddenActivations.T.tolist()))])
        self.yticklabels = [x for _,x in sorted(zip(scores,np.arange(curr_hidden_size)))] # keep the number of neurons also sorted

        ax = plt.gca(label="hidden")
        im = ax.imshow(np.array(self.hiddenActivations),cmap='hot',aspect='auto')#,vmin=0, vmax=1)
        cbar = ax.figure.colorbar(im,ax=ax)
        cbar.ax.set_ylabel("Val",rotation=-90,va="bottom")
        ax.set_xticks(np.arange(len(self.hiddenAngles)))
        ax.set_xticklabels(self.hiddenAngles)
        ax.tick_params(axis='x',labelrotation=90)
        ax.set_yticks(np.arange(curr_hidden_size)[::int((curr_hidden_size/41)+1)])
        ax.set_yticklabels(self.yticklabels[::int((curr_hidden_size/41)+1)])
        plt.title("Heatmap of the Layer activations dB: {}".format(self.db_levels))
        plt.ylabel("Neuron")
        plt.xlabel("Angle")
        #plt.show()
        plt.savefig("plots\\testing\\hiddenHeatMap {} L-{}.png".format(curr_hidden_size,h_layer))
        plt.clf()

        NUM_COLORS = len(self.yticklabels)
        cm = plt.get_cmap('hsv')
        fig, ax = plt.subplots(1,1)
        ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        for hiddenact in self.hiddenActivations:
            print(np.max(np.abs(hiddenact)))
            hiddenact = hiddenact/np.max(np.abs(hiddenact))
            plt.plot(self.hiddenAngles,hiddenact)
        plt.xticks(np.arange(-90,91,30))
        plt.xlabel("Sound Location ")
        plt.ylabel("Normalized Neuron activation")
        plt.title("Spatial Tuning dBSPL: {}".format(self.db_levels))
        if (not self.L3 and self.hiddensize is 2) or (not self.Split and h_layer is 2) or (self.Split and h_layer is 3):
            leg = ["Neuron {}".format(i) for i in self.yticklabels]
            plt.legend(leg,loc='upper right', bbox_to_anchor=(1.1, 1.05))

        plt.savefig("plots\\testing\\hiddenPlot {} L-{}.png".format(curr_hidden_size,h_layer))
        plt.clf()

    def plotSpatialTuning(self):
        self.hiddenActivationsBB1 = []
        self.hiddenActivationsBB2 = []
        self.hiddenActivationsBB3 = []

        self.hiddenActivationsHP1 = []
        self.hiddenActivationsHP2 = []
        self.hiddenActivationsHP3 = []

        self.hiddenActivationsLP1 = []
        self.hiddenActivationsLP2 = []
        self.hiddenActivationsLP3 = []

        self.hiddenAngles = []

        # Broadband
        self.testset.broadbandFilter()
        self.testloader = torch.utils.data.DataLoader(self.testset,batch_size=1,shuffle=False)

        for i,data in enumerate(self.testloader):
            x = Variable(data[0]).cuda()
            output = self.model(x)
            if self.L3:
                self.hiddenActivationsBB1.append(self.model.getHiddenActivations(1).flatten())
                self.hiddenActivationsBB2.append(self.model.getHiddenActivations(2).flatten())
                if self.Split:
                    self.hiddenActivationsBB3.append(self.model.getHiddenActivations(3).flatten())
            else:
                self.hiddenActivationsBB1.append(self.model.getHiddenActivations().flatten())

            self.hiddenAngles.append(self.getScaling(data[1]))

        # Highpass
        self.testset.highpassFilter()
        self.testloader = torch.utils.data.DataLoader(self.testset,batch_size=1,shuffle=False)

        for i,data in enumerate(self.testloader):
            x = Variable(data[0]).cuda()
            output = self.model(x)
            if self.L3:
                self.hiddenActivationsHP1.append(self.model.getHiddenActivations(1).flatten())
                self.hiddenActivationsHP2.append(self.model.getHiddenActivations(2).flatten())
                if self.Split:
                    self.hiddenActivationsHP3.append(self.model.getHiddenActivations(3).flatten())
            else:
                self.hiddenActivationsHP1.append(self.model.getHiddenActivations().flatten())

        # Lowpass
        self.testset.lowpassFilter()
        self.testloader = torch.utils.data.DataLoader(self.testset,batch_size=1,shuffle=False)

        for i,data in enumerate(self.testloader):
            x = Variable(data[0]).cuda()
            output = self.model(x)
            if self.L3:
                self.hiddenActivationsLP1.append(self.model.getHiddenActivations(1).flatten())
                self.hiddenActivationsLP2.append(self.model.getHiddenActivations(2).flatten())
                if self.Split:
                    self.hiddenActivationsLP3.append(self.model.getHiddenActivations(3).flatten())
            else:
                self.hiddenActivationsLP1.append(self.model.getHiddenActivations().flatten())

        # Plotting
        if self.Split:
            save_name = "plots\\spatialtuning\\{}\\{}.png".format("latefusion",{})
        elif self.L3:
            save_name = "plots\\spatialtuning\\{}\\{}.png".format("earlyfusion",{})
        else:
            save_name = "plots\\spatialtuning\\{}\\{}.png".format("h{}".format(self.hiddensize),{})


        self.hiddenActivationsBB1 = np.array(self.hiddenActivationsBB1)
        self.hiddenActivationsBB2 = np.array(self.hiddenActivationsBB2)
        self.hiddenActivationsBB3 = np.array(self.hiddenActivationsBB3)

        self.hiddenActivationsHP1 = np.array(self.hiddenActivationsHP1)
        self.hiddenActivationsHP2 = np.array(self.hiddenActivationsHP2)
        self.hiddenActivationsHP3 = np.array(self.hiddenActivationsHP3)

        self.hiddenActivationsLP1 = np.array(self.hiddenActivationsLP1)
        self.hiddenActivationsLP2 = np.array(self.hiddenActivationsLP2)
        self.hiddenActivationsLP3 = np.array(self.hiddenActivationsLP3)

        if self.Split:
            for neuron in range(len(self.hiddenActivationsBB3[0])):
                plt.plot(self.hiddenAngles,self.normalize(self.hiddenActivationsHP3[:,neuron]),'*',c='darkorange')
                plt.plot(self.hiddenAngles,self.normalize(self.hiddenActivationsLP3[:,neuron]),'*',c='green')
                plt.plot(self.hiddenAngles,self.normalize(self.hiddenActivationsBB3[:,neuron]),'*',c='black')
                plt.title("Spatial Tuning Neuron {} L3".format(neuron))
                plt.legend(["HP","LP","BB"])
                plt.xlabel("Stimulus Location (deg)")
                plt.ylabel("Normalized Neuron Activity")
                plt.ylim(-0.1,1.1)
                plt.savefig(save_name.format("L3 N{}".format(neuron)))
                plt.clf()

        if self.L3:
            for neuron in range(len(self.hiddenActivationsBB2[0])):
                plt.plot(self.hiddenAngles,self.normalize(self.hiddenActivationsHP2[:,neuron]),'*',c='darkorange')
                plt.plot(self.hiddenAngles,self.normalize(self.hiddenActivationsLP2[:,neuron]),'*',c='green')
                plt.plot(self.hiddenAngles,self.normalize(self.hiddenActivationsBB2[:,neuron]),'*',c='black')
                plt.title("Spatial Tuning Neuron {} L2".format(neuron))
                plt.legend(["HP","LP","BB"])
                plt.xlabel("Stimulus Location (deg)")
                plt.ylabel("Normalized Neuron Activity")
                if not self.Split:
                    plt.ylim(-0.1,1.1)
                plt.savefig(save_name.format("L2 N{}".format(neuron)))
                plt.clf()

        for neuron in range(len(self.hiddenActivationsBB1[0])):
            plt.plot(self.hiddenAngles,self.normalize(self.hiddenActivationsHP1[:,neuron]),'*',c='darkorange')
            plt.plot(self.hiddenAngles,self.normalize(self.hiddenActivationsLP1[:,neuron]),'*',c='green')
            plt.plot(self.hiddenAngles,self.normalize(self.hiddenActivationsBB1[:,neuron]),'*',c='black')
            plt.title("Spatial Tuning Neuron {} L1".format(neuron))
            plt.legend(["HP","LP","BB"])
            plt.xlabel("Stimulus Location (deg)")
            plt.ylabel("Normalized Neuron Activity")
            if not self.L3:
                plt.ylim(-0.1,1.1)
            plt.savefig(save_name.format("L1 N{}".format(neuron)))
            plt.clf()


    def plotDTFTuning(self):
        """
        Isn't working properly. Inputting the DTFs leads to very large activity, despite trying to adjust the level.
        It seems like the problem is that the DTFs have large amplitudes over wide spectra, which is a different
        Frequency profile compared to random noise, where consecutive frequencies are not related.
        """
        DTFdict = io.loadmat("OlHead-DTFs.mat")
        print(DTFdict)
        freqs = self.testset.getFreqBins()
        dtffreqs = DTFdict['frequency'][0]

        """
        print(DTFdict['left'][0].shape)
        print(min(DTFdict['left'][0]))
        print(max(DTFdict['left'][0]))
        print(20*np.log10(max(DTFdict['left'][0])/20e-6)/100)
        input("hi")
        """

        outputs = []
        activationsL1 = []
        activationsL2 = []
        activationsL3 = []

        azi_labels = []
        azi_idx = []

        for dtfidx in range(len(DTFdict['labels'][0])):
            if DTFdict['labels_ele'][0][dtfidx] == 0:
                azi_labels.append(DTFdict['labels'][0][dtfidx])
                azi_idx.append(dtfidx)

            inL = 20*np.log10(np.array(DTFdict['left'][dtfidx]/1)/20e-6)/100 # To DBSPL and scale 0-1
            inR = 20*np.log10(np.array(DTFdict['right'][dtfidx]/1)/20e-6)/100
            #inL = np.array(DTFdict['left'][dtfidx])/100000
            #inR = np.array(DTFdict['right'][dtfidx])/100000

            inNet = np.concatenate((inL,inR)).reshape(1,-1).astype(np.float32)
            x = Variable(torch.from_numpy(inNet)).cuda()
            outputs.append(self.model(x))

            if self.Split:
                activationsL1.append(self.model.getHiddenActivations(1)).flatten()
                activationsL2.append(self.model.getHiddenActivations(2)).flatten()
                activationsL3.append(self.model.getHiddenActivations(3)).flatten()
            elif self.L3:
                activationsL1.append(self.model.getHiddenActivations(1)).flatten()
                activationsL2.append(self.model.getHiddenActivations(2)).flatten()
            else:
                print(self.model.getHiddenActivations())
                activationsL1.append(self.model.getHiddenActivations().flatten()) #n_dtf x n_nodes_layer

        activationsL1 = np.array(activationsL1)
        print(activationsL1.shape)
        if self.add_ele:
            # 3d plot
            pass
        else:
            # 2d plot
            for neuron in range(len(activationsL1[0][:])):
                plt.title("Neuron {}".format(neuron))
                plt.plot(azi_labels,activationsL1[azi_idx,neuron],'*')
                plt.xlabel("DTF-Azi")
                plt.ylabel("Activity")
                plt.show()

    def plotILDTuning(self):
        ild_range = np.arange(-10,11,1)
        db0s = [40,60,80]
        sp = FreqSoundProcessor()
        sp.generateNoise()
        sp.bandpassfilter(f_min=100,f_max=12000)
        #sp.bandpassfilter(f_min=1000,f_max=8000)

        outputs = []
        activationsL1 = []
        activationsL2 = []
        activationsL3 = []

        for db0 in db0s:
            actL1 = []
            actL2 = []
            actL3 = []
            for ild in ild_range:
                # inL
                dbL = max(db0-ild,db0)
                sp.setdBSPL(dbL)
                inL = sp.getSNDF()
                # inR
                dbR = max(db0+ild,db0)
                sp.setdBSPL(dbR)
                inR = sp.getSNDF()
                print("DB L: {} DB R: {} ILD: {}-{}".format(dbL,dbR,dbR-dbL,ild))

                inNet = np.concatenate((inL,inR)).reshape(1,-1).astype(np.float32)
                x = Variable(torch.from_numpy(inNet)).cuda()
                outputs.append(self.model(x))

                if self.Split:
                    actL1.append(self.model.getHiddenActivations(1).flatten())
                    actL2.append(self.model.getHiddenActivations(2).flatten())
                    actL3.append(self.model.getHiddenActivations(3).flatten())
                elif self.L3:
                    actL1.append(self.model.getHiddenActivations(1).flatten())
                    actL2.append(self.model.getHiddenActivations(2).flatten())
                else:
                    #print(self.model.getHiddenActivations())
                    hiddenact = self.model.getHiddenActivations().flatten()
                    #hiddenact = hiddenact/np.max(np.abs(hiddenact))

                    actL1.append(hiddenact) #n_dtf x n_nodes_layer

            activationsL1.append(actL1)
            activationsL2.append(actL2)
            activationsL3.append(actL3)

        activationsL1 = np.array(activationsL1)
        activationsL2 = np.array(activationsL2)
        activationsL3 = np.array(activationsL3)

        if self.Split:
            save_name = "plots\\ildtuning\\{}\\{}.png".format("latefusion",{})
        elif self.L3:
            save_name = "plots\\ildtuning\\{}\\{}.png".format("earlyfusion",{})
        else:
            save_name = "plots\\ildtuning\\{}\\{}.png".format("h{}".format(self.hiddensize),{})

        if self.Split:
            for neuron in range(len(activationsL3[0][0][:])):
                for db0i in range(len(db0s)):
                    acts = activationsL3[db0i,:,neuron]
                    acts -= min(acts)
                    acts = acts/(np.max(np.abs(acts))+0.000001)
                    plt.title("ILD Tuning Neuron {} L3".format(neuron))
                    plt.plot(ild_range,acts,['o-','v-','s-'][db0i],c="black")
                    plt.xlabel("ILD (dB SPL)")
                    plt.ylabel("Normalized Activity")
                    plt.ylim(-0.1,1.1)
                plt.legend([str(db0s[0])+"dB SPL",str(db0s[1])+"dB SPL",str(db0s[2])+"dB SPL"])
                plt.savefig(save_name.format("L3 N{}".format(neuron)))
                plt.clf()
        if self.L3:
            for neuron in range(len(activationsL2[0][0][:])):
                for db0i in range(len(db0s)):
                    acts = activationsL2[db0i,:,neuron]
                    acts -= min(acts)
                    acts = acts/(np.max(np.abs(acts))+0.000001)
                    plt.title("ILD Tuning Neuron {} L2".format(neuron))
                    plt.plot(ild_range,acts,['o-','v-','s-'][db0i],c="black")
                    plt.xlabel("ILD (dB SPL)")
                    plt.ylabel("Normalized Activity")
                    plt.ylim(-0.1,1.1)
                plt.legend([str(db0s[0])+"dB SPL",str(db0s[1])+"dB SPL",str(db0s[2])+"dB SPL"])
                plt.savefig(save_name.format("L2 N{}".format(neuron)))
                plt.clf()

        for neuron in range(len(activationsL1[0][0][:])):
            for db0i in range(len(db0s)):
                acts = activationsL1[db0i,:,neuron]
                acts = acts/(np.max(np.abs(acts))+0.000001)
                acts -= min(acts)
                plt.title("ILD Tuning Neuron {} L1".format(neuron))
                plt.plot(ild_range,acts,['o-','v-','s-'][db0i],c="black")
                plt.xlabel("ILD (dB SPL)")
                plt.ylabel("Normalized Activity")
                plt.ylim(-0.1,1.1)
            plt.legend([str(db0s[0])+"dB SPL",str(db0s[1])+"dB SPL",str(db0s[2])+"dB SPL"])
            plt.savefig(save_name.format("L1 N{}".format(neuron)))
            plt.clf()



    def plotFrequencytuning(self):
        cut_weight_idx = int(self.inputsize/2)

        l_weights = []
        r_weights = []
        weight_sum = []

        freq_bins = [int(a) for a in self.testset.getFreqBins()]

        if self.Split:
            for lneuron in self.model.linearL1.weight:
                l_weights.append(lneuron.detach().cpu().numpy())
            for rneuron in self.model.linearR1.weight:
                r_weights.append(rneuron.detach().cpu().numpy())
            #for i in range(len(l_weights)):
                #weight_sum.append(l_weights[i]+r_weights[i])
            for i in range(len(l_weights)):
                plt.semilogx(freq_bins,(l_weights[i]),c="blue")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Weight Strength")
                plt.title("Frequency Tuning Neuron {}L".format(i))
                plt.savefig("plots\\weights\\lateFusion\\L1L N{}.png".format(i))
                plt.clf()

                plt.semilogx(freq_bins,(r_weights[i]),c="red")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Weight Strength")
                plt.title("Frequency Tuning Neuron {}R".format(i))
                plt.savefig("plots\\weights\\lateFusion\\L1R N{}.png".format(i))
                plt.clf()

        else:
            for hneuron in self.model.linear1.weight:
                l_weight = hneuron.detach().cpu().numpy()[:cut_weight_idx]
                r_weight = hneuron.detach().cpu().numpy()[cut_weight_idx:]
                #weight_sum.append(r_weight+l_weight)
                l_weights.append(l_weight)
                r_weights.append(r_weight)

            for i in range(len(l_weights)):
                #plt.semilogx(freq_bins,np.array(l_weights[i])+np.array(r_weights[i]),c="black")
                plt.semilogx(freq_bins,(l_weights[i]),c="blue")
                plt.semilogx(freq_bins,(r_weights[i]),c="red")
                plt.legend(["Left Ear","Right Ear"])
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Weight Strength")
                plt.title("Frequency Tuning Neuron {}".format(i))
                if self.L3:
                    plt.savefig("plots\\weights\\earlyFusion\\L1 N{}.png".format(i))
                else:
                    plt.savefig("plots\\weights\\h{}\\L1 N{}.png".format(self.hiddensize,i))
                plt.clf()

        if self.L3:
            l2_weights_l = []
            l2_weights_r = []
            for i,w2 in enumerate(self.model.linear2.weight):
                weights_l = []
                weights_r = []
                weighing = w2.detach().cpu().numpy()
                print(weighing)
                if self.Split:
                    for j,w1 in enumerate(self.model.linearL1.weight):
                        weights_l.append(w1.detach().cpu().numpy()*weighing[j])
                    for j,w1 in enumerate(self.model.linearR1.weight):
                        weights_r.append(w1.detach().cpu().numpy()*weighing[j])
                else:
                    for j,w1 in enumerate(self.model.linear1.weight):
                        weights_l.append(w1.detach().cpu().numpy()[:cut_weight_idx]*weighing[j])
                        weights_r.append(w1.detach().cpu().numpy()[cut_weight_idx:]*weighing[j])


                plt.semilogx(freq_bins,np.sum(weights_l,axis=0),c="blue")
                plt.semilogx(freq_bins,np.sum(weights_r,axis=0),c="red")
                plt.legend(["Left Ear","Right Ear"])
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Weight Strength")
                plt.title("Frequency Tuning Neuron {}".format(i))
                if self.Split:
                    plt.savefig("plots\\weights\\lateFusion\\L2 N{}.png".format(i))
                else:
                    plt.savefig("plots\\weights\\earlyFusion\\L2 N{}.png".format(i))
                plt.clf()

    def weightAnalysisLayer1(self):
        cut_weight_idx = int(self.inputsize/2)

        l_weights = []
        r_weights = []
        diffs = []

        if self.Split:
            for lneuron in self.model.linearL1.weight:
                l_weights.append(lneuron.detach().cpu().numpy())
            for rneuron in self.model.linearR1.weight:
                r_weights.append(rneuron.detach().cpu().numpy())
            for i in range(len(l_weights)):
                diffs.append(l_weights[i]-r_weights[i])
        else:
            for hneuron in self.model.linear1.weight:
                l_weight = hneuron.detach().cpu().numpy()[:cut_weight_idx]
                r_weight = hneuron.detach().cpu().numpy()[cut_weight_idx:]
                diffs.append(r_weight-l_weight)
                l_weights.append(l_weight)
                r_weights.append(r_weight)

            l_weights = [x for _,x in sorted(zip(self.hiddenorder,l_weights),key=lambda x: x[0])]
            r_weights = [x for _,x in sorted(zip(self.hiddenorder,r_weights),key=lambda x: x[0])]
            diffs = [x for _,x in sorted(zip(self.hiddenorder,diffs),key=lambda x: x[0])]

        freq_bins = [int(a) for a in self.testset.getFreqBins()]

        r_border = max([abs(np.max(r_weights)),abs(np.min(r_weights))])
        l_border = max([abs(np.max(l_weights)),abs(np.min(l_weights))])
        d_border = max([abs(np.max(diffs)),abs(np.min(diffs))])

        ax = plt.gca(label="weights1")
        im = ax.imshow(np.array(r_weights),cmap='bwr',aspect='auto',vmin=-r_border, vmax=r_border)
        cbar = ax.figure.colorbar(im,ax=ax)
        cbar.ax.set_ylabel("Val",rotation=-90,va="bottom")
        ax.set_xticks(np.arange(len(freq_bins))[::50])
        ax.set_xticklabels(freq_bins[::50])
        ax.tick_params(axis='x',labelrotation=90)
        ax.set_yticks(np.arange(self.hidden_size_L1))
        ax.set_yticklabels(self.yticklabels)
        plt.title("Weights Right Side")
        plt.xlabel("Frequency")
        plt.ylabel("Neuron")
        plt.savefig("plots\\weights\\L1 rweights {}.png".format(self.hidden_size_L1))
        plt.clf()

        ax = plt.gca(label="weights2")
        im = ax.imshow(np.array(l_weights),cmap='bwr',aspect='auto',vmin=-l_border, vmax=l_border)
        cbar = ax.figure.colorbar(im,ax=ax)
        cbar.ax.set_ylabel("Val",rotation=-90,va="bottom")
        ax.set_xticks(np.arange(len(freq_bins))[::50])
        ax.set_xticklabels(freq_bins[::50])
        ax.tick_params(axis='x',labelrotation=90)
        ax.set_yticks(np.arange(self.hidden_size_L1))
        ax.set_yticklabels(self.yticklabels)
        plt.title("Weights Left Side")
        plt.xlabel("Frequency")
        plt.ylabel("Neuron")
        plt.savefig("plots\\weights\\L1 lweights {}.png".format(self.hidden_size_L1))
        plt.clf()

        ax = plt.gca(label="weights3")
        im = ax.imshow(np.array(diffs),cmap='bwr',aspect='auto',vmin=-d_border, vmax=d_border)
        cbar = ax.figure.colorbar(im,ax=ax)
        cbar.ax.set_ylabel("Val",rotation=-90,va="bottom")
        ax.set_xticks(np.arange(len(freq_bins))[::50])
        ax.set_xticklabels(freq_bins[::50])
        ax.tick_params(axis='x',labelrotation=90)
        ax.set_yticks(np.arange(self.hidden_size_L1))
        ax.set_yticklabels(self.yticklabels)
        plt.title("Difference R-L")
        plt.xlabel("Frequency")
        plt.ylabel("Neuron")
        plt.savefig("plots\\weights\\L1 diff {}.png".format(self.hiddensize))
        plt.clf()


    def weightAnalysisLayer2(self):
        weights = []

        for hneuron in self.model.linear2.weight:
            weights.append(hneuron.detach().cpu().numpy())

        w_border = max([abs(np.max(weights)),abs(np.min(weights))])

        ax = plt.gca(label="weights3")
        im = ax.imshow(np.array(weights),cmap='bwr',aspect='auto',vmin=-w_border, vmax=w_border)
        cbar = ax.figure.colorbar(im,ax=ax)
        cbar.ax.set_ylabel("Val",rotation=-90,va="bottom")
        #ax.set_xticks(np.arange(len(freq_bins))[::50])
        #ax.set_xticklabels(freq_bins[::50])
        ax.tick_params(axis='x',labelrotation=90)
        ax.set_yticks(np.arange(self.hidden_size_L2))
        ax.set_yticklabels(self.yticklabels)
        plt.title("Weight Matrix")
        plt.xlabel("Hidden Neuron L1")
        plt.ylabel("Hidden Neuron L2")
        plt.savefig("plots\\weights\\L2 weights {}.png".format(self.hidden_size_L2))
        plt.clf()
        print(self.yticklabels)


    def freqsweepLayer(self,h_layer=0):
        if not self.L3 and not self.Split:
            outdir = "plots/freqsweep/h{}/".format(self.hiddensize)
        elif self.L3 and not self.Split:
            outdir = "plots/freqsweep/earlyFusion/l{}/".format(h_layer)
        elif self.L3 and self.Split:
            outdir = "plots/freqsweep/lateFusion/l{}/".format(h_layer)

        DTFdict = io.loadmat("OlHead-DTFs.mat")
        print(DTFdict)
        freqs = self.testset.getFreqBins()
        dtffreqs = DTFdict['frequency'][0]


        plotdtf_s = []
        dtfsel = -90
        dtfidx = np.where(DTFdict['labels'][0]==dtfsel)
        plotdtf = DTFdict['right'][dtfidx] - DTFdict['left'][dtfidx]
        plotdtf = (plotdtf.flatten()/max(abs(plotdtf.flatten()))) # normalize
        plotdtf_ = plotdtf/2
        plotdtf_s.append(plotdtf_)

        """
        #earlier code to follow the envelope
        plotdtf_ = (np.diff(plotdtf)/0.01)
        plotdtf_ = np.insert(plotdtf_,-1,0)
        plotdtf_s.append(plotdtf_)
        """

        print(freqs)
        print(dtffreqs)
        print(plotdtf_.shape)

        inL = np.identity(1015,dtype='f')*5.0
        inR = np.identity(1015,dtype='f')*5.0

        inL = gaussian_filter1d(inL,1)
        inR = gaussian_filter1d(inR,1)

        outputs = []
        activations = []

        for i in range(len(inL)):
            inNet = np.concatenate((inL[i],inR[i])).reshape(1,-1)
            x = Variable(torch.from_numpy(inNet)).cuda()

            outputs.append(self.model(x))

            if not self.L3 and not self.Split:
                activations.append(self.model.getHiddenActivations().flatten())
            else:
                activations.append(self.model.getHiddenActivations(h_layer).flatten())

        plt.title("Location per frequency bin")
        plt.xlabel("Frequency")
        plt.ylabel("Estimated Location")
        plt.axhline(0.0,0.0,12000,c='black')
        plt.semilogx(freqs,np.array(outputs)*180)
        #plt.semilogx(freqs,plotdtf)
        plt.savefig(outdir+"output.png")
        plt.clf()

        activations = np.array(activations).T

        for i,a in enumerate(activations):
            plt.title("Neuron {}".format(i))
            plt.axhline(0.0,0.0,12000,c='black')

            if h_layer is 1 or h_layer is 2:
                plt.semilogx(freqs,a-0.5,c='black')#-0.5)
            else:
                plt.semilogx(freqs,a-0.5,c='black')
            #plt.semilogx(freqs,plotdtf)
            #plt.semilogx(dtffreqs,plotdtf_)
            plt.gca().invert_yaxis()
            #plt.legend(['Neuron'])
            plt.xlabel("Frequency")
            plt.ylabel("Activation")
            plt.savefig(outdir+"n{}.png".format(i))
            plt.clf()

    def freqsweepLayerBestDTFFit(self,h_layer=0):
        from scipy.ndimage import gaussian_filter1d
        import scipy.io as io

        if not self.L3 and not self.Split:
            outdir = "plots/freqsweep/h{}/".format(self.hiddensize)
        elif self.L3 and not self.Split:
            outdir = "plots/freqsweep/earlyFusion/l{}/".format(h_layer)
        elif self.L3 and self.Split:
            outdir = "plots/freqsweep/lateFusion/l{}/".format(h_layer)

        inL = np.identity(1015,dtype='f')*5.0
        inR = np.identity(1015,dtype='f')*5.0

        inL = gaussian_filter1d(inL,1)
        inR = gaussian_filter1d(inR,1)

        outputs = []
        activations = []

        for i in range(len(inL)):
            inNet = np.concatenate((inL[i],inR[i])).reshape(1,-1)
            x = Variable(torch.from_numpy(inNet)).cuda()

            outputs.append(self.model(x))

            if not self.L3 and not self.Split:
                activations.append(self.model.getHiddenActivations().flatten())
            else:
                activations.append(self.model.getHiddenActivations(h_layer).flatten())

        DTFdict = io.loadmat("OlHead-DTFs.mat")
        print(DTFdict)
        freqs = self.testset.getFreqBins()
        dtffreqs = DTFdict['frequency'][0]


        actDTFdiff = []
        plotdtf_s = []
        for dtfidx in range(len(DTFdict['labels'][0])):
            #dtfsel = -90
            #dtfidx = np.where(DTFdict['labels'][0]==dtfsel)
            #plotdtf = DTFdict['right'][dtfidx] - DTFdict['left'][dtfidx]
            plotdtf = DTFdict['left'][dtfidx] - DTFdict['right'][dtfidx]
            plotdtf = (plotdtf.flatten()/max(abs(plotdtf.flatten()))) # normalize
            plotdtf_ = plotdtf/2
            plotdtf_s.append(plotdtf_)

            """
            #earlier code to follow the envelope
            plotdtf_ = (np.diff(plotdtf)/0.01)
            plotdtf_ = np.insert(plotdtf_,-1,0)
            plotdtf_s.append(plotdtf_)
            """


            diff = []
            for a in np.array(activations).T:
                corr = np.corrcoef(a,plotdtf_)
                #print(corr)
                diff.append(corr[0][1])
                #diff.append(sum((a-plotdtf_)**2))
            actDTFdiff.append(diff)

        minNeuronVal = np.max(actDTFdiff,axis=0)
        print("Mean Neuron DTF correlation: {}".format(np.mean(minNeuronVal)))

        plt.title("Location per frequency bin")
        plt.xlabel("Frequency")
        plt.ylabel("Estimated Location")
        plt.axhline(0.0,0.0,12000,c='black')
        plt.semilogx(freqs,np.array(outputs)*180)
        #plt.semilogx(freqs,plotdtf)
        plt.savefig(outdir+"output.png")
        plt.clf()

        activations = np.array(activations).T

        for i,a in enumerate(activations):
            dtf_idx_plot = np.array(actDTFdiff)[:,i].tolist().index(minNeuronVal[i])
            plt.title("Neuron {} Corr {:.3f}".format(i,minNeuronVal[i]))

            #Only if layer activation function is LeakyRelU
            #if False:
            if h_layer is 1 or h_layer is 2:
                plt.semilogx(freqs,a,c='black')#-0.5)
            else:
                plt.semilogx(freqs,a-0.5,c='black')
            #plt.semilogx(freqs,plotdtf)
            plt.semilogx(dtffreqs,plotdtf_s[dtf_idx_plot],c='orange')
            plt.legend(['Neuron','DTF az: {} ele: {}'.format(DTFdict['labels'][0][dtf_idx_plot],DTFdict['labels_ele'][0][dtf_idx_plot])])
            #plt.legend(['Neuron','DTF az: {}'.format(DTFdict['labels'][0][dtf_idx_plot])])
            plt.xlabel("Frequency")
            plt.ylabel("Activation/Normalized ILD")
            plt.axhline(0.0,0.0,12000,c='grey')
            plt.gca().invert_yaxis()
            plt.savefig(outdir+"n{}.png".format(i))
            plt.clf()

    def normalize(self,vec):
        return vec/np.max(np.abs(vec))

    def getScaling(self,val,scaling="DEG"):
        if self.network_labeltype is "ONE" and scaling is "DEG":
            return self.one_to_deg(val)
        if self.network_labeltype is "ONE" and scaling is "RAD":
            return self.one_to_rad(val)
        if self.network_labeltype is "DEG" and scaling is "ONE":
            return self.deg_to_one(val)
        if self.network_labeltype is "DEG" and scaling is "RAD":
            return self.deg_to_rad(val)
        if self.network_labeltype is "RAD" and scaling is "ONE":
            return self.rad_to_one(val)
        if self.network_labeltype is "RAD" and scaling is "DEG":
            return self.rad_to_deg(val)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

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
