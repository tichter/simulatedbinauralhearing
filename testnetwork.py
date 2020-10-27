import sys
sys.path.insert(0,"D:\\1_Uni\\0_Master\\5_CI-Thesis\\01Deliverable\\Code\\utils")
from testutils import testNetwork
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

# Test params
test_set_size = 10
db_levels = [25,30,45,60,75,90,100]
db_levels_hid_tn = [25]
bandpasstest = False

# Network params
Layers3 = True
Split = True
h_layer = 1 #to analyze hidden layer if multiple are available if L3 is true

inputsize = 2030
hidden_size = 20

hidden_size_L1 = 40
hidden_size_L2 = 4

add_ele = True
doublepolar = True
network_labeltype = "ONE"

# Init variables
if Layers3:
    if Split:
        if add_ele:
            modelpath = "models/L3SplitbinauralNN-ELE-Rand-{}-{}-{}-{}.pth".format(inputsize,hidden_size_L1,hidden_size_L2,doublepolar)
        else:
            modelpath = "models/L3SplitbinauralNN-Rand-{}-{}-{}-{}.pth".format(inputsize,hidden_size_L1,hidden_size_L2,doublepolar)
    else:
        if add_ele:
            modelpath = "models/L3binauralNN-ELE-Rand-{}-{}-{}-{}.pth".format(inputsize,hidden_size_L1,hidden_size_L2,doublepolar)
        else:
            modelpath = "models/L3binauralNN-Rand-{}-{}-{}-{}.pth".format(inputsize,hidden_size_L1,hidden_size_L2,doublepolar)
else:
    if add_ele:
        modelpath = "models/binauralNN-ELE-Rand-{}-{}-{}.pth".format(inputsize,hidden_size,doublepolar)
    else:
        modelpath = "models/binauralNN-Rand-{}-{}-{}.pth".format(inputsize,hidden_size,doublepolar)

tn = testNetwork(modelpath=modelpath,testsize=test_set_size,inputsize=inputsize,hiddensize=hidden_size,hidden_size_L1=hidden_size_L1,hidden_size_L2=hidden_size_L2,db_levels=db_levels,add_ele=add_ele,doublepolar=doublepolar,L3=Layers3,Split=Split)
hid_tn = testNetwork(modelpath=modelpath,testsize=1,inputsize=inputsize,hiddensize=hidden_size,hidden_size_L1=hidden_size_L1,hidden_size_L2=hidden_size_L2,db_levels=db_levels_hid_tn,add_ele=False,doublepolar=doublepolar,L3=Layers3,Split=Split)

#hid_tn.freqsweepLayerBestDTFFit(h_layer)
#hid_tn.calcHiddenActivations(h_layer)

if bandpasstest:
    if Layers3:
        tn.bandpassname = "Early Fusion"
    if Split:
        tn.bandpassname = "Late Fusion"
    tn.testBroadBand()
    tn.testLowPass()
    tn.testHighPass()


"""
hid_tn.freqsweepLayerBestDTFFit(1)
if Layers3:
    hid_tn.freqsweepLayerBestDTFFit(2)
    if Split:
        hid_tn.freqsweepLayerBestDTFFit(3)
"""

hid_tn.freqsweepLayerBestDTFFit(1)
if Layers3:
    hid_tn.freqsweepLayerBestDTFFit(2)
    if Split:
        hid_tn.freqsweepLayerBestDTFFit(3)


hid_tn.freqsweepLayer(h_layer=1)
if Layers3:
    hid_tn.freqsweepLayer(h_layer=2)
    if Split:
        hid_tn.freqsweepLayer(h_layer=3)

hid_tn.weightAnalysisLayer2()

"""
if bandpasstest:
    tn.testBroadBand()
    tn.testBroadBand(dbsplit=True)
    tn.testLowPass()
    tn.testLowPass(dbsplit=True)
    tn.testHighPass()
    tn.testHighPass(dbsplit=True)

if Layers3:
    hid_tn.calcHiddenActivations(h_layer=1)
    hid_tn.weightAnalysisLayer1()

    hid_tn.calcHiddenActivations(h_layer=2)
    hid_tn.weightAnalysisLayer2()

    if Split:
        hid_tn.calcHiddenActivations(h_layer=3)

else:
    hid_tn.calcHiddenActivations(h_layer=1)
    hid_tn.weightAnalysisLayer1()

"""












#end
