import sys
sys.path.insert(0,"D:\\1_Uni\\0_Master\\5_CI-Thesis\\01Deliverable\\Code\\utils")
from dataset import DataSet
from networks import BinauralNeuralNetwork
from networks import L3BinauralNeuralNetwork
from networks import L3SplitBinauralNeuralNetwork

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

train_size = 2000 # number of generated noise samples
EPOCHS = 400
BATCH_SIZE = 100
add_ele = False
randmanip = True # should stay true
doublepolar = True
Layers3 = False
Split = False

#input_size = 4098 # of the neural network
hidden_size = 2 # of the neural network
hidden_size_L1 = 40
hidden_size_L2 = 4

learning_rate = 1e-4 #1e-4 if dB scaled 5e-7 if not dB rescaled

device = [torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')][0] # check whether cuda is available
print("Torch is using {}.".format(device))

trainset = DataSet(train_size,db_levels=[40,45,50,55,60,65,70],add_ele=add_ele,randmanip=randmanip,doublepolar=doublepolar) # db_levels as Ausili
print("Trainset length: {}".format(len(trainset)))

freqs = trainset.getFreqBins()
input_size = 2*len(freqs)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)

if Layers3:
    if Split:
        model = L3SplitBinauralNeuralNetwork(input_size=input_size,hidden_size_L1=hidden_size_L1,hidden_size_L2=hidden_size_L2).to(device)
    else:
        model = L3BinauralNeuralNetwork(input_size=input_size,hidden_size_L1=hidden_size_L1,hidden_size_L2=hidden_size_L2).to(device)
else:
    model = BinauralNeuralNetwork(input_size=input_size,hidden_size=hidden_size).to(device)

model.float()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

lossfun = nn.MSELoss()

losses = []

for epoch in tqdm(range(EPOCHS)):
    total_loss = 0
    true_angle_train = []
    pred_angle_train = []
    for trainsteps,data in enumerate(trainloader):
        x = Variable(data[0]).to(device)
        y = Variable(data[1]).to(device)

        output = model(x)
        loss = lossfun(torch.flatten(output),y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data

        for t_angle in data[1]:
            true_angle_train.append(t_angle)
        for p_angle in output.detach().cpu():
            pred_angle_train.append(p_angle)

    losses.append((total_loss/(trainsteps+1))/BATCH_SIZE)
    print("Loss: {}".format(losses[-1]))


    plt.plot(np.array(true_angle_train)*180,np.array(pred_angle_train)*180,'bx')
    plt.xlabel("Stimulus Location (deg)")
    plt.ylabel("Response Location (deg)")
    plt.title("Training E {}".format(epoch))
    plt.savefig("plots\\trainperepoch\\{}.png".format(epoch))
    plt.clf()
if Layers3:
    if Split:
        if add_ele:
            torch.save(model.state_dict(), "models/L3SplitbinauralNN-ELE-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar))
        else:
            torch.save(model.state_dict(), "models/L3SplitbinauralNN-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar))
    else:
        if add_ele:
            torch.save(model.state_dict(), "models/L3binauralNN-ELE-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar))
        else:
            torch.save(model.state_dict(), "models/L3binauralNN-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar))

else:
    if add_ele:
        torch.save(model.state_dict(), "models/binauralNN-ELE-Rand-{}-{}-{}.pth".format(input_size,hidden_size,doublepolar))
    else:
        torch.save(model.state_dict(), "models/binauralNN-Rand-{}-{}-{}.pth".format(input_size,hidden_size,doublepolar))
