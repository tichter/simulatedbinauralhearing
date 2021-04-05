import sys
sys.path.insert(0,"D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\utils")
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

train_size = 200000 # number of generated noise samples
validation_size = 10000

EPOCHS = 400
BATCH_SIZE = 100
add_ele = True # add HRTFs with non zero elevation

#randmanip = True # should stay true
doublepolar = True
Layers3 = True # True for 3 layer architetures
Split = True # True for Late fusion, False for early fusion, only if Layers3 = True

#input_size = 4098 # calculated from the freq bins
hidden_size = 20 # of the one hidden layer neural networks
hidden_size_L1 = 40 # of the first hidden layer of the two hidden layer neural networks
hidden_size_L2 = 4 # of the first hidden layer of the two hidden layer neural networks

learning_rate = 1e-4 #1e-4 if dB scaled 5e-7 if not dB rescaled

device = [torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')][0] # check whether cuda is available
print("Torch is using {}.".format(device))

print("Generating Train Set")
trainset = DataSet(name_add="train",nsounds=train_size,train=True,add_ele=add_ele,doublepolar=doublepolar)
print("Generating Validation Set")
valiset = DataSet(name_add="valid",nsounds=validation_size,train=True,add_ele=add_ele,doublepolar=doublepolar)

print("Trainset length: {}".format(len(trainset)))
print("Validationset length: {}".format(len(valiset)))

freqs = trainset.getFreqBins()
input_size = 2*len(freqs)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)
valiloader = torch.utils.data.DataLoader(valiset,batch_size=BATCH_SIZE,shuffle=True)

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

lossfun = nn.MSELoss(reduction="sum")

losses = []
valiloss = []
best_valiloss = 9999

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

    losses.append((total_loss/(trainsteps*BATCH_SIZE)))

    # Validation step
    total_valiloss = 0
    with torch.no_grad():
        for valisteps,data in enumerate(valiloader):
            x = Variable(data[0]).to(device)
            y = Variable(data[1]).to(device)

            output = model(x)
            loss = lossfun(torch.flatten(output),y)

            total_valiloss += loss.data

    valiloss.append(total_valiloss/(valisteps*BATCH_SIZE))

    print("\nLoss T: {} Loss V: {}".format(losses[-1],valiloss[-1]))

    plt.plot(np.array(true_angle_train)*180,np.array(pred_angle_train)*180,'bx')
    plt.xlabel("Stimulus Location (deg)")
    plt.ylabel("Response Location (deg)")
    plt.title("Training E {}".format(epoch))
    plt.savefig("plots\\trainperepoch\\{}.png".format(epoch))
    plt.clf()

    if epoch > 10 and valiloss[-1] < best_valiloss:
        print("Validation set saved E: {}".format(epoch))
        best_valiloss = valiloss[-1]

        if Layers3:
            if Split:
                if add_ele:
                    torch.save(model.state_dict(), "models/L3SplitbinauralNN-Valid-ELE-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar))
                else:
                    torch.save(model.state_dict(), "models/L3SplitbinauralNN-Valid-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar))
            else:
                if add_ele:
                    torch.save(model.state_dict(), "models/L3binauralNN-Valid-ELE-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar))
                else:
                    torch.save(model.state_dict(), "models/L3binauralNN-Valid-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar))

        else:
            if add_ele:
                torch.save(model.state_dict(), "models/binauralNN-Valid-ELE-Rand-{}-{}-{}.pth".format(input_size,hidden_size,doublepolar))
            else:
                torch.save(model.state_dict(), "models/binauralNN-Valid-Rand-{}-{}-{}.pth".format(input_size,hidden_size,doublepolar))




if Layers3:
    if Split:
        if add_ele:
            model_name = "models/L3SplitbinauralNN-Final-ELE-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar)
            loss_name = "models/losses/L3SplitbinauralNN-Final-ELE-Rand-{}-{}-{}-{}-{}.npy".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar,"{}")
            torch.save(model.state_dict(), model_name)
            np.save(loss_name.format("Ltrain"),losses)
            np.save(loss_name.format("Lvalid"),valiloss)
        else:
            model_name = "models/L3SplitbinauralNN-Final-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar)
            loss_name = "models/losses/L3SplitbinauralNN-Final-Rand-{}-{}-{}-{}-{}.npy".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar,"{}")
            torch.save(model.state_dict(), model_name)
            np.save(loss_name.format("Ltrain"),losses)
            np.save(loss_name.format("Lvalid"),valiloss)
    else:
        if add_ele:
            model_name = "models/L3binauralNN-Final-ELE-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar)
            loss_name = "models/losses/L3binauralNN-Final-ELE-Rand-{}-{}-{}-{}-{}.npy".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar,"{}")
            torch.save(model.state_dict(), model_name)
            np.save(loss_name.format("Ltrain"),losses)
            np.save(loss_name.format("Lvalid"),valiloss)
        else:
            model_name = "models/L3binauralNN-Final-Rand-{}-{}-{}-{}.pth".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar)
            loss_name = "models/losses/L3binauralNN-Final-Rand-{}-{}-{}-{}-{}.npy".format(input_size,hidden_size_L1,hidden_size_L2,doublepolar,"{}")
            torch.save(model.state_dict(), model_name)
            np.save(loss_name.format("Ltrain"),losses)
            np.save(loss_name.format("Lvalid"),valiloss)
else:
    if add_ele:
        model_name = "models/binauralNN-Final-ELE-Rand-{}-{}-{}.pth".format(input_size,hidden_size,doublepolar)
        loss_name = "models/losses/binauralNN-Final-ELE-Rand-{}-{}-{}-{}.npy".format(input_size,hidden_size,doublepolar,"{}")
        torch.save(model.state_dict(), model_name)
        np.save(loss_name.format("Ltrain"),losses)
        np.save(loss_name.format("Lvalid"),valiloss)
    else:
        model_name = "models/binauralNN-Final-Rand-{}-{}-{}.pth".format(input_size,hidden_size,doublepolar)
        loss_name = "models/losses/binauralNN-Final-Rand-{}-{}-{}-{}.npy".format(input_size,hidden_size,doublepolar,"{}")
        torch.save(model.state_dict(), model_name)
        np.save(loss_name.format("Ltrain"),losses)
        np.save(loss_name.format("Lvalid"),valiloss)
