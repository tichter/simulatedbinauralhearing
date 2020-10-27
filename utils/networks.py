import torch
import torch.nn as nn
import torch.nn.functional as F

class BinauralNeuralNetwork(nn.Module):
    """
    Adepted from Ausili PhD thesis.
    """
    def __init__(self,input_size,hidden_size):
        super(BinauralNeuralNetwork,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,1)
        self.act = nn.Sigmoid()
        #self.act = nn.ReLU()

        self.linearact = None

    def forward(self,x):
        x = self.act(self.linear1(x))
        self.linearact = x
        x = self.linear2(x)
        return x

    def getAvgHiddenActivations(self):
        return torch.mean(self.linearact).detach().cpu().numpy()

    def getHiddenActivations(self):
        return self.linearact.detach().cpu().numpy()

class L3BinauralNeuralNetwork(nn.Module):
    """
    Three layer architecture to split up frequency extraction from ILD comparison.
    """
    def __init__(self,input_size,hidden_size_L1,hidden_size_L2):
        super(L3BinauralNeuralNetwork,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size_L1)
        self.linear2 = nn.Linear(hidden_size_L1,hidden_size_L2)
        self.linear3 = nn.Linear(hidden_size_L2,1)
        self.act = nn.Sigmoid()
        self.actR = nn.ReLU()
        self.actLR = nn.LeakyReLU() #TEMP Change to LeakyReLU


        self.linearact1 = None
        self.linearact2 = None

    def forward(self,x):
        x = self.actLR(self.linear1(x))
        self.linearact1 = x
        x = self.act(self.linear2(x))
        self.linearact2 = x
        x = self.linear3(x)
        return x

    def getAvgHiddenActivations(self,layer):
        if layer is 1:
            return torch.mean(self.linearact1).detach().cpu().numpy()
        elif layer is 2:
            return torch.mean(self.linearact2).detach().cpu().numpy()

    def getHiddenActivations(self,layer):
        if layer is 1:
            return self.linearact1.detach().cpu().numpy()
        elif layer is 2:
            return self.linearact2.detach().cpu().numpy()

class L3SplitBinauralNeuralNetwork(nn.Module):
    """
    Three layer architecture with left right ear fusion after second layer.
    hidden_size_L1 total size across both ears
    """
    def __init__(self,input_size,hidden_size_L1,hidden_size_L2):
        super(L3SplitBinauralNeuralNetwork,self).__init__()

        self.input_size2 = int(input_size/2)
        self.hidden_size_L1_2 = int(hidden_size_L1/2)

        self.linearL1 = nn.Linear(self.input_size2,self.hidden_size_L1_2)
        self.linearR1 = nn.Linear(self.input_size2,self.hidden_size_L1_2)
        self.linear2 = nn.Linear(2*self.hidden_size_L1_2,hidden_size_L2)
        self.linear3 = nn.Linear(hidden_size_L2,1)

        self.act = nn.Sigmoid()
        self.actR = nn.ReLU()
        self.actLR = nn.LeakyReLU() #TEMP Change to LeakyReLU


        self.linearactL1 = None
        self.linearactR1 = None
        self.linearact2 = None

    def forward(self,x):
        xL = x[:,:self.input_size2]
        xR = x[:,self.input_size2:]

        xL = self.actLR(self.linearL1(xL))
        xR = self.actLR(self.linearR1(xR))
        self.linearactL1 = xL
        self.linearactR1 = xR

        x = torch.cat((xL,xR),1)
        #self.linearact1 = x

        x = self.act(self.linear2(x))
        self.linearact2 = x

        x = self.linear3(x)

        return x

    def getAvgHiddenActivations(self,layer):
        """
        1: L1
        2: R1
        3: 2
        """
        if layer is 1:
            return torch.mean(self.linearactL1).detach().cpu().numpy()
        elif layer is 2:
            return torch.mean(self.linearactR1).detach().cpu().numpy()
        elif layer is 3:
            return torch.mean(self.linearact2).detach().cpu().numpy()

    def getHiddenActivations(self,layer):
        """
        1: L1
        2: R1
        3: 2
        """
        if layer is 1:
            return self.linearactL1.detach().cpu().numpy()
        elif layer is 2:
            return self.linearactR1.detach().cpu().numpy()
        elif layer is 3:
            return self.linearact2.detach().cpu().numpy()
