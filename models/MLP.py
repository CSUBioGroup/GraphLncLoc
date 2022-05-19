from torch import nn as nn

class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.2, actFunc=nn.ReLU):
        super(MLP, self).__init__()
        layers = nn.Sequential()
        for i,os in enumerate(hiddenList):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), actFunc())
            inSize = os
        self.hiddenLayers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(inSize, outSize)
    def forward(self, x):
        x = self.hiddenLayers(x)
        return self.out(self.dropout(x))