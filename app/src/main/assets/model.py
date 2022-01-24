import torch
import torch.nn as nn



class NeuralNetwork(nn.Module):

  def __init__(self):
    super(NeuralNetwork,self).__init__()
    self.l1 = nn.Linear(1,1)
    self.l2 = nn.Linear(1,1)

  def forward(self,x):
    x = self.l1(x)
    x = self.l2(x)


    return x




