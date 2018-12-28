from __future__ import print_function
import sys
import torch
from torch import nn 
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
torch.manual_seed(1)    # reproducible
np.random.seed(1)

n = 200
d0 = 100
d = 10
LR = 1e-2

class AA(nn.Module):
    def __init__(self):
        super(AA,self).__init__()
        self.fc = nn.Linear(d0,d)
        self.Dictionary = self.fc.weight.transpose(1,0)
        assert self.Dictionary.requires_grad == True 

    def forward(self, X):
        DTD = torch.matmul(self.Dictionary.transpose(1,0), self.Dictionary)   #### D is Dictionary;  D^T D  ## R^{d,d}
        DTD_inv = torch.inverse(DTD)        ### (D^T D)^{-1}  ### R^{d,d}
        DTD_inv_DT = torch.matmul(DTD_inv, self.Dictionary.transpose(1,0))   ###  (D^T D)^{-1} D^T ## R^{d,d0} 10,100
        assert DTD_inv_DT.requires_grad == True
        X_o = X.matmul(DTD_inv_DT.transpose(1,0))
        X_o = torch.clamp(X_o, min = 0)
        assert X_o.requires_grad == True 
        return X_o         

    @property
    def check_dictionary(self):
        #return self.Dictionary[1:4,2:5], self.fc.weight[1:4,2:5]
        return self.fc.weight[1:4,2:5]

aa = AA()
#print(aa.parameters())


opt_ = torch.optim.SGD(aa.parameters(), lr=LR)

X = Variable(torch.rand(n,d0))
for i in range(10):
    output = aa(X)
    loss = output.norm()
    opt_.zero_grad()
    loss.backward()
    opt_.step()
    loss_value = loss.data[0]
    #print(loss_value)
    print(aa.check_dictionary)



