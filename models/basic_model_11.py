# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:27:31 2019

@author: Ronglai Zuo
"""
#Implement the network's architechture 


import random
import torch as t
from torch import nn
from torch.nn import functional as F

k_size = 3  #kernel_size 3*3*3
depth = 11


class ResModule(nn.Module):
#each residual module, the FFN has 9 resModules in total  
    def __init__(self, inchannel, outchannel):
        super(ResModule, self).__init__()
        self.conv_part = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv3d(inchannel, outchannel, k_size, padding = k_size//2),
                nn.Conv3d(inchannel, outchannel, k_size, padding = k_size//2),
                )
        
    def forward(self, x):
        out = self.conv_part(x)
        out += x
        return out
        
    
class FFN(nn.Module):
    
    def __init__(self):
        super(FFN, self).__init__()
        self.net = self.make_entire_network()    
    
    def make_entire_network(self):
        modules=[]
        modules.append(nn.Conv3d(2, 32, k_size, padding = k_size//2))
        modules.append(nn.Conv3d(32, 32, k_size, padding = k_size//2))
        
        for i in range(depth):
            modules.append(ResModule(32, 32))
        
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv3d(32, 1, 1))
        return nn.Sequential(*modules)
    
    def forward(self, x):
        logits = self.net(x)
        return logits


class pre_FFN(nn.Module):
    
    def __init__(self):
        super(pre_FFN, self).__init__()
        self.net = self.make_entire_network()    
    
    def make_entire_network(self):
        modules=[]
        modules.append(nn.Conv3d(2, 32, k_size, padding = k_size//2))
        modules.append(nn.Conv3d(32, 32, k_size, padding = k_size//2))
        
        for i in range(depth):
            modules.append(ResModule(32, 32))

        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv3d(32, 3, 1))
        return nn.Sequential(*modules)
    
    def forward(self, x):
        bd = self.net(x)
        return bd
    

class FFN_share(nn.Module):
    
    def __init__(self):
        super(FFN_share, self).__init__()
        self.net = self.make_entire_network()    
    
    def make_entire_network(self):
        modules=[]
        modules.append(nn.Conv3d(2, 32, k_size, padding = k_size//2))
        modules.append(nn.Conv3d(32, 32, k_size, padding = k_size//2))
        
        for i in range(depth):
            modules.append(ResModule(32, 32))
        
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv3d(32, 1, 1))
        return nn.Sequential(*modules)
    
    def forward(self, x):
        logits = self.net(x)
        return logits
    
    
def fov_shifts(deltas = [8,8,8]):
    
    shifts = []
    for dx in (-deltas[0], 0, deltas[0]):
        for dy in (-deltas[1], 0, deltas[1]):
            for dz in (-deltas[2], 0, deltas[2]):
                if dx==0 and dy==0 and dz==0:
                    continue
                shifts.append((dz, dy, dx))
    
    random.shuffle(shifts)
    return shifts


class setup_loss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, labels):
        
        return F.binary_cross_entropy_with_logits(outputs, labels)

'''
model = pre_FFN()
print(model.net[9].conv_part[2])
'''


