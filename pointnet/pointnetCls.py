# -*- coding: utf-8 -*-
"""
Created on Fri May  6 18:04:40 2022

@author: ThinkPad
"""
import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import custom_layers
import hyperlayers
import model
from torchmeta.modules import (MetaModule, MetaSequential)
from collections import OrderedDict



class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.latent_dim = 512
        self.feature_transform = feature_transform
        self.feat = model.PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.phi = PointnetHyper(outputDim = k)
       
        self.hyper_phi = hyperlayers.HyperNetwork(hyper_in_features=self.latent_dim,
                                                          hyper_hidden_layers=1,
                                                          hyper_hidden_features=self.latent_dim,
                                                          hypo_module=self.phi)
    def forward(self, input, z):
        x = input
        x, trans, trans_feat = self.feat(x)
        phi_weights = self.hyper_phi(z)
        phi = lambda i: self.phi(i, params=phi_weights)
        x = phi(x)
            
        return F.log_softmax(x, dim=1), trans, trans_feat
        
class PointnetHyper(MetaModule):
    def __init__(self,
                 outputDim = 1,
                 activation='relu',
                 nonlinearity='relu'):
        super().__init__()

        self.net = []
        self.net.append(custom_layers.FCLayer(in_features=1024, out_features=512, nonlinearity='relu', norm='layernorm'))
        self.net.append(custom_layers.FCLayer(in_features=512, out_features=256, nonlinearity='relu', norm='layernorm'))
        self.net.append(custom_layers.FCLayer(in_features=256, out_features=outputDim, nonlinearity=None, norm=None))

        self.net = MetaSequential(*self.net)
        self.net.apply(custom_layers.init_weights_normal)

    def forward(self, input, params=None):
        return self.net(input, params=self.get_subdict(params, 'net'))
    
    
    
if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    z = Variable(torch.rand(1, 384))

    cls = PointNetCls(k = 2)
    out, _, _ = cls(sim_data,z)
    print('class', out.size())
    out = out[0]
    # each element in target has to have 0 <= value < C
    target = np.zeros((32,))
    target = torch.from_numpy(target).to(torch.int64)
    print(out.shape)
    print(target.shape)
    output = F.nll_loss(out, target)
    # print(output.shape)
