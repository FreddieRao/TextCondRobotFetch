import torch
import numpy as np

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        norm            = 'None',     # Normalization Function: None, BN, LN
        activation      = 'None', # Activation function: 'relu', 'lrelu', etc.
        orthogonal           = False, 
    ):
        super(FullyConnectedLayer, self).__init__()
        fc = torch.nn.Linear(in_features, out_features, bias)
        
        self.fc = torch.nn.utils.parametrizations.orthogonal(fc) if orthogonal else fc

        if norm == 'None':
            self.norm = torch.nn.Identity()
        elif norm == 'BN':
            self.norm = torch.nn.BatchNorm1d(out_features)
        elif norm == 'LN':
            self.norm = torch.nn.LayerNorm(out_features)
        else:
            raise ValueError('{} is not defined.'.format(norm))

        if activation == 'None':
            self.activation_f = torch.nn.Identity()
        elif activation == 'relu':
            self.activation_f = torch.nn.ReLU()
        elif activation == 'lrelu':
            self.activation_f = torch.nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation_f = torch.nn.GELU()
        elif activation == 'tanh':
            self.activation_f = torch.nn.Tanh()
        elif activation == 'hswish':
            self.activation_f = torch.nn.Hardswish()
        else:
            raise ValueError('{} is not defined.'.format(activation))

    def forward(self, x):

        x = self.fc(x)
        x = self.norm(x)
        x = self.activation_f(x)
        
        return x

#----------------------------------------------------------------------------

class F_MappingNetwork(torch.nn.Module):
    def __init__(self,
        in_features     = 512,                # Number of input features.
        hidden_features = 512,               # Number of hidden features.
        out_features    = 512,               # Number of output feature
        bias            = True,     # Apply additive bias before the activation function?
        norm            = 'LN',     # Normalization Function: None, BN, LN
        activation      = 'tanh', # Activation function: 'relu', 'lrelu', etc.
        l2norm          = True,    # Use L2 Norm
        num_layers      = 1,        # Number of Network Layers
        logit_scale      = 1,        # logit_scale
        orthogonal      =False
    ):
        super(F_MappingNetwork, self).__init__()

        self.l2norm = l2norm

        features_list = [in_features] + [hidden_features] * (num_layers - 1) + [out_features]

        layers = []
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            if idx == num_layers - 1:
                if activation in ['relu', 'lrelu', 'gelu', 'hswish'] and not l2norm:
                    activation = 'None'
            layer = FullyConnectedLayer(in_features, out_features, bias=bias, norm=norm, activation=activation, orthogonal=orthogonal)
            layers.append(layer)
        
        

        self.layers = torch.nn.Sequential()
        for idx, l in enumerate(layers):
            self.layers.add_module('layer_'+str(idx), l)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / logit_scale))

    def forward(self, x, y):
        # y = normalize_2nd_moment(y.to(torch.float32))
        y = self.layers(y)
        if self.l2norm:
            y = torch.nn.functional.normalize(y)
        return x, y, self.logit_scale


#----------------------------------------------------------------------------

class B_MappingNetwork(torch.nn.Module):
    def __init__(self,
        in_features     = 512,                # Number of input features.
        hidden_features = 512,               # Number of hidden features.
        out_features    = 512,               # Number of output feature
        bias            = True,     # Apply additive bias before the activation function?
        norm            = 'LN',     # Normalization Function: None, BN, LN
        activation      = 'tanh', # Activation function: 'relu', 'lrelu', etc.
        l2norm          = True,    # Use L2 Norm
        num_layers      = 1,        # Number of Network Layers
        lrelu      = 1,        # LRELU Slope
        logit_scale      = 1,        # logit_scale
        orthogonal      =False
    ):
        super(B_MappingNetwork, self).__init__()

        self.l2norm = l2norm

        features_list = [in_features] + [hidden_features] * (num_layers - 1) + [out_features]

        t_layers = []
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            if idx == num_layers - 1:
                if activation in ['relu', 'lrelu', 'gelu', 'hswish'] and not l2norm:
                    activation = 'None'
            t_layer = FullyConnectedLayer(in_features, out_features, bias=bias, norm=norm, activation=activation, orthogonal=orthogonal)
            t_layers.append(t_layer)
        
        i_layers = []
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            if idx == num_layers - 1:
                if activation in ['relu', 'lrelu', 'gelu', 'hswish'] and not l2norm:
                    activation = 'None'
            i_layer = FullyConnectedLayer(in_features, out_features, bias=bias, norm=norm, activation=activation)
            i_layers.append(i_layer)        

        self.t_layers = torch.nn.Sequential()
        for idx, l in enumerate(t_layers):
            self.t_layers.add_module('t_layer_'+str(idx), l)

        self.i_layers = torch.nn.Sequential()
        for idx, l in enumerate(i_layers):
            self.i_layers.add_module('i_layer_'+str(idx), l)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / logit_scale))

    def forward(self, i, t):
        i = self.i_layers(i)
        t = self.t_layers(t)
        if self.l2norm:
            i = torch.nn.functional.normalize(i)
            t = torch.nn.functional.normalize(t)
        return i, t, self.logit_scale

if __name__ == '__main__':
    model = B_MappingNetwork()
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        total_params+=param
    print(total_params)