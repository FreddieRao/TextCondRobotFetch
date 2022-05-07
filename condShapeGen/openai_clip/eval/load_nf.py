import torch
import torch.nn as nn
from normalizing_flows.maf import MADE, MADEMOG, MAF, MAFMOG, RealNVP
from pytorch_flows import flows as fnn
# from mapping_flows import flows as map_fnn # remove this line for gitpush
from openai_clip.finetune.stylegan2networks import MappingNetwork
from openai_clip.finetune.mlp_networks import F_MappingNetwork, B_MappingNetwork

def load_pytorch_flows(args, is_map=False):
    # model
    modules = []

    assert args.nf_model in ['maf', 'maf-split', 'maf-split-glow', 'realnvp', 'glow']
    if args.nf_model == 'glow':
        mask = torch.arange(0, args.input_size) % 2
        mask = mask.to(args.device).float()

        print("Warning: Results for GLOW are not as good as for MAF yet.")
        for _ in range(args.n_blocks):
            modules += [
                fnn.BatchNormFlow(args.input_size),
                fnn.LUInvertibleMM(args.input_size),
                fnn.CouplingLayer(
                    args.input_size, args.hidden_size, mask, args.cond_label_size,
                    s_act='tanh', t_act='relu')
            ]
            mask = 1 - mask
    elif args.nf_model == 'realnvp':
        input_size = args.input_size * 2 if is_map else args.input_size

        mask = torch.arange(0, input_size) % 2
        mask = mask.to(args.device).float()

        for _ in range(args.n_blocks):
            modules += [
                fnn.CouplingLayer(
                    input_size, args.hidden_size, mask, args.cond_label_size,
                    s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(input_size)
            ]
            mask = 1 - mask
    elif args.nf_model == 'maf':
        for _ in range(args.n_blocks):
            modules += [
                fnn.MADE(args.input_size, args.hidden_size, args.cond_label_size, act='relu'),
                fnn.BatchNormFlow(args.input_size),
                fnn.Reverse(args.input_size)
            ]
    elif args.nf_model == 'maf-split':
        for _ in range(args.n_blocks):
            modules += [
                fnn.MADESplit(args.input_size, args.hidden_size, args.cond_label_size,
                            s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(args.input_size),
                fnn.Reverse(args.input_size)
            ]
    elif args.nf_model == 'maf-split-glow':
        for _ in range(args.n_blocks):
            modules += [
                fnn.MADESplit(args.input_size, args.hidden_size, args.cond_label_size,
                            s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(args.input_size),
                fnn.InvertibleMM(args.input_size)
            ]
    else:
        raise ValueError('Unrecognized nf_model.')

    nf_model = fnn.FlowSequential(*modules)

    for module in nf_model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    if args.use_mlp == True:
        mlps = fnn.Feedforward(512, args.cond_label_size)
    else:
        mlps = nn.Identity()
    from pytorch_flows.flows import generate_conditioned
    return mlps, nf_model, generate_conditioned

def load_mapping_flows(args):
    # model
    modules = []

    assert args.nf_model in ['maf', 'maf-split', 'maf-split-glow', 'realnvp', 'glow']
    if args.nf_model == 'glow':
        mask = torch.arange(0, 512) % 2
        mask = mask.to(args.device).float()

        print("Warning: Results for GLOW are not as good as for MAF yet.")
        for _ in range(args.num_layers):
            modules += [
                map_fnn.BatchNormFlow(512),
                map_fnn.LUInvertibleMM(512),
                map_fnn.CouplingLayer(
                    512, args.h_features, mask, args.cond_label_size,
                    s_act='tanh', t_act='relu')
            ]
            mask = 1 - mask
    elif args.nf_model == 'realnvp':

        mask = torch.arange(0, 1024) % 2
        mask = mask.to(args.device).float()

        for _ in range(args.num_layers):
            modules += [
                map_fnn.CouplingLayer(
                    512, args.h_features, mask, args.cond_label_size,
                    s_act='tanh', t_act='relu'),
                map_fnn.BatchNormFlow(512)
            ]
            mask = 1 - mask
    elif args.nf_model == 'maf':
        for _ in range(args.num_layers):
            modules += [
                map_fnn.MADE(512, args.h_features, args.cond_label_size, act='relu'),
                map_fnn.BatchNormFlow(512),
                map_fnn.Reverse(512)
            ]
    elif args.nf_model == 'maf-split':
        for _ in range(args.num_layers):
            modules += [
                map_fnn.MADESplit(512, args.h_features, args.cond_label_size,
                            s_act='tanh', t_act='relu'),
                map_fnn.BatchNormFlow(512),
                map_fnn.Reverse(512)
            ]
    elif args.nf_model == 'maf-split-glow':
        for _ in range(args.num_layers):
            modules += [
                map_fnn.MADESplit(512, args.h_features, args.cond_label_size,
                            s_act='tanh', t_act='relu'),
                map_fnn.BatchNormFlow(512),
                map_fnn.InvertibleMM(512)
            ]
    else:
        raise ValueError('Unrecognized nf_model.')

    map_model = map_fnn.FlowSequential(*modules)

    for module in map_model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    if args.use_mlp == True:
        mlps = map_fnn.Feedforward(512, args.cond_label_size)
    else:
        mlps = nn.Identity()
    from mapping_flows.flows import generate_conditioned
    return mlps, map_model, generate_conditioned

def load_normalizing_flows(args):
    # model
    if args.nf_model == 'made':
        nf_model = MADE(args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                        args.activation_fn, args.input_order)
    elif args.nf_model == 'mademog':
        assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
        nf_model = MADEMOG(args.n_components, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                        args.activation_fn, args.input_order)
    elif args.nf_model == 'maf':
        nf_model = MAF(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                    args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm)
    elif args.nf_model == 'mafmog':
        assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
        nf_model = MAFMOG(args.n_blocks, args.n_components, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                    args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm)
    elif args.nf_model =='realnvp':
        nf_model = RealNVP(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                        batch_norm=not args.no_batch_norm)
    else:
        raise ValueError('Unrecognized nf_model.')

    from normalizing_flows.maf import generate_conditioned
    return None, nf_model, generate_conditioned

def load_latent_gan(args):
    img_shape = (256, )
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(100 + 512, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )

        def forward(self, noise, labels):
            # Concatenate label embedding and image to produce input
            gen_input = torch.cat((labels.view(labels.size(0), -1), noise), -1)
            img = self.model(gen_input)
            img = img.view(img.size(0), *img_shape)
            return img

    nf_model = Generator()
    from pytorch_gan.implementations.cgan.inference import generate_conditioned

    return None, nf_model, generate_conditioned

def load_nf_model(args):
    if args.nf_code_base == 'pytorch_flows':
        f = load_pytorch_flows
    elif args.nf_code_base == 'normalizing_flows':
        f = load_normalizing_flows
    elif args.nf_code_base == 'pytorch_gans':
        f = load_latent_gan
    
    return f(args)

def load_map_model(args):
    if args.finetune_clip:
        if args.map_code_base == 'stylegan':
            mp_model = MappingNetwork(512, 0, 512, None)
            mp_mlps = nn.Identity()
            generate_conditioned = None
        elif args.map_code_base == 'MLPS':
            mp_model = F_MappingNetwork(hidden_features=args.h_features, norm=args.norm, activation=args.act_f, l2norm=args.l2norm, num_layers=args.num_layers)
            mp_mlps = nn.Identity()
            generate_conditioned = None
        elif args.map_code_base == 'B_MLPS':
            mp_model = B_MappingNetwork(hidden_features=args.h_features, norm=args.norm, activation=args.act_f, l2norm=args.l2norm, num_layers=args.num_layers)
            mp_mlps = nn.Identity()
            generate_conditioned = None
        elif args.map_code_base == 'NF':
            mp_mlps, mp_model, generate_conditioned = load_mapping_flows(args)
    else:
        mp_model = nn.Identity()
        mp_mlps = nn.Identity()
        generate_conditioned = None
    return mp_mlps, mp_model, generate_conditioned