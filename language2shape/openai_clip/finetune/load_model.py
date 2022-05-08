# from stylegan2networks import MappingNetwork
from mlp_networks import F_MappingNetwork, B_MappingNetwork

def fetch_model(args):
    if args.n_model == 'StyleGAN2':
        mp_model = MappingNetwork(512, 0, 512, None)
    elif args.n_model == 'MLPS':
        mp_model = F_MappingNetwork(hidden_features=args.h_features, norm=args.norm, activation=args.act_f, l2norm=args.l2norm, num_layers=args.num_layers)
    elif args.n_model == 'B_MLPS':
        mp_model = B_MappingNetwork(hidden_features=args.h_features, norm=args.norm, activation=args.act_f, l2norm=args.l2norm, num_layers=args.num_layers, lrelu=args.lrelu, logit_scale=args.logit_scale, orthogonal=args.orthogonal)
    return mp_model