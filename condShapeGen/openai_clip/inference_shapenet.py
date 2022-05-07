import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import argparse
import shutil
import random
import clip
import trimesh
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

## relative import

from openai_clip.eval import *
from occupancy_networks.im2mesh import config

# made parameters
parser = argparse.ArgumentParser()
# Text Input
parser.add_argument('--cate', type=str, default='chair', help='Category name [chair, car, table]')
parser.add_argument('--from_txt', action='store_true', help='Inference from text input')
parser.add_argument('--no_from_txt', dest='from_txt', action='store_false')
parser.add_argument('--split_by_text', action='store_true', help='Split dataset with text; Otherwise shape id')
parser.add_argument('--split_by_id', dest='split_by_text', action='store_false')
parser.add_argument('--use_text2shape', default=False, action='store_true', help='Use Text2Shape for quantitative and qualitative measure')
parser.add_argument('--no_use_text2shape', dest='use_text2shape', action='store_false')
parser.add_argument('--clip_txt_feat_path', type=str, default='./', help='Path to Preprocessed CLIP Text Feature from Text2Shape')
parser.add_argument('--text', type=str, default='a chair', help='free-form text input, will be overwrite if use_text2shape is on')
parser.add_argument('--use_txt_prefixs', action='store_true', help='Use mean of Txt Prefixs')
parser.add_argument('--no_use_txt_prefixs', dest='use_txt_prefixs', action='store_false')
parser.add_argument('--valid_cate_anno_path', type=str, default='./', help='Path to annotations of ShapeNet models with a valid subcategory')

# Image Input
parser.add_argument('--n_from_img', type=int, default=1, help='Number of Inference instances from clip image feature.')
parser.add_argument('--img_gt_path', type=str, default='./', help='Path to GT Rendered ShapeNet Images')
parser.add_argument('--clip_img_feat_path', type=str, default='./', help='Path to Preprocessed CLIP Image Feature')

# Analysis And Visualization
parser.add_argument('--n_retrieval_clip', type=int, default=1, help='Number of retrieved shape of each clip feature.')
parser.add_argument('--n_retrieval_shape', type=int, default=1, help='Number of retrieved mesh of each shape feature.')
parser.add_argument('--quality', action='store_true', help='Qualititively analyse all the results')
parser.add_argument('--quantity', action='store_true', help='Quantitively analyse all the results')
parser.add_argument('--quant_clip', action='store_true', help='TODO: Quantitively analyse Clip Loss')
parser.add_argument('--vis_correct_only', action='store_true', default=False, help='Visualize gt img-shape and gen shape in shape embedding')
parser.add_argument('--no_vis_correct_only', dest='vis_correct_only', action='store_false')
parser.add_argument('--vis_img_embed', action='store_true', help='Visualize gt img-shape and gen shape in shape embedding')
parser.add_argument('--vis_txt_embed', action='store_true', help='Visualize rtv txt-shape and gen shape in shape embedding')

parser.add_argument('--device', type=str, default='cpu', help='Available cuda devices')

# CLIP Mapping Network
parser.add_argument('--finetune_clip', default=False, action='store_true', help='Finetune CLIP by Using Mapping Function')
parser.add_argument('--no_finetune_clip', dest='finetune_clip', action='store_false')
parser.add_argument('--map_code_base', type=str, default='stylegan', help='CLIP Mapping Function Code Base')
parser.add_argument('--map_restore_file', type=str, default='openai_clip/finetune/results', help='path to the checkpoint of Normalizing Flow Model')
parser.add_argument('--h_features', type=int, default=512, help='Number of Hidden Features.')
parser.add_argument('--norm', type=str, default='None', help='Normalization Layer.')
parser.add_argument('--act_f', type=str, default='None', help='Activation Layer.')
parser.add_argument('--l2norm', action='store_true', default=True, help='Use L2 Norm at the last layer.')
parser.add_argument('--no_l2norm', dest='l2norm', action='store_false')
parser.add_argument('--num_layers', type=int, default=1, help='Number of Layers.')
parser.add_argument('--map_result_file', type=str, default='inference_results.txt', help='path to save inference results')

# Normalizing Flow
parser.add_argument('--nf_code_base', type=str, default='pytorch_flows', help='Normalizing Flow Code Base')
parser.add_argument('--nf_model', type=str, default='realnvp', help='Normalizing Flow Model')
parser.add_argument('--nf_restore_file', type=str, default='normalizing_flows/results', help='path to the checkpoint of Normalizing Flow Model')
parser.add_argument('--unconditional', default=False, action='store_true', help='Whether to use a conditional model.')
parser.add_argument('--input_size', type=int, default=256, help='Input feature dimension == OCNET embedding dimension.')
parser.add_argument('--cond_label_size', type=int, default=1024, help='Condition feature dimension == CLIP embedding dimension.')
parser.add_argument('--n_components', type=int, default=10, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--n_blocks', type=int, default=10, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--hidden_size', type=int, default= 1024, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--activation_fn', type=str, default='relu', help='What activation function to use in the MADEs.')
parser.add_argument('--input_order', type=str, default='sequential', help='What input order to use (sequential | random).')
parser.add_argument('--no_batch_norm', action='store_true')
parser.add_argument('--use_mlp', default=False, action='store_true', help='Whether to use MLP for condition.')
parser.add_argument('--no_use_mlp', dest='use_mlp', action='store_false')

# OCNET
parser.add_argument('--shape_gt_path', type=str, default='./', help='Path to GT ShapeNet Shapes')
parser.add_argument('--embed_feat_path', type=str, default='./', help='Path to Preprocessed Shape Embed Feature')
parser.add_argument('--embed_feat_test_path', type=str, default='./', help='Path to Preprocessed Shape Embed Feature')
parser.add_argument('--n_row', type=int, default=5, help='Number of generated mesh of each text phrase.')
parser.add_argument('--out_mesh_folder', type=str, default='./', help='Folder to the outputmesh')
parser.add_argument('--post_process', type=str, default='', help='Config For Post Process')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device
args.out_mesh_folder = args.out_mesh_folder + args.post_process
if args.use_text2shape:
    args.out_mesh_folder = args.out_mesh_folder + '_t2s'
rand_num = random.randint(0, 9999)
args.out_mesh_folder = args.out_mesh_folder + str(rand_num).zfill(4)

args.nf_restore_file = os.path.join(args.nf_restore_file, 'best_model_checkpoint.pt')
args.map_result_file = os.path.join(args.map_restore_file, 'inference_results.txt')
args.map_restore_file = os.path.join(args.map_restore_file, 'best_model_checkpoint.pt')

start_time = time.time()

cfg_path = 'occupancy_networks/configs/voxels/onet_'+args.cate+'_latent_gen' + args.post_process + '.yaml'
cfg = config.load_config(cfg_path, 'occupancy_networks/configs/default.yaml')
print('Cfg File:', cfg_path)
print("NF Model: ", args.nf_model)
print("Restore from: ", args.nf_restore_file)
print("Restore from: ", args.map_restore_file)

print('Restoring map model...')
mp_mlps, mp_model, map_generate_conditioned = load_map_model(args)
mp_mlps, mp_model = restore_map_model(args, mp_mlps, mp_model)
if args.map_code_base == 'NF':
    def map_model(x):
        return map_generate_conditioned(mp_model, mp_mlps(x), args.device, n_row=1)
else:
    map_model = mp_model

print('Restoring nf model...') 
mlps, nf_model, generate_conditioned = load_nf_model(args)
mlps, nf_model = restore_nf_model(args, nf_model, mlps)
# mlps, nf_model, generate_conditioned = torch.nn.Identity(), torch.nn.Identity(), None

print_dict = {}

# Inference from txt
if args.from_txt:
    print("Infering CLIP Text Feature...")
    text_str_list = parse_txt_input(args)
    text_features_list, text_feats_name_list, text_str_list = encode_clip_txt_feats(args, 'ViT-B/32', text_str_list, map_model)
    print("Num of Text features {}".format(len(text_features_list)))

    print("Infering Shape Feature from Text Feature...")
    shape_features_list = generate_embed_from_clip_txt(args, nf_model, mlps, generate_conditioned, text_features_list)
    top_shape_id_list = shape_features_list

    if args.n_retrieval_shape > 0: # Test only for split_by_id
        print("Query Text Feature from Shape Features...")
        txt_shape_r_precision, txt_shape_r_precision5, all_precision, all_precision5, top_shape_id_list = retrieve_in_shape_embed(args, True, shape_features_list, text_str_list, text_feats_name_list, test_only=not args.split_by_text) # For split_by_text, NF & OCNET test set may not include the gt in split_by_text test set.
        print_dict.update({'Text R-Precision@1 in Shape Embedding is {}': txt_shape_r_precision})
        print_dict.update({'Text R-Precision@5 in Shape Embedding is {}': txt_shape_r_precision5})
        if args.vis_correct_only:
            print(len(all_precision), len(all_precision5), len(shape_features_list))
            assert len(all_precision) == len(shape_features_list) and len(all_precision5) == len(shape_features_list)
            indices = [i for i, x in enumerate(all_precision) if x == 1 ]
            shape_features_list = [feat for idx, feat in enumerate(shape_features_list) if idx in indices]
            top_shape_id_list = [feat for idx, feat in enumerate(top_shape_id_list) if idx in indices]
            text_features_list = [feat for idx, feat in enumerate(text_features_list) if idx in indices]
            text_str_list = [feat for idx, feat in enumerate(text_str_list) if idx in indices]
            text_feats_name_list = [feat for idx, feat in enumerate(text_feats_name_list) if idx in indices]
            print('Available Index length is {}'.format(len(indices)))
            print('Available Index length is {}'.format(len(indices)), file=open(args.map_result_file, 'a'))
    
    print("Generating Shape from Shape Feature...")
    out_mesh_folder = generate_shape_from_embed(args, cfg, shape_features_list, top_shape_id_list, text_str_list, n_row=args.n_row, n_max=60)
   
    if args.n_retrieval_clip > 0: # Text2Shape Eval -> test; Otherview, all
        print("Query Text Feature from Image Features...")
        txt_clip_r_precision, txt_clip_r_precision5 = retrieve_in_clip(args, True, map_model, text_features_list, text_str_list, text_feats_name_list, args.out_mesh_folder, test_only=not args.split_by_text)
        print_dict.update({'Text R-Precision@1 in CLIP Embedding is {}': txt_clip_r_precision})
        print_dict.update({'Text R-Precision@5 in CLIP Embedding is {}': txt_clip_r_precision5})

    if args.vis_txt_embed:
        print("Shape Embedding Visualization...")
        visualize_shape_embed(args, True, map_model, nf_model, mlps, generate_conditioned, shape_features_list, text_features_list, text_str_list)

# Inference from img
if args.n_from_img > 0:
    print("Infering CLIP Image Feature...") # Test Only...
    img_shape_features_list, img_str_list = generate_embed_from_clip_img(args, map_model, nf_model, mlps, generate_conditioned, k_gen=args.n_from_img, test_only=True)
    print("Num of Image features {}".format(len(img_shape_features_list)))

    print("Infering Shape Feature from Image Feature...")
    out_mesh_folder = generate_shape_from_embed(args, cfg, img_shape_features_list, img_shape_features_list, img_str_list, n_row=args.n_row)
    
    if args.n_retrieval_clip > 0: # GT
        print("Query Image Feature from Image Features...")
        _ = retrieve_in_clip(args, False, map_model, None, img_str_list, img_str_list, out_mesh_folder)

    if args.n_retrieval_shape > 0: # Test only
        print("Query Image Feature from Shape Features...")
        img_shape_r_precision, _, _, _, _ = retrieve_in_shape_embed(args, False, img_shape_features_list, img_str_list, img_str_list, test_only=True)
        print_dict.update({'Image R-Precision in Shape Embedding is {}': img_shape_r_precision})


    if args.vis_img_embed:
        print("Shape Embedding Visualization...")
        visualize_shape_embed(args, False, map_model, nf_model, mlps, generate_conditioned, shape_features_list, text_features_list, text_str_list)

if args.quantity:
    for key in print_dict:
        print(key.format(print_dict[key]))
        print(key.format(print_dict[key]), file=open(args.map_result_file, 'a'))

end_time = time.time()
print('Total Inference Time is {}'.format(end_time-start_time), file=open(args.map_result_file, 'a'))