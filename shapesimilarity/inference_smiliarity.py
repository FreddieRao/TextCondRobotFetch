import torch
# import torch.distributions as dist
import os

import sys
sys.path.append(os.path.dirname(__file__))

import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
from im2mesh.data.core import ShapeNetPairDataset
import numpy as np
import pickle

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('--config', type=str, default='configs/pointcloud/onet_pretrained.yaml', help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'], 'partial')

batch_size = cfg['generation']['batch_size']
input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
split = 'val'
dataset = config.get_dataset(split, cfg, return_idx=True)

complete_dataset_folder= 'data/ShapeNet/03001627'
partial_dataset_folder = 'data/PartialShapeNetChair/dataset_small_partial/dataset_small_partial/03001627/03001627'
inference_split='train'
inference_dataset = ShapeNetPairDataset(complete_dataset_folder, partial_dataset_folder, inference_split)
inference_loader = torch.utils.data.DataLoader(
    inference_dataset, batch_size=1, num_workers=0, shuffle=False)
# Model
model = config.get_model(cfg, device=device, dataset=dataset)
checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Statistics
embed_feats = {}

# Generate
model.eval()

# Count how many models already created
for it, data in enumerate(tqdm(inference_loader)):
    # Output folders
    complete_points_dict, partial_points_dict, unpair_partial_points_dict = data
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    embed_dir = os.path.join(generation_dir, 'embed')

    if not os.path.exists(pointcloud_dir):
        os.makedirs(pointcloud_dir)

    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)

    complete_embed = generator.generate_embed(complete_points_dict)
    partial_embed = generator.generate_embed(partial_points_dict)
    unpair_partial_embed = generator.generate_embed(unpair_partial_points_dict)
    distance = torch.cdist(complete_embed, partial_embed).squeeze()
    unpair_distance = torch.cdist(complete_embed, unpair_partial_embed).squeeze()
