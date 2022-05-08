import torch
# import torch.distributions as dist
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))

from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
from im2mesh.utils.visualize import visualize_data
from im2mesh.utils.voxels import VoxelGrid
from im2mesh import data
import numpy as np

def ocnet_generation_from_z(args, cfg, feats=None, shape_ids=None, folder='./', text='default', device='cpu', n_row=10, vis=True):
    # data [B, dim]
    out_dir = cfg['training']['out_dir']

    # Create directories if necessary
    mesh_dir = folder
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    print(mesh_dir)

    if vis:
        # Dataset
        dataset = config.get_dataset('train', cfg, return_idx=True)

        # Model
        model = config.get_model(cfg, device=device, dataset=dataset)
        checkpoint_io = CheckpointIO(out_dir, model=model)
        checkpoint_io.load(cfg['test']['model_file'])
        
        # Generator
        generator = config.get_generator(model, cfg, device=device)
        # Generate
        model.eval()

        assert n_row == feats.shape[0]
        for idx_mesh in range(n_row):
            c = feats[idx_mesh, :].unsqueeze(0)
            z = generator.model.get_z_from_prior((1, ), sample=generator.sample).to(device)
            out = generator.generate_from_latent(z, c)
            # Get statistics
            try:
                mesh, stats_dict = out
            except TypeError:
                mesh, stats_dict = out, {}
    
            # Write output
        
            text_out = text.replace(' ', '_')
            mesh_out_file = os.path.join(mesh_dir, '{}_{}.off'.format(text_out, str(idx_mesh).zfill(2)) )
            mesh.export(mesh_out_file)
    
    return mesh_dir