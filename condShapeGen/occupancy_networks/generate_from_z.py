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

    if vis:
        # Dataset
        dataset = config.get_dataset('train', cfg, return_idx=True)
        points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
        with_transforms = cfg['model']['use_camera']
        pointfield = data.PointsField(
            cfg['data']['points_file'], points_transform,
            with_transforms=with_transforms,
            unpackbits=cfg['data']['points_unpackbits'],
        )
        inputsfield = data.VoxelsField(
            cfg['data']['voxels_file']
        )

        # Model
        model = config.get_model(cfg, device=device, dataset=dataset)
        checkpoint_io = CheckpointIO(out_dir, model=model)
        checkpoint_io.load(cfg['test']['model_file'])
        
        # Generator
        generator = config.get_generator(model, cfg, device=device)
        # Generate
        model.eval()

        # Generate outputs
        z_dim = cfg['model']['z_dim']
        c_dim = cfg['model']['c_dim']
        feats_dim = feats.shape[1]
        cut = cfg['generation']['cut']

        assert n_row == feats.shape[0]
        for idx_mesh in range(n_row):
            # if feats_dim == c_dim: # Input feature is c, normally c_dim==256
            #     c = feats[idx_mesh, :].unsqueeze(0)
            #     if z_dim != 0:
            #         shape_id = shape_ids[idx_mesh]
            #         model_path = os.path.join(cfg['data']['path'], cfg['data']['classes'][0], shape_id)
            #         shape_data = pointfield.load(model_path, 0, 0)
            #         p = torch.from_numpy(shape_data.get(None)).to(device).unsqueeze(0)
            #         occ = torch.from_numpy(shape_data.get('occ')).to(device).unsqueeze(0)
            #         # z, = model.encoder_latent(p, occ, c)
            #         q_z = model.infer_z(p, occ, c)
            #         z = q_z.rsample()
            #     else:
            #         z = generator.model.get_z_from_prior((1, ), sample=generator.sample).to(device)

            # elif feats_dim == z_dim: # Input feature is z, normally z_dim==128/256
            #     z = feats[idx_mesh, :].unsqueeze(0)
            #     if c_dim != 0:
            #         shape_id = shape_ids[idx_mesh]
            #         model_path = os.path.join(cfg['data']['path'], cfg['data']['classes'][0], shape_id)
            #         shape_data = inputsfield.load(model_path, 0, 0)
            #         inputs = torch.from_numpy(shape_data).to(device).unsqueeze(0)
            #         c = model.encode_inputs(inputs)
            #     else:
            #         c = None
            # elif feats_dim == z_dim+c_dim: # Input feature is z, normally z_dim==128/256
            #     if cut == 'cut':
            #         shape_id = shape_ids[idx_mesh]
            #         model_path = os.path.join(cfg['data']['path'], cfg['data']['classes'][0], shape_id)
                    
            #         shape_data = inputsfield.load(model_path, 0, 0)
            #         inputs = torch.from_numpy(shape_data).to(device).unsqueeze(0)
            #         c = model.encode_inputs(inputs)

            #         shape_data = pointfield.load(model_path, 0, 0)
            #         p = torch.from_numpy(shape_data.get(None)).to(device).unsqueeze(0)
            #         occ = torch.from_numpy(shape_data.get('occ')).to(device).unsqueeze(0)
            #         # z, = model.encoder_latent(p, occ, c)
            #         q_z = model.infer_z(p, occ, c)
            #         z = q_z.rsample()

            #     else:
            #         z = feats[idx_mesh, :z_dim].unsqueeze(0)
            #         c = feats[idx_mesh, z_dim:].unsqueeze(0)
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