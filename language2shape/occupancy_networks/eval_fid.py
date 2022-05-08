import os
import torch
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from im2mesh.encoder.voxels import VoxelEncoder
from im2mesh.utils import binvox_rw
import open3d as o3d
import trimesh
from trimesh.voxel import creation
import tqdm

# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def test():
    # define two collections of activations
    act1 = random(10*2048)
    act1 = act1.reshape((10,2048))
    act2 = random(10*2048)
    act2 = act2.reshape((10,2048))
    # fid between act1 and act1
    fid = calculate_fid(act1, act1)
    print('FID (same): %.3f' % fid)
    # fid between act1 and act2
    fid = calculate_fid(act1, act2)
    print('FID (different): %.3f' % fid)

def save_reference_feat(voxelnet):
    input_folder = 'occupancy_networks/data/ShapeNet/04379243'
    output_path = 'occupancy_networks/out/voxels/onet_pretrained/table_voxel_feats.npy'
    feat_list = []
    for shape_id in os.listdir(input_folder):
        if os.path.isdir(os.path.join(input_folder, shape_id)):
            file_path = os.path.join(input_folder, shape_id, 'model.binvox')
        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
            voxels = torch.from_numpy(voxels.data.astype(np.float32)).cuda().unsqueeze(0)
            
        feat = voxelnet(voxels)
        feat = feat.detach().cpu().numpy()
        feat_list.append(feat)
    feats_array = np.concatenate(feat_list, axis=0)
    print(feats_array.shape)
    np.save(output_path, feats_array)

def load_encoder():
    state = torch.load('occupancy_networks/out/voxels/onet_pretrained/model_best.pt', map_location='cuda')
    state_dict = state['model']
    restore_dict = {}
    for k in state_dict:
        if k.startswith('encoder'):
            v = state_dict[k]
            restore_dict[k.replace('encoder.', '')] = v

    voxelnet = VoxelEncoder(c_dim=256)
    voxelnet.eval().cuda()
    voxelnet.load_state_dict(restore_dict)

    return voxelnet

def calculate_fid_from_folder(voxelnet, input_folder='', tar_feat_path=''):
    print(input_folder)
    tar_feats_array = np.load(tar_feat_path)
    feat_list = []
    for file in tqdm.tqdm(os.listdir(input_folder)):
        if file.endswith('.off') and 'rtv' not in file and 'gt' not in file and '00' in file:
            file_path = os.path.join(input_folder, file)
            try:
                mesh = trimesh.load(file_path)
                voxel_grid=creation.local_voxelize(mesh,mesh.centroid, mesh.extents.max() / 32, 16, fill=True)
                tmp_raw=voxel_grid.encoding.data
                tmp_raw = torch.from_numpy(tmp_raw[:32, :32, :32].astype(np.float32)).cuda().unsqueeze(0)
                feat = voxelnet(tmp_raw)
                feat = feat.detach().cpu().numpy()
                feat_list.append(feat)
            except:
                pass
            if len(feat_list) >= 8:
                break
    feats_array = np.concatenate(feat_list, axis=0)

    fid = calculate_fid(feats_array, tar_feats_array)
    print(fid)

def calculate_fid_for_Text2Shape(voxelnet, input_folder='', tar_feat_path=''):
    tar_feats_array = np.load(tar_feat_path)
    split_file = os.path.join('occupancy_networks/data/ShapeNet/03001627', 'val' + '.lst')
    with open(split_file, 'r') as f:
        valid_model_names = f.read().split('\n')
    feat_list = []
    t2s_model_id_file = 'occupancy_networks/out/voxels/onet_pretrained/t2s_model_ids.txt'
    with open(t2s_model_id_file, 'r') as f:
        model_names = f.readlines()
    for model_name in model_names:
        
        model_id = model_name.split(' ')[0]
        shape_id = model_name.split(' ')[-1][:-1]
        if shape_id in valid_model_names:
            voxel_file_path = os.path.join('occupancy_networks/out/voxels/onet_pretrained/npys', model_id + '_voxel_tensor_output.npy')
            feat = np.load(voxel_file_path)
            feat = feat[:, :, :, 3]
            feat = np.where(feat >0.9, True, False)
            feat = torch.from_numpy(feat.astype(np.float32)).cuda().unsqueeze(0)
            feat = voxelnet(feat)
            feat = feat.detach().cpu().numpy()
            feat_list.append(feat)
    print('Loaded')
    feats_array = np.concatenate(feat_list, axis=0)
    fid = calculate_fid(feats_array, tar_feats_array)
    print(fid)

    
    # feat_list = []
    # for file in tqdm.tqdm(os.listdir(input_folder)):
    #     if file.endswith('.off') and 'rtv' not in file and 'gt' not in file and '00' in file:
    #         file_path = os.path.join(input_folder, file)
    #         try:
    #             mesh = trimesh.load(file_path)
    #             voxel_grid=creation.local_voxelize(mesh,mesh.centroid, mesh.extents.max() / 32, 16, fill=True)
    #             tmp_raw=voxel_grid.encoding.data
    #             tmp_raw = torch.from_numpy(tmp_raw[:32, :32, :32].astype(np.float32)).cuda().unsqueeze(0)
    #             feat = voxelnet(tmp_raw)
    #             feat = feat.detach().cpu().numpy()
    #             feat_list.append(feat)
    #         except:
    #             pass
    #         if len(feat_list) >= 8:
    #             break
    # feats_array = np.concatenate(feat_list, axis=0)

    # fid = calculate_fid(feats_array, tar_feats_array)
    # print(fid)

if __name__ == "__main__":
    # input_folder = ''
    # tar_feat_path = 'occupancy_networks/out/voxels/onet_pretrained/chair_voxel_feats.npy'
    # voxelnet = load_encoder()
    
    # calculate_fid_for_Text2Shape(voxelnet, input_folder, tar_feat_path)

# Play voxels 1b6c268811e1724ead75d368738e0b47
    gt_read_path = 'occupancy_networks/data/ShapeNet/03001627/1b6c268811e1724ead75d368738e0b47/model.binvox'
    gt_write_path = 'occupancy_networks/out/voxels/onet_pretrained/1b6c268811e1724ead75d368738e0b47_vox.obj'
    with open(gt_read_path, 'rb') as fp:
        vox = trimesh.exchange.binvox.load_binvox(fp)
    vox_mesh = vox.as_boxes()
    vox_mesh.export(gt_write_path)