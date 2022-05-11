# -*- coding: utf-8 -*-
"""
Created on Tue May 10 04:27:29 2022

@author: ThinkPad
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from PartialScan import PartialScans,unpickle,inferencePartialScans
from model import feature_transform_regularizer
from pointnetCls import PointNetCls
import torch.nn.functional as F
from tqdm import tqdm
import random
# import open3d as o3d
from normalizeData import normalizePoints

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=3, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--checkpoint', type=str, default='/gpfs/data/ssrinath/ychen485/TextCondRobotFetch/pointnet/cls/cls_model_10.pth', help="checkpoint dir")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

latent_code = "/gpfs/data/ssrinath/ychen485/hyperPointnet/pointnet/03001627/ocnet_shapefeature_pc/embed_feats_train.pickle"
latent_code_test = "/gpfs/data/ssrinath/ychen485/hyperPointnet/pointnet/03001627/ocnet_shapefeature_pc/embed_feats_test.pickle"
latent_code_val = "/gpfs/data/ssrinath/ychen485/hyperPointnet/pointnet/03001627/ocnet_shapefeature_pc/embed_feats_val.pickle"
shape_folder = "/gpfs/data/ssrinath/ychen485/partialPointCloud/03001627"
latent_dim = 512

dataset = PartialScans(latentcode_dir = latent_code, shapes_dir = shape_folder)

test_dataset = PartialScans(latentcode_dir = latent_code_test, shapes_dir = shape_folder)

val_dataset = PartialScans(latentcode_dir = latent_code_val, shapes_dir = shape_folder)

inference_loader = inferencePartialScans(shapes_dir = "")

inferdataloader = torch.utils.data.DataLoader(
    inference_loader,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

valdataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

latent_dict = unpickle(latent_code)
keylist = list(latent_dict.keys())
latent_dict_test = unpickle(latent_code_test)
keylist_test = list(latent_dict_test.keys())
latent_dict_val = unpickle(latent_code_val)
keylist_val = list(latent_dict_val.keys())

print("train set lenth: "+ str(len(dataset)) +", test set length: "+ str(len(test_dataset)))
try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=2, feature_transform=opt.feature_transform)

if opt.checkpoint != " ":
    checkpoint = torch.load(opt.checkpoint)
    classifier.load_state_dict(checkpoint)
    pass
    
classifier.cuda()

num_batch = len(dataset) / opt.batchSize
total_correct = 0
for epoch in range(1):
    
    for i, data in enumerate(valdataloader, 0):
        points_o, label = data
        points = points_o[:,0:1024,:].to(torch.float32)
        # print(points.shape)
        points.to(torch.float32)
        points = points.transpose(2, 1)
        target_np = np.zeros((len(label),))
        t_idx = random.randint(0,len(label)-1)
        target_np[t_idx] = 1
        target = torch.from_numpy(target_np).to(torch.int64)
        latents = np.zeros((1, latent_dim))
        latents[0] = latent_dict_val[label[t_idx]]
        # for j in range(opt.batchSize):
        #     if target[j] == 1:
        #         latents[j] = latent_dict[label[j]]
        #     else:
        #         idx = random.randint(0,len(keylist))
        #          name = keylist[idx]
        #        while(name == label[j]):
        #             idx = random.randint(0,len(keylist))
        #            name = keylist[idx]
        #         latents[j] = latent_dict[name]
        z  = torch.from_numpy(latents).to(torch.float32)
        points, target, z = points.cuda(), target.cuda(), z.cuda()
        # optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points, z)
        # print(pred.shape)
        pred = pred[0]
        # loss = F.nll_loss(pred, target)
        # if opt.feature_transform:
        #     loss += feature_transform_regularizer(trans_feat) * 0.001
        # loss.backward()
        # optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct = total_correct + correct.item()
        if i%100 == 0:
            print('[%d: %d/%d] accuracy: %f' % (epoch, i, num_batch,  total_correct / (100* opt.batchSize)))
            total_correct = 0
            print(pred,pred_choice)
def inference(scanpoints, latentcode, classifier):
    points_r = normalizePoints(scanpoints)
    points = np.random.rand(3,1024,3)
    points[0] = points_r[0:1024,:]
    
    idx = random.randint(0,len(label)-1)
    i = random.randint(0,2)
    j = random.randint(0,7)
    path = shape_folder +"/" + label[t_idx] + "/pointcloud"+str(j)+str(i)+"_partial.npz"
    data = np.load(path)
    scanpoints = data['points_r']
    points_r = normalizePoints(scanpoints)
    points_r = scanpoints
    points[1] = points_r[0:1024,:]

    idx = random.randint(0,len(label)-1)
    i = random.randint(0,2)
    j = random.randint(0,7)
    path = shape_folder +"/" + label[t_idx] + "/pointcloud"+str(j)+str(i)+"_partial.npz"
    data = np.load(path)
    scanpoints = data['points_r']
    points_r = normalizePoints(scanpoints)
    points_r = scanpoints
    points[2] = points_r[0:1024,:]
    ischair = 0;
    haveTarget = False
    for j in range(5):
        for i in range(10):
            points = torch.from_numpy(points[:,0:1024,:]).to(torch.float32)
            points.to(torch.float32)
            points = points.transpose(2, 1)
            latents = np.zeros((1, latent_dim))
            latents[0] = latentcode[j]
            z = torch.from_numpy(latents).to(torch.float32)
            points, z = points.cuda(), z.cuda()
            classifier = classifier.eval()
            pred, trans, trans_feat = classifier(points, z)
            pred = pred[0]
            pred_choice = pred.data.max(1)[1].cpu()
            # print(torch.exp(pred),pred_choice)
            ischair = pred_choice[0] + ischair
        print (ischair)
        ischair = 0
        if ischair - 7 > 0:
            haveTarget = True
        # if ischair > 7 :haveTarget = True
    return haveTarget 



# print(points)
# print("inferencing:" )
path = "testpoints.npz"
# path = "/gpfs/data/ssrinath/ychen485/partialPointCloud/03001627/ff9915c51ece4848cfc689934e433906/pointcloud70_partial.npz"
data = np.load(path)
# lst = data.files
scanpoints = data['points_r']

# pcd1 = o3d.io.read_point_cloud(path)
# scanpoints = np.asarray(pcd1.points)

# print(scanpoints.shape)
points_r = normalizePoints(scanpoints)
# points_r = scanpoints

# points_o[2] = points[0:1024,:]
points = np.random.rand(3,1024,3)
# points[0] = points_r[0:1024,:]
points[0] = points_r[0:1024,:]

idx = random.randint(0,len(label)-1)
i = random.randint(0,2)
j = random.randint(0,7)
path = shape_folder +"/" + label[t_idx] + "/pointcloud"+str(j)+str(i)+"_partial.npz"
# path = "/gpfs/data/ssrinath/ychen485/partialPointCloud/03001627/ff9915c51ece4848cfc689934e433906/pointcloud41_partial.npz"
data = np.load(path)
# lst = data.files
scanpoints = data['points_r']

# pcd1 = o3d.io.read_point_cloud(path)
# scanpoints = np.asarray(pcd1.points)

# print(scanpoints.shape)
points_r = normalizePoints(scanpoints)
points_r = scanpoints

# points_o[2] = points[0:1024,:]
# points = np.zeros((3,1024,3))
# points[0] = points_r[0:1024,:]

points[1] = points_r[0:1024,:]
idx = random.randint(0,len(label)-1)
i = random.randint(0,2)
j = random.randint(0,7)
path = shape_folder +"/" + label[t_idx] + "/pointcloud"+str(j)+str(i)+"_partial.npz"
# path  = "/gpfs/data/ssrinath/ychen485/partialPointCloud/03001627/589e717feb809e7c1c5a16cc04345597/pointcloud62_partial.npz"
data = np.load(path)
# lst = data.files
scanpoints = data['points_r']

# pcd1 = o3d.io.read_point_cloud(path)
# scanpoints = np.asarray(pcd1.points)

# print(scanpoints.shape)
points_r = normalizePoints(scanpoints)
points_r = scanpoints

# points_o[2] = points[0:1024,:]
# points = np.zeros((3,1024,3))
# points[0] = points_r[0:1024,:]

points[2] = points_r[0:1024,:]

# from torch.autograd import Variable
# sim_data = Variable(torch.rand(32,3,1024))

# print(points)
# print(points_o)
points = torch.from_numpy(points[:,0:1024,:]).to(torch.float32)
points.to(torch.float32)
# print(points)
points = points.transpose(2, 1)
# print(points)
latents = np.zeros((1, latent_dim))
latents[0] = latent_dict['46323c7986200588492d9da2668ec34c']
z = torch.from_numpy(latents).to(torch.float32)
# print(z)
points, target, z = points.cuda(), target.cuda(), z.cuda()
classifier = classifier.eval()
pred, trans, trans_feat = classifier(points, z)
pred = pred[0]
pred_choice = pred.data.max(1)[1]
print(torch.exp(pred),pred_choice)

latents[0] = latent_dict_val['ba673ea75085e46cbfd72d7396bc040a']
z = torch.from_numpy(latents).to(torch.float32)
points, target, z = points.cuda(), target.cuda(), z.cuda()
classifier = classifier.train()
pred, trans, trans_feat = classifier(points, z)
pred = pred[0]
pred_choice = pred.data.max(1)[1]
print(torch.exp(pred),pred_choice)

latents[0] = latent_dict_test['ff9915c51ece4848cfc689934e433906']
z = torch.from_numpy(latents).to(torch.float32)
points, target, z = points.cuda(), target.cuda(), z.cuda()
classifier = classifier.train()
pred, trans, trans_feat = classifier(points, z)
pred = pred[0]
pred_choice = pred.data.max(1)[1]
print(torch.exp(pred),pred_choice)

latents[0] = latent_dict_test['fc07472e4dd1b6698ae97f14e63e7e01']
z = torch.from_numpy(latents).to(torch.float32)
points, target, z = points.cuda(), target.cuda(), z.cuda()
classifier = classifier.train()
pred, trans, trans_feat = classifier(points, z)
pred = pred[0]
pred_choice = pred.data.max(1)[1]
print(torch.exp(pred),pred_choice)

latents[0] = latent_dict['3bd437d38068f4a61f285be552b78f9a']
latents[0] = (np.load('../language2shape/results/shape_0032.npy')[2])
z = torch.from_numpy(latents).to(torch.float32)
# print(z)
points, target, z = points.cuda(), target.cuda(), z.cuda()
classifier = classifier.eval()
pred, trans, trans_feat = classifier(points, z)
pred = pred[0]
pred_choice = pred.data.max(1)[1]
path = "testpoints.npz"
# path = "/gpfs/data/ssrinath/ychen485/partialPointCloud/03001627/ff9915c51ece4848cfc689934e433906/pointcloud70_partial.npz"
data = np.load(path)
# lst = data.files
scanpoints = data['points_r']

# pcd1 = o3d.io.read_point_cloud(path)
# scanpoints = np.asarray(pcd1.points)

# print(scanpoints.shape)
points_r = normalizePoints(scanpoints)
inference(scanpoints, np.load('../language2shape/results/shape_0032.npy'),classifier)
