# -*- coding: utf-8 -*-
"""
Created on Sat May  7 17:36:23 2022

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
from PartialScan import PartialScans,unpickle
from model import feature_transform_regularizer
from pointnetCls import PointNetCls
import torch.nn.functional as F
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=3, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--checkpoint', type=str, default=' ', help="checkpoint dir")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

latent_code = "/gpfs/data/ssrinath/ychen485/hyperPointnet/pointnet/03001627/ocnet_shapefeature_pc/embed_feats_trian.pickle"
latent_code_test = "/gpfs/data/ssrinath/ychen485/hyperPointnet/pointnet/03001627/ocnet_shapefeature_pc/embed_feats_test.pickle"
shape_folder = "/gpfs/data/ssrinath/ychen485/partialPointCloud/03001627"
latent_dim = 512

dataset = PartialScans(latentcode_dir = latent_code, shapes_dir = shape_folder)

test_dataset = PartialScans(latentcode_dir = latent_code_test, shapes_dir = shape_folder)

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

latent_dict = unpickle(latent_code)
keylist = list(latent_dict.keys())
latent_dict_test = unpickle(latent_code_test)
keylist_test = list(latent_dict_test.keys())

print("train set lenth: "+ str(len(dataset)) +", test set length: "+ str(len(test_dataset)))
try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=2, feature_transform=opt.feature_transform)

if opt.checkpoint != " ":
    checkpoint = torch.load(opt.checkpoint)
    classifier.load_state_dict(checkpoint)

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize
total_correct = 0
for epoch in range(opt.nepoch):
    
    for i, data in enumerate(dataloader, 0):
        points_o, label = data
        points = points_o[:,0:1024,:].to(torch.float32)
        points.to(torch.float32)
        points = points.transpose(2, 1)
        target_np = np.zeros((len(label),))
        t_idx = random.randint(0,len(label)-1)
        target_np[t_idx] = 1
        target = torch.from_numpy(target_np).to(torch.int64)
        latents = np.zeros((1, latent_dim))
        latents[0] = latent_dict[label[t_idx]]
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
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points, z)
        # print(pred.shape)
        pred = pred[0]
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct = total_correct + correct.item()
        if i%100 == 0:
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), total_correct / (100* opt.batchSize)))
            total_correct = 0
    scheduler.step()
    test_correct = 0
    for j, data in enumerate(testdataloader, 0):
        points_o, label = data
        points = points_o[:,0:1024,:].to(torch.float32)
        points.to(torch.float32)
        points = points.transpose(2, 1)
        target_np = np.zeros((len(label),))
        t_idx = random.randint(0,len(label)-1)
        target_np[t_idx] = 1
        target = torch.from_numpy(target_np).to(torch.int64)
        latents = np.zeros((1, latent_dim))
        latents[0] = latent_dict_test[label[t_idx]]
        z  = torch.from_numpy(latents).to(torch.float32)
        points, target, z = points.cuda(), target.cuda(), z.cuda()
        # optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points, z)
        # print(pred.shape)
        pred = pred[0]
        
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        test_correct = test_correct + correct.item()
    print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch,
                                                        blue('test'), loss.item(), test_correct/float(len(test_dataset))))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' %
               (opt.outf, epoch))
