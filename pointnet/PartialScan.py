# -*- coding: utf-8 -*-
"""
Created on Sat May  7 16:51:20 2022

@author: ThinkPad
"""
import pickle
import os
import pandas as pd
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
import random

def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled" 
	object produced with cPickle. The code below will open up a 
	"pickled" object (each file) and return a dictionary.

	NOTE: DO NOT EDIT

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

class PartialScans(data.Dataset):
    def __init__(self, latentcode_dir, shapes_dir):
        self.latent_code = unpickle(latentcode_dir)
        self.shapes_dir = shapes_dir
        self.labels = list(self.latent_code.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        i = random.randint(0,2)
        j = random.randint(0,7)
        # print(self.labels[idx])
        # for j in range(8):
            # with Pool(5) as p:
            #     p.map(save_partial,[0,1,2])
            # for i in range(3):
        shapes_path = os.path.join(self.shapes_dir, self.labels[idx]+"/pointcloud"+str(j)+str(i)+"_partial.npz")
        # shapes_path = os.path.join(self.shapes_dir, self.labels[idx]+"/pointcloud.npz")
        shape = np.load(shapes_path)['points_r']
        # shape = np.load(shapes_path)['points']
        label = self.labels[idx]
        latent_code = self.latent_code[label]
        return shape[0:1024,:],label
    
# unpickled_file = unpickle(r"E:\Code\IVL\shapeSearch\HyperPointnet\pointnet\03001627\embed_feats.pickle")
# print(unpickled_file.keys())
# shapesdir = r"D:\data\dataset_small_partial\03001627"
# i = random.randint(0,3)
# j = random.randint(0,8)
# shapes_path = os.path.join(shapesdir, "b2ba1569509cdb439451566a8c6563ed\pointcloud"+str(j)+str(i)+"_partial.npz")
# print(shapes_path)
# data = np.load(shapes_path)
# lst = data.files
# points = data['points_r']
# print(points.shape)
class inferencePartialScans(data.Dataset):
    def __init__(self, shapes_dir):
        self.shapes_dir = shapes_dir
        self.labels = [1,0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, pointcloud):
        i = random.randint(0,2)
        j = random.randint(0,7)
        # print(self.labels[idx])
        # for j in range(8):
            # with Pool(5) as p:
            #     p.map(save_partial,[0,1,2])
            # for i in range(3):
        shapes_path = os.path.join(self.shapes_dir)
        # shapes_path = os.path.join(self.shapes_dir, self.labels[idx]+"/pointcloud.npz")
        shape = np.load(shapes_path)['points_r']
        # shape = np.load(shapes_path)['points']
        label = self.labels
        return pointcloud[0:1024,:],label



