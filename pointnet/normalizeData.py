# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:16:49 2022

@author: ThinkPad
"""
import numpy as np
import copy
import open3d as o3d
from numpy import linalg as LA
import time
import os
import open3d.visualization.rendering as rendering
import math
from multiprocessing import Pool
import array



def normalizePoints(scanpoints):
    x = scanpoints[scanpoints[:, 0].argmin(),0]
    y = scanpoints[scanpoints[:, 1].argmin(),1]
    z = scanpoints[scanpoints[:, 2].argmin(),2]
    offset = np.array([x,y,z])
    scanpoints = scanpoints - offset
    scanpoints = scanpoints/scanpoints[scanpoints[:, 1].argmax(),1]
    scanpoints = scanpoints-0.5*np.array([scanpoints[scanpoints[:, 0].argmax(),0],scanpoints[scanpoints[:, 1].argmax(),1],scanpoints[scanpoints[:, 2].argmax(),2]])
    np.random.shuffle(scanpoints)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scanpoints[0:1024,:])
    R = pcd.get_rotation_matrix_from_xyz((3*(1)*np.pi / 2, 0*np.pi / 2, 1*np.pi / 2))
    pcd.rotate(R, center=(0, 0, 0))
    points = np.asarray(pcd.points)
    return points




# name = '589e717feb809e7c1c5a16cc04345597'
# path2 = r"D:\data\dataset_small_partial\03001627\46323c7986200588492d9da2668ec34c\pointcloud41_partial.npz"
# data = np.load(path2)
# # lst = data.files
# points = data['points_r']
# print(points[points[:, 0].argmax(),1])


# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(points[0:-1,:])
# o3d.visualization.draw_geometries([pcd1]) 
