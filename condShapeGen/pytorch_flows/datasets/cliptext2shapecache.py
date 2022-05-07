import os
import torch
import numpy as np
import pickle

image_folder = "occupancy_networks/data/ShapeNet/04379243/"
# origin
# clip_img_feat_path = "occupancy_networks/data/ShapeNet/clip_img_feat.pickle"
# my rendered clip features
# clip_img_feat_path = "../../data/embed_ShapeNetCore.v1/clip_img_feat_20view_shade.pickle"
clip_txt_feat_path = "../../data/embed_ShapeNetCore.v1/04379243/clip_txt_feat_text2shape.pickle"
# clip_img_feat_path = "../../data/embed_ShapeNetCore.v1/clip_img_feat_1view_shade.pickle"

# embed_feat_path = "occupancy_networks/out/voxels/onet_chair/generation/embed/03001627/embed_feats.pickle"
embed_feat_path = "occupancy_networks/out/voxels/onet_table/generation/embed/04379243/embed_feats.pickle"
class ClipText2ShapeCache:
    class Data:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def __init__(self, args, clip_txt_feat_path=clip_txt_feat_path, embed_feat_path=embed_feat_path, istrain=False):
        split = "train"
        split_file = os.path.join(image_folder, split + '.lst')

        with open(split_file, 'r') as f:
            train_model_names = f.read().split('\n')
        print("Number of Instance", len(train_model_names))

        split = "test"
        split_file = os.path.join(image_folder, split + '.lst')
        with open(split_file, 'r') as f:
            test_model_names = f.read().split('\n')
        print("Number of Instance", len(test_model_names))
        
        with open(embed_feat_path, 'rb') as handle:
            embed_feats = pickle.load(handle) 
        with open(clip_txt_feat_path, 'rb') as handle:
            clip_text_feats = pickle.load(handle)

        # Train
        trn_embed_feats_list = []
        trn_clip_feats_list = []
        for model_name in train_model_names:
            if model_name in list(clip_text_feats.keys()):
                for view_key in list(clip_text_feats[model_name].keys()):
                    trn_embed_feats_list.append(embed_feats[model_name].reshape(1, -1))
                    trn_clip_feats_list.append(clip_text_feats[model_name][view_key])
        print(len(trn_clip_feats_list))
        tst_embed_feats = np.vstack(trn_embed_feats_list)
        trn_clip_feats = np.vstack(trn_clip_feats_list)
        trn_clip_feats = trn_clip_feats / np.linalg.norm(trn_clip_feats, axis=-1, keepdims=True)
        
        self.trn = self.Data(tst_embed_feats, trn_clip_feats)
        
        # Test and Val
        embed_feat_path = embed_feat_path.replace('embed_feats', 'embed_feats_test')
        with open(embed_feat_path, 'rb') as handle:
            embed_feats = pickle.load(handle) 
        
        tst_embed_feats_list = []
        tst_clip_feats_list = []
        for model_name in test_model_names:
            if model_name in list(clip_text_feats.keys()):
                for view_key in list(clip_text_feats[model_name].keys()):
                    tst_embed_feats_list.append(embed_feats[model_name].reshape(1, -1))
                    tst_clip_feats_list.append(clip_text_feats[model_name][view_key])
        print(len(tst_clip_feats_list))
        tst_embed_feats = np.vstack(tst_embed_feats_list)
        tst_clip_feats = np.vstack(tst_clip_feats_list)
        tst_clip_feats = tst_clip_feats / np.linalg.norm(tst_clip_feats, axis=-1, keepdims=True)

        self.tst = self.Data(tst_embed_feats, tst_clip_feats)
        self.val = self.Data(tst_embed_feats, tst_clip_feats)
        self.n_dims = self.trn.x.shape[1]
        self.n_labels = self.trn.y.shape[1]