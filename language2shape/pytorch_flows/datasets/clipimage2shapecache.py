import os
import torch
import numpy as np
import pickle

image_folder_chair = "occupancy_networks/data/ShapeNet/03001627/"

# origin
clip_img_feat_path_chair = "data/clip_img_feat_1view_texture_ground_keyshot.pickle"
embed_feat_path_chair = "occupancy_networks/out/pointcloud/onet/pretrained/embed/03001627/embed_feats_train.pickle"

chair_data = [image_folder_chair, clip_img_feat_path_chair, embed_feat_path_chair]
# table_data = [image_folder_table, clip_img_feat_path_table, embed_feat_path_table]
class ClipImage2ShapeCache():
    class Data:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def __init__(self, args, B_MappingNetwork=None, clip_img_feat_path=clip_img_feat_path_chair, embed_feat_path=embed_feat_path_chair, istrain=False):
        if args.cate == 'chair':
            image_folder, clip_img_feat_path, embed_feat_path = chair_data[0], chair_data[1], chair_data[2]
        elif args.cate == 'table':
            image_folder, clip_img_feat_path, embed_feat_path = table_data[0], table_data[1], table_data[2]
        
        split = "train"
        split_file = os.path.join(image_folder, split + '.lst')

        with open(split_file, 'r') as f:
            train_model_names = f.read().split('\n')
        print("Number of Instance", len(train_model_names))

        split = "val"
        split_file = os.path.join(image_folder, split + '.lst')
        with open(split_file, 'r') as f:
            test_model_names = f.read().split('\n')
        print("Number of Instance", len(test_model_names))
        
        with open(embed_feat_path, 'rb') as handle:
            embed_feats = pickle.load(handle) 
        with open(clip_img_feat_path, 'rb') as handle:
            clip_img_feats = pickle.load(handle)

        # Map CLIP Features
        if args.finetune_clip:
            map_model = B_MappingNetwork().to(args.device)
            state = torch.load(args.map_restore_file, map_location=args.device)
            map_model.load_state_dict(state['model_state'])
            map_model.eval()

        # Train
        trn_embed_feats_list = []
        trn_clip_feats_list = []
        i_stop = -4 if len(list(clip_img_feats.keys())[0]) < 32 else None
        for model_name in train_model_names:
            if model_name[:i_stop] in list(clip_img_feats.keys()):
                for view_key in list(clip_img_feats[model_name[:i_stop]].keys()):
                    trn_embed_feats_list.append(embed_feats[model_name].reshape(1, -1))
                    trn_clip_feats_list.append(clip_img_feats[model_name[:i_stop]][view_key])
        print(len(trn_clip_feats_list))
        tst_embed_feats = np.vstack(trn_embed_feats_list)
        trn_clip_feats = np.vstack(trn_clip_feats_list)
        trn_clip_feats = trn_clip_feats / np.linalg.norm(trn_clip_feats, axis=-1, keepdims=True)
        
        if args.finetune_clip:
            with torch.no_grad():
                trn_clip_feats = torch.from_numpy(trn_clip_feats).to(args.device).float()
                trn_clip_feats = trn_clip_feats.view(-1, 512)
                trn_clip_feats, _, _ = map_model(trn_clip_feats, trn_clip_feats)
                trn_clip_feats = trn_clip_feats.cpu().numpy()

        # Other different conditional feature coding strategy: positional encoding, duplicate...
        # This part of the code is crappy.
        sin_trn_clip_feats = np.sin(trn_clip_feats * np.pi)
        cos_trn_clip_feats = np.cos(trn_clip_feats * np.pi)
        sin2_trn_clip_feats = np.sin(2 * trn_clip_feats * np.pi)
        cos2_trn_clip_feats = np.cos(2 * trn_clip_feats * np.pi)
        pos_trn_clip_feats = np.concatenate([sin_trn_clip_feats, cos_trn_clip_feats, sin2_trn_clip_feats, cos2_trn_clip_feats], axis=-1)
        dbl_trn_clip_feats = np.concatenate([trn_clip_feats, trn_clip_feats], axis=-1)
        self.trn = self.Data(tst_embed_feats, trn_clip_feats)
        
        # Test and Val
        embed_feat_path = embed_feat_path.replace('embed_feats_train', 'embed_feats_val')
        with open(embed_feat_path, 'rb') as handle:
            embed_feats = pickle.load(handle) 
        
        tst_embed_feats_list = []
        tst_clip_feats_list = []
        for model_name in test_model_names:
            if model_name[:i_stop] in list(clip_img_feats.keys()):
                for view_key in list(clip_img_feats[model_name[:i_stop]].keys()):
                    tst_embed_feats_list.append(embed_feats[model_name].reshape(1, -1))
                    tst_clip_feats_list.append(clip_img_feats[model_name[:i_stop]][view_key])
        print(len(tst_clip_feats_list))
        tst_embed_feats = np.vstack(tst_embed_feats_list)
        tst_clip_feats = np.vstack(tst_clip_feats_list)
        tst_clip_feats = tst_clip_feats / np.linalg.norm(tst_clip_feats, axis=-1, keepdims=True)

        if args.finetune_clip:
            with torch.no_grad():
                tst_clip_feats = torch.from_numpy(tst_clip_feats).to(args.device).float()
                tst_clip_feats = tst_clip_feats.view(-1, 512)
                tst_clip_feats, _, _ = map_model(tst_clip_feats, tst_clip_feats)
                tst_clip_feats = tst_clip_feats.cpu().numpy()

        # Other different conditional feature coding strategy.
        # This part of the code is crappy.
        sin_tst_clip_feats = np.sin(tst_clip_feats * np.pi)
        cos_tst_clip_feats = np.cos(tst_clip_feats * np.pi)
        sin2_tst_clip_feats = np.sin(2 * tst_clip_feats * np.pi)
        cos2_tst_clip_feats = np.cos(2 * tst_clip_feats * np.pi)
        pos_tst_clip_feats = np.concatenate([sin_tst_clip_feats, cos_tst_clip_feats, sin2_tst_clip_feats, cos2_tst_clip_feats], axis=-1)
        dbl_tst_clip_feats = np.concatenate([tst_clip_feats, tst_clip_feats], axis=-1)
        
        self.tst = self.Data(tst_embed_feats, tst_clip_feats)
        self.val = self.Data(tst_embed_feats, tst_clip_feats)
        self.n_dims = self.trn.x.shape[1]
        self.n_labels = self.trn.y.shape[1]