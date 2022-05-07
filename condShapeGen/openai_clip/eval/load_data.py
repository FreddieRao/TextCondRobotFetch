import os
import pickle
import numpy as np
import torch

def parse_txt_input(args):
    if args.unconditional: 
        text_str_list = [str(_).zfill(2) for _ in range(10)]
    else:
        text_str_list = [item for item in args.text.split(';')]
    return text_str_list

def get_shapenet_full_names(args):
    full_names = [ _ for _ in os.listdir(args.shape_gt_path) if os.path.isdir(os.path.join(args.shape_gt_path, _))]
    return full_names

def load_clip_feats(args, map_model=None, load_valid_cates_only=False, test_only=False):
    full_names = get_shapenet_full_names(args)
    if load_valid_cates_only:
        with open(args.valid_cate_anno_path + '_all.lst', 'r') as f:
            all_valid_full_names = f.read().split('\n')
        all_valid_short_names = [_[:-4] for _ in all_valid_full_names]
        all_valid_names = all_valid_short_names + all_valid_full_names
    if test_only:
        split_file = os.path.join(args.shape_gt_path, 'val' + '.lst')
        with open(split_file, 'r') as f:
            val_model_names = f.read().split('\n')

    with open(args.clip_img_feat_path, 'rb') as handle:
        clip_img_feats = pickle.load(handle)
        clip_feats_array = []
        clip_name_array = []
        for key in clip_img_feats:
            full_name = [_ for _ in full_names if _.startswith(key)][0]
            if load_valid_cates_only:
                if full_name not in all_valid_names:
                    continue
            if test_only:
                if full_name not in val_model_names:
                    continue
            for idx in clip_img_feats[key]:
                clip_feats_array.append(clip_img_feats[key][idx])
                clip_name_array.append(key + '@' + idx)
        clip_feats = np.stack(clip_feats_array, axis=0)
        clip_feats = torch.from_numpy(clip_feats).to(args.device).float()
        clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True)
        if args.finetune_clip:
            if args.map_code_base == 'B_MLPS':
                clip_feats, _, _ = map_model(clip_feats, clip_feats)
        # Other different conditional feature coding strategy: positional encoding, duplicate...
        # This part of the code is crappy.
        sin_img_clip_feats = torch.sin(clip_feats * torch.tensor(np.pi))
        cos_img_clip_feats = torch.cos(clip_feats * torch.tensor(np.pi))
        sin2_img_clip_feats = torch.sin(2 * clip_feats * torch.tensor(np.pi))
        cos2_img_clip_feats = torch.cos(2 * clip_feats * torch.tensor(np.pi))
        # clip_feats = torch.cat([sin_img_clip_feats, cos_img_clip_feats, sin2_img_clip_feats, cos2_img_clip_feats], dim=-1)
    
    return clip_feats, clip_name_array

def load_clip_txt_feats(args, map_model, load_valid_cates_only=False):
    if args.split_by_text:
        split_file = os.path.join(args.shape_gt_path, 'val' + '.elst')
    else:
        split_file = os.path.join(args.shape_gt_path, 'val' + '.lst')
    with open(split_file, 'r') as f:
        model_names = f.read().split('\n')

    with open(args.valid_cate_anno_path + '_all.lst', 'r') as f:
        all_valid_full_names = f.read().split('\n')

    with open(args.clip_txt_feat_path, 'rb') as handle:
        feats = pickle.load(handle)

    text_features_list = []
    text_feats_name_list = []
    text_str_list = []

    if args.split_by_text:
        for m_name in list(feats.keys()):
            t_feats_dict = feats[m_name]
            for t_key in list(t_feats_dict.keys()):
                if t_key in model_names:
                    if load_valid_cates_only:
                        if m_name not in all_valid_full_names:
                            continue
                    text_feat = t_feats_dict[t_key]
                    # text_feat = torch.from_numpy(text_feat).to(args.device).float()
                    text_feat = torch.from_numpy(text_feat).to(args.device).float()
                    text_feat = text_feat.view(1, -1)
                    text_feat /= text_feat.norm(dim=-1, keepdim=True)
                    
                    if args.finetune_clip:
                        if args.map_code_base == 'B_MLPS':
                            _,  text_feat, _ = map_model(text_feat, text_feat)
                        else:
                             _, text_feat, _ = map_model(text_feat, text_feat)
                    text_feat = text_feat.to(args.device)
                    text_features_list.append(text_feat)
                    text_feats_name_list.append(m_name)
                    text_str_list.append(t_key.replace('.', '_'))
    else:
        for m_name in model_names:
            if m_name in list(feats.keys()):
                if load_valid_cates_only:
                    if m_name not in all_valid_full_names:
                        continue
                t_feats_dict = feats[m_name]
                for t_key in t_feats_dict:
                    text_feat = t_feats_dict[t_key]
                    # text_feat = torch.from_numpy(text_feat).to(args.device).float()
                    text_feat = torch.from_numpy(text_feat).to(args.device).float()
                    text_feat = text_feat.view(1, -1)
                    text_feat /= text_feat.norm(dim=-1, keepdim=True)
                    
                    if args.finetune_clip:
                        if args.map_code_base == 'B_MLPS':
                            _,  text_feat, _ = map_model(text_feat, text_feat)
                        else:
                            _, text_feat, _ = map_model(text_feat, text_feat)
                    text_feat = text_feat.to(args.device)
                    text_features_list.append(text_feat)
                    text_feats_name_list.append(m_name)
                    text_str_list.append(t_key.replace('.', '_'))
    return text_features_list, text_feats_name_list, text_str_list

def load_shape_embed_feats(args, split='val', load_valid_cates_only=False):
    if load_valid_cates_only:
        with open(args.valid_cate_anno_path + '_all.lst', 'r') as f:
            all_valid_full_names = f.read().split('\n')
    else:
        all_valid_full_names = []

    with open(args.embed_feat_path, 'rb') as handle:
        embed_trn_feats_dict = pickle.load(handle) 
        embed_trn_key_list = [key for key in list(embed_trn_feats_dict) if (not load_valid_cates_only) or (key in all_valid_full_names)]
        embed_trn_feats_array = [embed_trn_feats_dict[key] for key in embed_trn_feats_dict if (not load_valid_cates_only) or (key in all_valid_full_names)]  
    embed_trn_feats = np.stack(embed_trn_feats_array, axis=0)
    embed_trn_feats = torch.from_numpy(embed_trn_feats).to(args.device)

    with open(args.embed_feat_path.replace('embed_feats', 'embed_feats_test'), 'rb') as handle:
        embed_tst_feats_dict = pickle.load(handle) 
        embed_tst_key_list = [key for key in list(embed_tst_feats_dict) if (not load_valid_cates_only) or (key in all_valid_full_names)]  
        embed_tst_feats_array = [embed_tst_feats_dict[key] for key in embed_tst_feats_dict if (not load_valid_cates_only) or (key in all_valid_full_names)]  
    embed_tst_feats = np.stack(embed_tst_feats_array, axis=0)
    embed_tst_feats = torch.from_numpy(embed_tst_feats).to(args.device)

    with open(args.embed_feat_path.replace('embed_feats', 'embed_feats_val'), 'rb') as handle:
        embed_val_feats_dict = pickle.load(handle) 
        embed_val_key_list = [key for key in list(embed_val_feats_dict) if (not load_valid_cates_only) or (key in all_valid_full_names)]  
        embed_val_feats_array = [embed_val_feats_dict[key] for key in embed_val_feats_dict if (not load_valid_cates_only) or (key in all_valid_full_names)]  
    embed_val_feats = np.stack(embed_val_feats_array, axis=0)
    embed_val_feats = torch.from_numpy(embed_val_feats).to(args.device)

    if split == 'train':
        return embed_trn_key_list, embed_trn_feats_dict, embed_trn_feats
    elif split == 'test':
        return embed_tst_key_list, embed_tst_feats_dict, embed_tst_feats
    elif split == 'val':
        return embed_val_key_list, embed_val_feats_dict, embed_val_feats
    elif split == 'all':
        embed_all_key_list = embed_trn_key_list + embed_tst_key_list + embed_val_key_list
        embed_all_feats_dict = {}
        for d in [embed_trn_feats_dict, embed_tst_feats_dict, embed_val_feats_dict]:
            embed_all_feats_dict.update(d)
        embed_all_feats = torch.cat((embed_trn_feats, embed_tst_feats, embed_val_feats), dim=0)
        return embed_all_key_list, embed_all_feats_dict, embed_all_feats
