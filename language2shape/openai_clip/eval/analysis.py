import os
import shutil
import random
import trimesh
import numpy as np
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

from eval_r_precision import *
from load_data import get_shapenet_full_names, load_clip_feats, load_shape_embed_feats
from occupancy_networks.generate_from_z import ocnet_generation_from_z

def retrieve_in_clip(args, is_text, map_model=None, text_features_list=None, text_str_list=None, text_feats_name_list=None, out_mesh_folder='./', test_only=False, max_vis_nam=60):
    # Retrieve CLIP Feature
    full_names = get_shapenet_full_names(args)
    clip_feats, clip_name_array = load_clip_feats(args, map_model, load_valid_cates_only= not args.use_text2shape and is_text, test_only=test_only)
    
    clip_r_precision = 0
    clip_r_precision5 = 0

    if is_text:    
        if args.quantity:
            all_precision = []  
            all_precision5 = []  

        for t_idx, (text_feat, text_str, text_feats_name) in enumerate(zip(text_features_list, text_str_list, text_feats_name_list)):
            # sort
            # distance = torch.cdist(clip_feats, text_feat).squeeze()
            # idx_sort = torch.argsort(distance)
            similarity = (text_feat @ clip_feats.T).squeeze()
            # print(clip_feats[0].norm(dim=-1, keepdim=True))
            # print(text_feat.norm(dim=-1, keepdim=True))
            # print(similarity)
            # exit(0)
            idx_sort = torch.argsort(similarity, descending=True)
            # print(clip_feats, text_feat)
            # print(similarity[idx_sort])
            # exit(0)
            for idx_rtv in range(args.n_retrieval_clip):
                # copy rtv results
                clip_name = clip_name_array[idx_sort[idx_rtv]]
                clip_model_name = clip_name.split('@')[0]
                clip_img_name = int(clip_name.split('@')[-1])
                full_name = [_ for _ in full_names if _.startswith(clip_model_name)][0]
                if args.quantity:
                    if args.use_text2shape:
                        precision, precision5 = precision_of_text2shape(text_feats_name, clip_name_array, idx_sort, full_names, rtv_num=10)
                    else:
                        precision = precision_by_txt(args, text_str, full_name)
                        precision5 = None
                    precision_str = 'True' if precision else 'False'
                    if precision is not None:
                        all_precision.append(precision)
                    if precision5 is not None:
                        all_precision5.append(precision5)
                        
                if args.quality:
                    if t_idx <= max_vis_nam:
                        if args.use_text2shape:
                            gt_img_read_path = os.path.join(args.img_gt_path, text_feats_name.split('@')[0] + '.png')
                            gt_img_write_path = os.path.join(out_mesh_folder, '{}_gt_img_{}.jpg'.format(text_str.replace(' ', '_'), str(idx_rtv).zfill(2)))
                            try:
                                shutil.copy2(gt_img_read_path, gt_img_write_path)
                            except OSError as oserr:
                                if oserr.errno == 36:
                                    pass
                        gt_read_path = os.path.join(args.shape_gt_path, full_name, 'model.binvox')
                        gt_write_path = os.path.join(out_mesh_folder, '{}_clip_rtv{}_shape.off'.format(text_str.replace(' ', '_'), str(idx_rtv).zfill(2)))
                        with open(gt_read_path, 'rb') as fp:
                            vox = trimesh.exchange.binvox.load_binvox(fp)
                        vox_mesh = vox.as_boxes()
                        try:
                            vox_mesh.export(gt_write_path)
                        except OSError as oserr:
                            if oserr.errno == 36:
                                pass
                        gt_img_read_path = os.path.join(args.img_gt_path, clip_model_name + '.png')
                        gt_img_write_path = os.path.join(out_mesh_folder, '{}_clip_rtv{}_img_{}.jpg'.format(text_str.replace(' ', '_'), str(idx_rtv).zfill(2), precision_str))
                        try:
                            shutil.copy2(gt_img_read_path, gt_img_write_path)
                        except OSError as oserr:
                            if oserr.errno == 36:
                                pass
        if args.quantity:
            if len(all_precision) > 0:
                clip_r_precision = sum(all_precision) / len(all_precision)
            if len(all_precision5) > 0:
                clip_r_precision5 = sum(all_precision5) / len(all_precision5)
    else:
        for text_str in text_str_list:
            if args.quality:
                # Retrivel from CLIP==GT
                try:
                    gt_img_read_path = os.path.join(args.img_gt_path, text_str + '.png')
                    gt_img_write_path = os.path.join(out_mesh_folder, '{}_clip_rtv{}_img.jpg'.format(text_str, str(0).zfill(2)))
                    shutil.copy2(gt_img_read_path, gt_img_write_path)

                    full_name = [_ for _ in full_names if _.startswith(text_str)][0]
                    gt_read_path = os.path.join(args.shape_gt_path, full_name, 'model.binvox')
                    gt_write_path = os.path.join(out_mesh_folder, '{}_clip_rtv{}_shape.off'.format(text_str, str(0).zfill(2)))
                    with open(gt_read_path, 'rb') as fp:
                        vox = trimesh.exchange.binvox.load_binvox(fp)
                    vox_mesh = vox.as_boxes()
                    vox_mesh.export(gt_write_path)
                except OSError as oserr:
                    if oserr.errno == 36:
                        pass
    return clip_r_precision, clip_r_precision5

def retrieve_in_shape_embed(args, is_text, shape_features_list, text_str_list, text_feats_name_list, test_only=False, num_max_vis=60):
    # Retrieval Shape Embedding
    if test_only:
        embed_key_list, _, embed_feats = load_shape_embed_feats(args, split='val', load_valid_cates_only= not args.use_text2shape)
        # For inference baseline(text-conditioned generation) only
        # _, embed_key_list, _, _, _, embed_feats = load_shape_embed_feats(args, split='val', load_valid_cates_only= True)
    else:
        embed_key_list, _, embed_feats = load_shape_embed_feats(args, split='all', load_valid_cates_only= not args.use_text2shape)
        
    full_names = get_shapenet_full_names(args)
    
    shape_r_precision = 0
    shape_r_precision5 = 0
    if args.quantity:
        all_precision = []  
        all_precision5 = []  
    top_shape_id_list = []

    for s_idx, (shape_features, text_str, text_feats_name) in enumerate(zip(shape_features_list, text_str_list, text_feats_name_list)):
        instance_top_shape_id_list = []
        n_instance = shape_features.shape[0]
        for idx_inst in range(n_instance):
            #sort and retrieve
            inst_feature = shape_features[idx_inst].unsqueeze(0)
            distance = torch.cdist(embed_feats, inst_feature).squeeze()
            idx_sort = torch.argsort(distance)
            # similarity = (embed_feats @ inst_feature.T).squeeze()
            # idx_sort = torch.argsort(similarity, descending=True)
            
            # copy rtv results
            for idx_rtv in range(args.n_retrieval_shape):
                rtv_feature = embed_feats[idx_sort[idx_rtv], :].unsqueeze(0)
                key = embed_key_list[idx_sort[idx_rtv]]
                if idx_rtv == 0:
                    instance_top_shape_id_list.append(key.split('@')[0])
                    
                if args.quantity:
                    if is_text:
                        if args.use_text2shape:
                            precision, precision5 = precision_of_text2shape(text_feats_name, embed_key_list, idx_sort, full_names, rtv_num=10)
                        else:
                            precision = precision_by_txt(args, text_str, key)
                            precision5 = None
                    else:
                        full_name = [_ for _ in full_names if _.startswith(text_str)][0]
                        precision = precision_by_idx(args, full_name, key)
                        precision5 = None
                    if idx_rtv==0 and idx_inst==0:
                        if precision is not None:
                            all_precision.append(precision)
                        if precision5 is not None:
                            all_precision5.append(precision5)  

                if args.quality:
                    pass
                    # if s_idx <= num_max_vis:
                    #     try:
                    #         vis = args.quality or args.quant_clip
                    #         out_mesh_folder = ocnet_generation_from_z(args, rtv_feature, list(key.split('@')), args.out_mesh_folder, '{}_{}_rtv{}'.format(text_str, str(idx_inst).zfill(2), str(idx_rtv).zfill(2)), args.device, n_row=1, vis=vis)
                    #         gt_read_path = os.path.join(args.shape_gt_path, key, 'model.binvox')
                    #         gt_write_path = os.path.join(out_mesh_folder, '{}_{}_gt{}.off'.format(text_str.replace(' ', '_'), str(idx_inst).zfill(2), str(idx_rtv).zfill(2)))
                    #         with open(gt_read_path, 'rb') as fp:
                    #             vox = trimesh.exchange.binvox.load_binvox(fp)
                    #         vox_mesh = vox.as_boxes()
                    #         vox_mesh.export(gt_write_path)
                    #     except OSError as oserr:
                    #         if oserr.errno == 36:
                    #             pass
        top_shape_id_list.append(instance_top_shape_id_list)
        
    if args.quantity:
        if len(all_precision) > 0:
            shape_r_precision = sum(all_precision) / len(all_precision)
        if len(all_precision5) > 0:
            shape_r_precision5 = sum(all_precision5) / len(all_precision5)

    return shape_r_precision, shape_r_precision5, all_precision, all_precision5, top_shape_id_list

## Archived Functions
def plot_tsne(dist_shape_feats, shape_feats1, shape_feats2, save_name='shape_embed_tsn_img'):
    # tsne 
    feats = np.vstack([dist_shape_feats, shape_feats1, shape_feats2])
    X = TSNE(n_components=2, learning_rate='auto',
                        init='pca').fit_transform(feats)
    X_i = X[: -2, :]
    X_1 = X[-2, :].reshape(1, -1)
    X_2 = X[-1, :].reshape(1, -1)

    plt.figure()
    plt.scatter(X_i[:, 0], X_i[:, 1], color="navy", alpha=0.8, label='all')
    plt.scatter(X_1[:, 0], X_1[:, 1], color="darkorange", alpha=0.8, label='rtv')
    plt.scatter(X_2[:, 0], X_2[:, 1], color="cyan", alpha=0.8, label='gen')
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("Shape Embedding of Instance")
    save_image_path = os.path.join('./utils/pca_results', save_name)
    plt.savefig(save_image_path, format="png", dpi=1200 )
    plt.clf()

def visualize_shape_embed(args, is_text, map_model, nf_model, mlps, generate_conditioned, shape_features_list, text_features_list, text_str_list):
    # Shape Embedding Visualization
    clip_feats, clip_name_array = load_clip_feats(args, map_model)
    full_names = get_shapenet_full_names(args)
    embed_trn_key_list, embed_tst_key_list, embed_trn_feats_dict, embed_tst_feats_dict, _, _ = load_shape_embed_feats(args)
    
    # build shape embed feats distribution
    rand1 = random.choices(range(0, len(clip_name_array)), k=2000)
    dist_shape_feats_array = []
    for r in rand1:
        clip_feat_name = clip_name_array[r]
        clip_model_name = clip_feat_name.split('@')[0]
        full_name = [_ for _ in full_names if _.startswith(clip_model_name)][0]
        if full_name in embed_trn_key_list:
            dist_shape_feats_array.append(embed_trn_feats_dict[full_name].reshape(1, -1))
        elif full_name in embed_tst_key_list:
            dist_shape_feats_array.append(embed_tst_feats_dict[full_name].reshape(1, -1))
    dist_shape_feats = np.vstack(dist_shape_feats_array)

    if is_text:
        # visualize shape embed of clip txt feats instances
        for shape_features, text_feat, text_str in zip(shape_features_list, text_features_list, text_str_list):  
            similarity = (clip_feats @ text_feat.T).squeeze()
            idx_sort = torch.argsort(similarity, descending=True)
            for idx_rtv in range(args.n_retrieval_clip):
                clip_name = clip_name_array[idx_sort[idx_rtv]]
                clip_model_name = clip_name.split('@')[0]
                full_name = [_ for _ in full_names if _.startswith(clip_model_name)][0]
                if full_name in embed_trn_key_list:
                    shape_feats1 = embed_trn_feats_dict[full_name].reshape(1, -1)
                elif full_name in embed_tst_key_list:
                    shape_feats1 = embed_tst_feats_dict[full_name].reshape(1, -1)
                if full_name in embed_trn_key_list+embed_tst_key_list:    
                    shape_feats2 = shape_features.cpu().numpy().reshape(1, -1)
                    plot_tsne(dist_shape_feats, shape_feats1, shape_feats2, save_name='shape_embed_tsn_txt_{}.png'.format(text_str))

    else:
        # visualize shape embed of clip img feats instances
        rand2 = random.choices(range(0, len(clip_name_array)), k=10)
        for r_idx, r in enumerate(rand2):
            clip_feat_name = clip_name_array[r]
            clip_model_name = clip_feat_name.split('@')[0]
            full_name = [_ for _ in full_names if _.startswith(clip_model_name)][0]
            if full_name in embed_trn_key_list:
                shape_feats1 = embed_trn_feats_dict[full_name].reshape(1, -1)
            elif full_name in embed_tst_key_list:
                shape_feats1 = embed_tst_feats_dict[full_name].reshape(1, -1)
            if full_name in embed_trn_key_list+embed_tst_key_list: 
                mlps.eval().to(args.device)
                input_ = mlps(clip_feats[r].reshape(1, -1))   
                shape_feats2 =  generate_conditioned(nf_model,input_, args.device, n_row=1, out_dim=args.input_size).cpu().numpy()
                plot_tsne(dist_shape_feats, shape_feats1, shape_feats2, save_name='shape_embed_tsn_img_{}.png'.format(r_idx) )


