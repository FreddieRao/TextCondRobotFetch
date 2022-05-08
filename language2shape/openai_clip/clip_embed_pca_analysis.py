import numpy as np
import torch
import pickle
import clip
import os
from PIL import Image
from numpy import linalg as LA
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
from finetune.mlp_networks import F_MappingNetwork, B_MappingNetwork

clip_txt_feat_path = '../../data/embed_ShapeNetCore.v1/clip_txt_feat_text2shape.pickle'
shape_gt_path = "occupancy_networks/data/ShapeNet/03001627"
embed_feat_path = "occupancy_networks/out/voxels/onet_chair/generation/embed/03001627/embed_feats_test.pickle"
full_names = [ _ for _ in os.listdir(shape_gt_path) if os.path.isdir(os.path.join(shape_gt_path, _))]
split_file = os.path.join(shape_gt_path, 'val' + '.lst')
with open(split_file, 'r') as f:
    val_model_names = f.read().split('\n')

templates80 = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

templates29 = [    
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a photo of a hard to see {}.',
    'a photo of a clean {}.',
    'a photo of my {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a cropped photo of a {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a photo of one {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'a low resolution photo of a {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a good photo of a {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'a photo of the large {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
]

no_template = ['This is a chair. A {}.']
gray_template = 'gray {}'

def NormalizeTorchData(data):
    return (data - torch.min(data, dim=-1).values) / (torch.max(data, dim=-1).values - torch.min(data, dim=-1).values)

def NormalizeNPData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def load_data(clip_img_feat_path, return_txt=False, view_name=False, finetune_clip=False, map_restore_file1='./', map_restore_file2='./'):
    # # ====================Dataset prepare : Clip images
    if finetune_clip:
        map_model1, map_model2 = restore_nf_model(map_restore_file1, map_restore_file2)
    with open(clip_img_feat_path, 'rb') as handle:
        clip_img_feats = pickle.load(handle)
        clip_feats_array = []
        clip_names_array = []
        for key in clip_img_feats:
            if not view_name:
                clip_names_array.append(key)
            
            for idx in clip_img_feats[key]:
                img_feat =clip_img_feats[key][idx].astype(np.half)
                if finetune_clip:
                    img_feat = torch.from_numpy(img_feat).cuda().float()
                    img_feat = img_feat.view(1, -1)
                    img_feat, _, _ = map_model1(img_feat, img_feat)
                    img_feat, _, _ = map_model2(img_feat, img_feat)
                    img_feat = img_feat.view(-1)
                    img_feat = img_feat.detach().cpu().numpy()
                clip_feats_array.append(img_feat)
                if view_name:
                    clip_names_array.append(key + '@' + idx)
        clip_feats = np.stack(clip_feats_array, axis=0)
        clip_feats /= LA.norm(clip_feats, axis=-1, keepdims=True)
        # clip_feats = NormalizeNPData(clip_feats)
    clip_feats = clip_feats#[:1000, :]
    print(clip_feats.shape)
    clip_feats_nvh = 0

    # # ====================Dataset: Shape Embed
    # with open(embed_feat_path, 'rb') as handle:
    #     embed_trn_feats_dict  = pickle.load(handle) 
    #     embed_trn_key_list = [key[:-4] for key in list(embed_trn_feats_dict)]
    #     embed_trn_feats_array = [embed_trn_feats_dict[key] for key in embed_trn_feats_dict]  
    # embed_trn_feats = np.stack(embed_trn_feats_array, axis=0)
    
    # clip_feats = embed_trn_feats
    # clip_names_array = embed_trn_key_list

    if return_txt:

        # #====================Dataset : Text2Shape
        with open(clip_txt_feat_path, 'rb') as handle:
            clip_txt_feats = pickle.load(handle)
            clip_txt_feats_array = []
            clip_txt_names_array = []
            for key in clip_txt_feats:     
                if key in val_model_names:      # visualize valid set only 
                    for idx in clip_txt_feats[key]:
                        text_feat = clip_txt_feats[key][idx]
                        if finetune_clip:
                                text_feat = torch.from_numpy(text_feat).cuda().float()
                                text_feat = text_feat.view(1, -1)
                                _, text_feat, _ = map_model1(text_feat, text_feat)
                                _, text_feat, _ = map_model2(text_feat, text_feat)
                                text_feat = text_feat.view(-1)
                                text_feat = text_feat.detach().cpu().numpy()

                        clip_txt_feats_array.append(text_feat)
                        clip_txt_names_array.append(key + '@' + idx)
            clip_txt_feats = np.stack(clip_txt_feats_array, axis=0)
            clip_txt_feats /= LA.norm(clip_txt_feats, axis=-1, keepdims=True)
            # clip_feats = NormalizeNPData(clip_feats)
        print(clip_txt_feats.shape)

        # #====================Dataset : Text Cate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.eval().requires_grad_(False)
        text = "ball chair;cantilever chair;armchair;chair;tulip chair;straight chair;side chair;club chair;swivel chair;easy chair;lounge chair;overstuffed chair;barcelona chair;chaise longue;chaise;daybed;deck chair;beach chair;folding chair;bean chair;butterfly chair;rocking chair;rocker;zigzag chair;recliner;reclining chair;lounger;lawn chair;garden chair;Eames chair;sofa;couch;lounge;rex chair;camp chair;X chair;Morris chair;NO. 14 chair;park bench;table;wassily chair;Windsor chair;love seat;loveseat;tete-a-tete;vis-a-vis;wheelchair;bench;wing chair;ladder-back;ladder-back chair;throne;double couch;settee"
        raw_text_str_list = [item for item in text.split(';')]
        text_str_list = [template.format(gray_template.format(txt)) for txt in raw_text_str_list for template in no_template ]

        text_features_list = []
        for text_str in text_str_list:
            text_token = clip.tokenize([text_str]).to(device)
            with torch.no_grad():
                text_feat = model.encode_text(text_token)#.float()
                text_feat /= text_feat.norm(dim=-1, keepdim=True)
                # text_feat = NormalizeTorchData(text_feat)
                text_features_list.append(text_feat.cpu().numpy())
        text_feats = np.concatenate(text_features_list, axis=0)
        print(text_feats.shape)
        # all_feats = np.concatenate([clip_feats, text_feats], axis=0)
        
    else:
        text_str_list, text_features_list, text_feats, all_feats = 0, 0, 0, 0
    
    all_feats = clip_txt_feats
    return clip_names_array, clip_feats, clip_feats_nvh, text_str_list, text_features_list, text_feats, all_feats


def restore_nf_model(map_restore_file1, map_restore_file2):
    if map_restore_file1 == './':
        map_model1 = torch.nn.Identity()
    else:
        map_model1 = B_MappingNetwork().cuda()
        state = torch.load(map_restore_file1, map_location='cuda')
        map_model1.load_state_dict(state['model_state'])
        map_model1.eval()
    if map_restore_file2 == './':
        map_model2 = torch.nn.Identity()
    else:
        map_model2 = F_MappingNetwork().cuda()
        state = torch.load(map_restore_file2, map_location='cuda')
        map_model2.load_state_dict(state['model_state'])
        map_model2.eval()
    return map_model1, map_model2

def clip_space_analysis(clip_feats, clip_feats_nvh, text_feats, all_feats):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_feats_nvh = torch.from_numpy(clip_feats_nvh).to(device).float()
    v_dist = torch.cdist(clip_feats_nvh, clip_feats_nvh).squeeze()
    v_dist_mean = torch.mean(v_dist)

    n_dist = torch.cdist(clip_feats_nvh[:, 0, :].squeeze(), clip_feats_nvh[:, 0, :].squeeze()).squeeze()
    n_dist_mean = torch.mean(n_dist)

    print('Mean Distance between Views is {}\n Mean Distance between Instances is {}'.format(v_dist_mean, n_dist_mean))
    print('Mean Cosine Similarity between Views is {}\n Mean Cosine Similarity between Instances is {}'.format(v_cdist_mean, n_cdist_mean))

# PCA Visualization of CLIP Embedding Space
def pca_visualize_instances(clip_names_array, clip_feats, clip_feats_nvh, text_str_list, text_features_list, text_feats, all_feats):
    pca = PCA(n_components=2)
    X_i = pca.fit(clip_feats).transform(clip_feats)
    X_t = pca.transform(text_feats)
    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    # Most values in the components_ are zero (sparsity)
    print(
        "explained variance ratio (first two components): %s"
        % str(np.mean(transformer.components_ == 0))
    )

    idx_rand = np.random.randint(clip_feats.shape[0], size= clip_feats.shape[0]//100)
    
    for idx, name in enumerate(clip_names_array):    
        plt.figure()  
        lw = 2
        plt.scatter(
            X_i[idx_rand, 0], X_i[idx_rand, 1], color="navy", alpha=0.8, label='image'
        )

        plt.scatter(
            X_t[:, 0], X_t[:, 1], color="darkorange", alpha=0.8, label='text'
        )

        for i in range(len(text_features_list)):
            plt.text(X_t[i, 0], X_t[i, 1], text_str_list[i], fontsize=1)

        plt.scatter(
            X_i[idx * 24: idx * 24 + 24, 0], X_i[idx * 24: idx * 24 + 24, 1], 
            color="cyan", alpha=0.8, label='instance'
        )

        plt.title("PCA of CLIP Embedding" + name)
        save_image_path = './shapenet_img/chair/' + name + '_pca_analysis.png'
        print(idx)
        plt.savefig(save_image_path, format="png", dpi=480)
        plt.close()

# PCA Visualization of CLIP Embedding Space
def pca_visualize(clip_feats, clip_feats_nvh, text_feats, all_feats, save_image_path):
    pca = SparsePCA(n_components=10)
    X_i = pca.fit(clip_feats).transform(clip_feats)
    X_t = pca.transform(text_feats)
    # Percentage of variance explained for each components
    # print(
    #     "explained variance ratio (first two components): %s"
    #     % str(pca.explained_variance_ratio_)
    # )

    # Most values in the components_ are zero (sparsity)
    print(
        "explained variance ratio (first two components): %s"
        % str(np.mean(pca.components_ == 0))
    )

    plt.figure()
    lw = 2
    plt.scatter(
        X_i[:, 0], X_i[:, 1], color="navy", alpha=0.8, label='image'
    )

    plt.scatter(
        X_t[:, 0], X_t[:, 1], color="darkorange", alpha=0.8, label='text'
    )

    for i in range(len(text_features_list)):
        plt.text(X_t[i, 0], X_t[i, 1], text_str_list[i], fontsize=1)

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("Sparse PCA of CLIP Embedding")
    plt.savefig(save_image_path, format="png", dpi=1200 )
    plt.clf()


# TSNE Visualization of CLIP Embedding Space
def tsn_visualize(clip_feats, clip_feats_nvh, text_str_list, text_feats, all_feats, save_image_path):
    tsne = TSNE(n_components=2, init='pca', n_iter=1000, learning_rate='auto', metric='cosine', verbose=True)
    draw_feats = np.concatenate([clip_feats, all_feats], axis=0)
    X_i = tsne.fit_transform(draw_feats)
    # Figure1
    plt.figure()
    lw = 2
    plt.scatter(
        X_i[clip_feats.shape[0]:, 0], X_i[clip_feats.shape[0]:, 1], color="darkorange", alpha=0.8, label='free-form'
    )
    plt.scatter(
        X_i[:clip_feats.shape[0], 0], X_i[:clip_feats.shape[0], 1], color="navy", alpha=0.8, label='images'
    )
    # for i in range(len(text_features_list)):
    #     plt.text(X_i[clip_feats.shape[0]+i, 0], X_i[clip_feats.shape[0]+i, 1], text_str_list[i], fontsize=1)

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("TSNE of CLIP Embedding")
    plt.savefig(save_image_path + '_1.png', format="png", dpi=1200 )
    plt.clf()
    # Figure2
    plt.figure()
    lw = 2

    plt.scatter(
        X_i[:clip_feats.shape[0], 0], X_i[:clip_feats.shape[0], 1], color="navy", alpha=0.8, label='images'
    )
    plt.scatter(
        X_i[clip_feats.shape[0]:, 0], X_i[clip_feats.shape[0]:, 1], color="darkorange", alpha=0.8, label='free-form'
    )
    # for i in range(len(text_features_list)):
    #     plt.text(X_i[clip_feats.shape[0]+i, 0], X_i[clip_feats.shape[0]+i, 1], text_str_list[i], fontsize=1)

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("TSNE of CLIP Embedding")
    plt.savefig(save_image_path + '_2.png', format="png", dpi=1200 )
    plt.clf()

# TSNE Visualization of CLIP Embedding Space
def tsn_visualize_txt(clip_feats, clip_feats_nvh, text_str_list, text_feats, all_feats, save_image_path):
    tsne = TSNE(n_components=2, init='pca', n_iter=1000, learning_rate='auto', metric='cosine', verbose=True)
    X_i = tsne.fit_transform(text_feats)

    plt.figure()
    lw = 2

    colors = []
    for c in range(80):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)  
        colors.append(color)
    
    for i in range(text_feats.shape[0]):
        c = colors[int(i / 29)] 
        # c = colors[i % 29] 
        plt.scatter(
            X_i[i, 0], X_i[i, 1], color=c, alpha=0.8
        )

    for i in range(len(text_features_list)):
        if i % 7 == 0:
            plt.text(X_i[i, 0], X_i[i, 1], text_str_list[i], fontsize=2)

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("TSNE of CLIP Embedding: Colored by class in gray+tempalte29")
    plt.savefig(save_image_path, format="png", dpi=1200 )

# TSNE Visualization of CLIP Embedding Space
def tsn_analysis_txt(clip_feats, clip_feats_nvh, text_str_list, text_feats, all_feats, save_image_path):
    tsne = TSNE(n_components=2, init='pca', n_iter=1000, learning_rate='auto', metric='cosine', verbose=True)
    X_i = tsne.fit_transform(text_feats)
    
    X_ir = X_i.reshape((80, 54, 2))
    std_list = []
    mean_list = []
    for i in range(X_ir.shape[0]):
        x = X_ir[i, :]
        std = np.std(x, axis=0)
        mean = np.std(x, axis=0)
        std_list.append(std)
        mean_list.append(mean)

    count = [0] * 80
    for i in range(text_feats.shape[0]):
        std = std_list[int(i / 80)] 
        mean = mean_list[int(i / 80)] 
        if np.any(np.abs(X_i[i] - mean) < std ):
            count[int(i / 80)] += 1
    print(count)

    colors = []
    for c in range(80):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)  
        colors.append(color)
    
    total = 0
    for i in range(text_feats.shape[0]):
        c = colors[int(i / 80)] 
        co = count[int(i / 80)] 
        
        if co < 27:
            total += 1
            plt.scatter(
                X_i[i, 0], X_i[i, 1], color=c, alpha=0.4
            )
            if i % 7 == 0:
                plt.text(X_i[i, 0], X_i[i, 1], text_str_list[i], fontsize=2)

    print(total)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("TSNE of CLIP Embedding: Colored by template in autotemplate")
    plt.savefig(save_image_path, format="png", dpi=1200 )


# TSNE Visualization of CLIP Embedding Space
def tsn_visualize_paired(clip_feats1, clip_feats2, text_str_list, text_feats, all_feats, save_image_path):
    tsne = TSNE(n_components=2, init='pca', n_iter=1000, learning_rate='auto', metric='cosine', verbose=True)
    all_feats = np.concatenate([clip_feats1, clip_feats2, text_feats], axis=0)
    X_i = tsne.fit_transform(all_feats)
    plt.figure()
    lw = 2
    plt.scatter(
        X_i[clip_feats1.shape[0]:clip_feats1.shape[0]+clip_feats2.shape[0], 0], X_i[clip_feats1.shape[0]:clip_feats1.shape[0]+clip_feats2.shape[0], 1], color="cyan", alpha=0.8, label='texture'
    )

    plt.scatter(
        X_i[:clip_feats1.shape[0], 0], X_i[:clip_feats1.shape[0], 1], color="navy", alpha=0.4, label='shade'
    )

    plt.scatter(
        X_i[clip_feats1.shape[0]+clip_feats2.shape[0]:, 0], X_i[clip_feats1.shape[0]+clip_feats2.shape[0]:, 1], color="darkorange", alpha=0.8, label='text'
    )

    for i in range(len(text_features_list)):
        plt.text(X_i[clip_feats1.shape[0]+clip_feats2.shape[0]+i, 0], X_i[clip_feats1.shape[0]+clip_feats2.shape[0]+i, 1], text_str_list[i], fontsize=1)

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("TSNE of CLIP Embedding")
    plt.savefig(save_image_path, format="png", dpi=1200 )

# TSNE Visualization of CLIP Embedding Space
def tsn_visualize_img(clip_names_array, clip_feats, clip_feats_nvh, text_str_list, text_feats, all_feats, save_image_path):
    tsne = TSNE(n_components=2, init='pca', n_iter=1000, learning_rate='auto', metric='cosine', verbose=True)
    X_i = tsne.fit_transform(clip_feats)

    fig, ax = plt.subplots()
    lw = 2
    ax.scatter(
        X_i[:, 0], X_i[:, 1], color="white", alpha=0.1, label='image'
    )
    
    
    for i, name in enumerate(clip_names_array):
        if i % 5 == 0:
            path = os.path.join('../../data/render_ShapeNetCore.v1/1_view_shade', name + '.png')
            img = mpimg.imread(path)
            imagebox = OffsetImage(img, zoom=0.02, alpha=0.6)
            ab = AnnotationBbox(imagebox, (X_i[i, 0], X_i[i, 1]), pad=0.01, bboxprops=dict(facecolor = "none", lw = 0))
            ax.add_artist(ab)

    fig.savefig(save_image_path, format="png", dpi=1200 )

# TSNE Visualization of CLIP Embedding Space
def tsn_visualize_paired_img(clip_names_array, clip_feats1, clip_feats2, text_str_list, text_feats, all_feats, save_image_path):
    all_feats = np.concatenate([clip_feats1, clip_feats2], axis=0)
    tsne = TSNE(n_components=2, init='pca', n_iter=1000, learning_rate='auto', metric='cosine', verbose=True)
    X_i = tsne.fit_transform(all_feats)

    fig, ax = plt.subplots()
    lw = 2
    ax.scatter(
        X_i[:, 0], X_i[:, 1], color="white", alpha=0.1, label='image'
    )
    
    
    for i, name in enumerate(clip_names_array):
        if i % 10 == 0:
            path1 = os.path.join('../../data/render_ShapeNetCore.v1/1_view_shade', name + '.png')
            img1 = mpimg.imread(path1)
            imagebox1 = OffsetImage(img1, zoom=0.020, alpha=0.6)
            ab1 = AnnotationBbox(imagebox1, (X_i[i, 0], X_i[i, 1]), pad=0.01, bboxprops=dict(facecolor = "none", lw = 0))
            ax.add_artist(ab1)

            full_name = [_ for _ in full_names if _.startswith(name.split('@')[0])][0]
            path2 = os.path.join('../../data/render_ShapeNetCore.v1/1_view_texture_keyshot_1sec', full_name + '.png')
            img2 = mpimg.imread(path2)
            img2_ = np.ones((img2.shape[0], img2.shape[1], 4))
            img2_[:, :, :3] = img2
            for h in range(img2_.shape[0]):
                for w in range(img2_.shape[1]):
                    if np.equal(img2_[h, w, :], [1, 1, 1, 1]).all():
                        img2_[h, w, 3] = 0
            imagebox2 = OffsetImage(img2_, zoom=0.045, alpha=0.6)
            ab2 = AnnotationBbox(imagebox2, (X_i[clip_feats1.shape[0] + i, 0], X_i[clip_feats1.shape[0] + i, 1]), pad=0.01, bboxprops=dict(facecolor = "none", lw = 0))
            ax.add_artist(ab2)

    fig.savefig(save_image_path, format="png", dpi=1200 )


# TSNE Visualization of CLIP Embedding Space
def tsn_visualize_img_multiview(clip_names_array, clip_feats, clip_feats_nvh, save_image_path):
    tsne = TSNE(n_components=2, init='pca', n_iter=1000, learning_rate='auto', metric='cosine', verbose=True)
    X_i = tsne.fit_transform(clip_feats)

    fig, ax = plt.subplots()
    lw = 2
    ax.scatter(
        X_i[:, 0], X_i[:, 1], color="white", alpha=0.1, label='image'
    )
    
    
    for i, name in enumerate(clip_names_array):
        if i % 1000 in range(20):
            name_ins, name_view = name.split("@")
            if int(name_view) != 20:
                name_view = str(int(name_view)).zfill(3)
                path = os.path.join('../../data/render_ShapeNetCore.v1/20_view_shade', name_ins, name_view + '.png')
                img = mpimg.imread(path)
                imagebox = OffsetImage(img, zoom=0.02, alpha=0.6)
                ab = AnnotationBbox(imagebox, (X_i[i, 0], X_i[i, 1]), pad=0.01, bboxprops=dict(facecolor = "none", lw = 0))
                ax.add_artist(ab)

    fig.savefig(save_image_path, format="png", dpi=1200 )

# TSNE Visualization of CLIP Embedding Space
def tsne_visualize_instances(clip_names_array, clip_feats, clip_feats_nvh, save_image_path):
    tsne = TSNE(n_components=2, init='pca', n_iter=1000, learning_rate='auto', metric='cosine', verbose=True)
    X_i = tsne.fit_transform(clip_feats)

    plt.figure()  
    lw = 2
    plt.scatter(
        X_i[:, 0], X_i[:, 1], color="navy", alpha=0.4, label='image'
    )

    idx_rands = np.random.randint(len(clip_names_array), size=5)
    
    colors = ["silver", "salmon", "gold", "lime", "magenta"]
    for c, r_idx in zip(colors, idx_rands):    
        plt.scatter(
            X_i[r_idx * 20: r_idx * 20 + 20, 0], X_i[r_idx * 20: r_idx * 20 + 20, 1], 
            color=c, alpha=0.8, label='instance'
        )

    plt.title("TSNE of CLIP Embedding")
    plt.savefig(save_image_path, format="png", dpi=1200)
    plt.close()


# Distribution Visualization of CLIP Embedding Features
def dist_visualize(clip_feats, clip_feats_nvh, text_str_list, text_feats, all_feats, save_image_path):
    clip_feats_mean = np.mean(clip_feats, axis=0)
    text_feats_mean = np.mean(text_feats, axis=0)
    fig, ax = plt.subplots()
    rects1 = ax.bar(range(512), clip_feats_mean, label='img')
    rects2 = ax.bar(range(512), text_feats_mean, label='txt')

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("Distribution of CLIP Embedding")
    plt.savefig(save_image_path, format="png", dpi=1200 )

if __name__ == '__main__':
    clip_img_feat_path1 = "../../data/embed_ShapeNetCore.v1/clip_img_feat_1_view_texture_keyshot.pickle"
    clip_img_feat_path2 = "../../data/embed_ShapeNetCore.v1/clip_img_feat_20view_shade.pickle"
    save_image_path1 = 'utils/pca_results/finetune/clip_embed_2stage_1645468753585_164573148881735.png'
    save_image_path2 = 'utils/pca_results/20view_shade_clip_embed_tsn_visualization_multiview.png'
    map_restore_file1 = "openai_clip/finetune/results/1645468753585/best_model_checkpoint.pt"
    map_restore_file2 = "openai_clip/finetune/results_map_stage/164573148881735/best_model_checkpoint.pt"
    clip_names_array1, clip_feats1, clip_feats_nvh, text_str_list, text_features_list, text_feats, all_feats = load_data(clip_img_feat_path1, True, finetune_clip=True, map_restore_file1=map_restore_file1, map_restore_file2=map_restore_file2)
    # clip_names_array2, clip_feats2, clip_feats_nvh, _, _, _, _ = load_data(clip_img_feat_path2, False, True)
    # pca_visualize_instances(clip_names_array, clip_feats, clip_feats_nvh, text_str_list, text_features_list, text_feats, all_feats)
    # pca_visualize(clip_feats, clip_feats_nvh, text_feats, all_feats, save_image_path)
    tsn_visualize(clip_feats1, clip_feats_nvh, text_str_list, text_feats, all_feats, save_image_path1)
    # tsn_visualize_txt(clip_feats1, clip_feats_nvh, text_str_list, text_feats, all_feats, save_image_path1)
    # tsn_analysis_txt(clip_feats1, clip_feats_nvh, text_str_list, text_feats, all_feats, save_image_path1)
    # tsn_visualize_paired(clip_feats1, clip_feats2, text_str_list, text_feats, all_feats, save_image_path)
    # tsn_visualize_img(clip_names_array1, clip_feats1, clip_feats_nvh, text_str_list, text_feats, all_feats, save_image_path1)
    # tsn_visualize_paired_img(clip_names_array1, clip_feats1, clip_feats2, text_str_list, text_feats, all_feats, save_image_path)
    # tsne_visualize_instances(clip_names_array1, clip_feats1, clip_feats_nvh, save_image_path1)
    # tsn_visualize_img_multiview(clip_names_array2, clip_feats2, clip_feats_nvh, save_image_path2)
    # dist_visualize(clip_feats, clip_feats_nvh, text_str_list, text_feats, all_feats, save_image_path)