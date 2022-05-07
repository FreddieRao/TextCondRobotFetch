
import random
import clip
import numpy as np
import torch
import torch.nn as nn

from load_data import load_clip_feats, load_clip_txt_feats
from load_nf import load_map_model
from occupancy_networks.generate_from_z import ocnet_generation_from_z
from train_CLIP.generate import clip_tiny

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

templates29 = [    'a bad photo of a {}.',
    'a photo of many {}.',
    # 'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    # 'a rendering of a {}.',
    # 'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    # 'a tattoo of a {}.',
    # 'the embroidered {}.',
    'a photo of a hard to see {}.',
    # 'a bright photo of a {}.',
    'a photo of a clean {}.',
    # 'a photo of a dirty {}.',
    # 'a dark photo of the {}.',
    # 'a drawing of a {}.',
    'a photo of my {}.',
    # 'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    # 'a black and white photo of the {}.',
    # 'a painting of the {}.',
    # 'a painting of a {}.',
    # 'a pixelated photo of the {}.',
    # 'a sculpture of the {}.',
    # 'a bright photo of the {}.',
    'a cropped photo of a {}.',
    # 'a plastic {}.',
    # 'a photo of the dirty {}.',
    # 'a jpeg corrupted photo of a {}.',
    # 'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    # 'a rendering of the {}.',
    # 'a {} in a video game.',
    'a photo of one {}.',
    # 'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    # 'the origami {}.',
    # 'the {} in a video game.',
    # 'a sketch of a {}.',
    # 'a doodle of the {}.',
    # 'a origami {}.',
    'a low resolution photo of a {}.',
    # 'the toy {}.',
    # 'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    # 'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    # 'a blurry photo of a {}.',
    # 'a cartoon {}.',
    # 'art of a {}.',
    # 'a sketch of the {}.',
    # 'a embroidered {}.',
    # 'a pixelated photo of a {}.',
    # 'itap of the {}.',
    # 'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    # 'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    # 'the cartoon {}.',
    # 'art of the {}.',
    # 'a drawing of the {}.',
    'a photo of the large {}.',
    # 'a black and white photo of a {}.',
    # 'the plushie {}.',
    # 'a dark photo of a {}.',
    # 'itap of a {}.',
    # 'graffiti of the {}.',
    # 'a toy {}.',
    # 'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    # 'a tattoo of the {}.',
]

notemplate = ['{}'] #templates80 templates29

no_template = '{}'
gray_template = 'grey {}'
type_template = '{}, a type of chair.'
gray_type_template = 'gray {}, a type of chair'


def restore_nf_model(args, nf_model, mlps):
    if args.n_from_img > 0 or args.n_retrieval_shape > 0 or args.n_row >0:
        nf_model = nf_model.to(args.device)
        state = torch.load(args.nf_restore_file, map_location=args.device)
        nf_model.load_state_dict(state['model_state'])
        if 'mlps_state' in list(state.keys()):
            mlps.load_state_dict(state['mlps_state'])
    mlps.eval().to(args.device)
    nf_model.eval().to(args.device)
    return mlps, nf_model

def restore_map_model(args, map_mlp, map_model):
    if args.finetune_clip:
        map_model = map_model.to(args.device)
        state = torch.load(args.map_restore_file, map_location=args.device)
        map_model.load_state_dict(state['model_state'])
        if 'mlps_state' in list(state.keys()):
            map_mlp.load_state_dict(state['mlps_state'])
    map_model.eval().to(args.device)
    map_mlp.eval().to(args.device)
    return map_mlp, map_model

# clip feature conditional model
def encode_clip_txt_feats(args, clip_model='ViT-B/32', text_str_list=list(''), map_model=None):
    model, preprocess = clip.load(clip_model, device=args.device)
    model.eval().to(args.device)
    # model = clip_tiny
    if not args.unconditional:
        ## Text -> CLIP Text Embedding
        text_features_list = []
        if args.use_text2shape:
            text_features_list, text_feats_name_list,  text_str_list= load_clip_txt_feats(args, map_model if args.finetune_clip else None)
        else:
            for text_str in text_str_list:
                if args.use_txt_prefixs:
                    text_strs = [template.format(no_template.format(text_str)) for template in templates80 ]
                    text_token = clip.tokenize(text_strs).to(args.device)
                else:
                    text_token = clip.tokenize([text_str]).to(args.device)
                
                with torch.no_grad():
                    text_feat = model.encode_text(text_token).float()
                    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                    
                    if args.use_txt_prefixs:
                        text_feat = text_feat.mean(dim=0, keepdim=True)
                        text_feat1 = text_feat / text_feat.norm(dim=-1, keepdim=True)

                    if args.finetune_clip:
                        if args.map_code_base == 'B_MLPS':
                            _, text_feat, _ = map_model(text_feat, text_feat)
                        else:
                            _, text_feat, _ = map_model(text_feat, text_feat)
                    
                    text_features_list.append(text_feat)
            text_feats_name_list = text_str_list
        
    return text_features_list, text_feats_name_list, text_str_list

def generate_embed_from_clip_txt(args, nf_model, mlps, generate_conditioned, text_features_list):
    if args.n_row == 0:
        return []
    ## CLIP Text Embedding -> OCNET Shape Embedding
    if args.unconditional:
        shape_feature_list= [generate_conditioned(nf_model, labels=None, device=device, n_row=args.n_row, out_dim=args.input_size) for _ in range(10)]  
    else:
        shape_features_list= [generate_conditioned(nf_model, mlps(text_feat), args.device, n_row=args.n_row, out_dim=args.input_size) for text_feat in text_features_list]
    return shape_features_list

def generate_embed_from_clip_img(args, map_model, nf_model, mlps, generate_conditioned, k_gen=20, test_only=False):
    ## CLIP Image Embedding -> OCNET Shape Embedding
    clip_feats, clip_name_array = load_clip_feats(args, map_model, load_valid_cates_only=True, test_only=test_only)
    # For inference baseline(text-conditioned generation) only
    # clip_feats, clip_name_array, _ = load_clip_txt_feats(args, map_model, load_valid_cates_only=True)
    rand1 = random.choices(range(0, len(clip_name_array)), k=k_gen)
    text_str_list = [clip_name_array[r].split('@')[0] for r in rand1]
    clip_feats_list = [clip_feats[r].reshape(1, -1) for r in rand1]

    # If not generate shape from txt, generate 1 shape from image 
    n_row = 1 if args.n_row==0 else args.n_row
    shape_features_list = [generate_conditioned(nf_model, mlps(clip_feat), args.device, n_row=n_row, out_dim=args.input_size) for clip_feat in clip_feats_list]
    return shape_features_list, text_str_list

def generate_shape_from_embed(args, cfg, shape_features_list, shape_id_list, text_str_list, n_row, n_max=60):
    if args.n_row == 0:
        return []
    n_max = n_max if n_max<len(shape_features_list) else len(shape_features_list)
    ## OCNET Shape Embedding -> Implicit Shape
    for shape_features, shape_ids, text_str in zip(shape_features_list[:n_max], shape_id_list[:n_max], text_str_list[:n_max]):
        vis = args.quality or args.quant_clip
        try:
            out_mesh_folder = ocnet_generation_from_z(args, cfg, feats=shape_features, shape_ids=shape_ids, folder=args.out_mesh_folder, text=text_str, device=args.device, n_row=n_row, vis=vis)
        except OSError as oserr:
            if oserr.errno == 36:
                pass
    return out_mesh_folder
