import pandas as pd
import math
import clip
import torch
import pickle
import numpy as np
import tqdm
from finetune.stylegan2networks import MappingNetwork

# 54 classes
all_valid_name = ['ball chair', 'cantilever chair', 'armchair', 'tulip chair', 'straight chair', 'side chair', 'club chair', 'swivel chair', 'easy chair', 'lounge chair', 'overstuffed chair', 'barcelona chair', 'chaise longue', 'chaise', 'daybed', 'deck chair', 'beach chair', 'folding chair', 'bean chair', 'butterfly chair', 'rocking chair', 'rocker', 'zigzag chair', 'recliner', 'reclining chair', 'lounger', 'lawn chair', 'garden chair', 'Eames chair', 'sofa', 'couch', 'lounge', 'rex chair', 'camp chair', 'X chair', 'Morris chair', 'NO. 14 chair', 'park bench', 'table', 'wassily chair', 'Windsor chair', 'love seat', 'loveseat', 'tete-a-tete', 'vis-a-vis', 'wheelchair', 'bench', 'wing chair', 'ladder-back', 'ladder-back chair', 'throne', 'double couch', 'settee']

file_name = "../../data/Shapeglot/main_data_for_chairs/language/shapenet_chairs.csv"
df = pd.read_csv(file_name, usecols= ['chair_a', 'chair_b', 'chair_c', 'context_condition', 'correct', 'target_chair', 'text'])

df = df.reset_index()  # make sure indexes pair with number of rows

def get_all_valid_data(df):
    ids_0 = []
    ids_1 = []
    ids_2 = []
    labels = []
    texts = []
    easy_ids = []
    hard_ids = []
    for index, row in df.iterrows():
        ids_0.append(str(row['chair_a']))
        ids_1.append(str(row['chair_b']))
        ids_2.append(str(row['chair_c']))
        labels.append(int(row['target_chair']))
        if str(row['context_condition']) == 'easy':
            if row['correct']:
                easy_ids.append(index)
        elif str(row['context_condition']) == 'hard':
            if row['correct']:
                hard_ids.append(index)
        else:
            print(row['context_condition'])
        texts.append(str(row['text']))
    return ids_0, ids_1, ids_2, easy_ids, hard_ids, labels, texts


def load_map_model():
    mp_model = MappingNetwork(512, 0, 512, None)
    return mp_model

def restore_map_model(map_model, map_restore_file):
    map_model = map_model.cuda()
    state = torch.load(map_restore_file, map_location='cuda')
    map_model.load_state_dict(state['model_state'])
    map_model.eval().cuda()
    return map_model

def infer_all_texts(texts, map_restore_file=None):
    model, _ = clip.load('ViT-B/32', device='cuda')
    if map_restore_file is not None:
        map_model = load_map_model()
        map_model = restore_map_model(map_model, map_restore_file)
    text_feats_list = []
    for idx in tqdm.tqdm(range(len(texts)//4706)):
        batch_text = texts[idx*4706 : idx*4706 + 4706]
        text_token = clip.tokenize(batch_text).cuda()
        with torch.no_grad():
            batch_text_feat = model.encode_text(text_token).float()
            batch_text_feat = batch_text_feat / batch_text_feat.norm(dim=-1, keepdim=True)
            if map_restore_file is not None:
                batch_text_feat = map_model(batch_text_feat)
        text_feats_list.append(batch_text_feat)
    text_feats = torch.vstack(text_feats_list)
    return text_feats

def load_clip_feats(path):
    with open(path, 'rb') as handle:
        clip_img_feats = pickle.load(handle)
        clip_feats_dict = {}
        for key in clip_img_feats:
            clip_feat_list = []
            for idx in clip_img_feats[key]:
                clip_feat_list.append(clip_img_feats[key][idx])
            clip_feat = np.stack(clip_feat_list, axis=0)
            clip_feat = torch.from_numpy(clip_feat).cuda().float()
            clip_feat = clip_feat.mean(dim=0, keepdim=True)
            clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
            clip_feats_dict[key] = clip_feat

    return clip_feats_dict

def accuracy_by_idx(img_0, img_1, img_2, label, text_feat):
    clip_feats = torch.vstack((img_0,img_1,img_2))
    similarity = (clip_feats @ text_feat.T).squeeze()
    idx_sort = torch.argsort(similarity, descending=True)
    if idx_sort[0] == label:
        return 1
    else:
        return 0

def accuracy(ids_0, ids_1, ids_2, labels, clip_feats_dict, text_feats, ids):
    acc_all = []
    for idx in ids:
        key_list = list(clip_feats_dict.keys())
        stop = -4
        # for key in key_list:
        #     if len(key) >= 32:
        #         stop = None
        try:
            img_0 = clip_feats_dict[ids_0[idx][:stop]]
            img_1 = clip_feats_dict[ids_1[idx][:stop]]
            img_2 = clip_feats_dict[ids_2[idx][:stop]]
            label = labels[idx]
            text_feat = text_feats[idx, :]
            acc = accuracy_by_idx(img_0, img_1, img_2, label, text_feat)
            acc_all.append(acc)
        except:
            pass
    return acc_all

if __name__ == '__main__':
    feature_path = '../../data/embed_ShapeNetCore.v1/clip_img_feat_1view_shade_10bright.pickle'
    map_restore_file = "openai_clip/finetune/results/1644794965089/best_model_checkpoint.pt"
    clip_feats_dict = load_clip_feats(feature_path)
    ids_0, ids_1, ids_2, easy_ids, hard_ids,  labels, texts = get_all_valid_data(df)
    text_feats = infer_all_texts(texts, None)

    acc_easy = accuracy(ids_0, ids_1, ids_2, labels, clip_feats_dict, text_feats, easy_ids)
    acc_hard = accuracy(ids_0, ids_1, ids_2, labels, clip_feats_dict, text_feats, hard_ids)
    acc_all = accuracy(ids_0, ids_1, ids_2, labels, clip_feats_dict, text_feats, range(len(ids_0)))
    print('Accuracy of Easy set is {:.3f}/{}'.format(sum(acc_easy)/len(acc_easy), len(acc_easy)))
    print('Accuracy of Hard set is {:.3f}/{}'.format(sum(acc_hard)/len(acc_hard), len(acc_hard)))
    print('Accuracy of All set is {:.3f}/{}'.format(sum(acc_all)/len(acc_all), len(acc_all)))
    