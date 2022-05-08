import pandas as pd
import math
import clip
import torch
import pickle
import numpy as np
import tqdm



def get_all_valid_data_text2shape():

    file_name = "../../data/ShapeNetCore.v1/meta/captions.tablechair.csv"
    df = pd.read_csv(file_name, usecols= ['modelId', 'description', 'category'])

    df = df.reset_index()  # make sure indexes pair with number of rows
    ids_0 = []
    labels = []
    texts = []

    for index, row in df.iterrows():
        if str(row['category']) == 'Table':
            ids_0.append(str(row['modelId']))
            texts.append(str(row['description']))
    print(len(texts))
    return ids_0, texts

def get_all_valid_data_subcate(df):
    # 53 classes
    all_valid_name = ['ball chair', 'cantilever chair', 'armchair', 'tulip chair', 'straight chair', 'side chair', 'club chair', 'swivel chair', 'easy chair', 'lounge chair', 'overstuffed chair', 'barcelona chair', 'chaise longue', 'chaise', 'daybed', 'deck chair', 'beach chair', 'folding chair', 'bean chair', 'butterfly chair', 'rocking chair', 'rocker', 'zigzag chair', 'recliner', 'reclining chair', 'lounger', 'lawn chair', 'garden chair', 'Eames chair', 'sofa', 'couch', 'lounge', 'rex chair', 'camp chair', 'X chair', 'Morris chair', 'NO. 14 chair', 'park bench', 'table', 'wassily chair', 'Windsor chair', 'love seat', 'loveseat', 'tete-a-tete', 'vis-a-vis', 'wheelchair', 'bench', 'wing chair', 'ladder-back', 'ladder-back chair', 'throne', 'double couch', 'settee']

    file_name = "../../data/ShapeNetCore.v1/03001627.csv"
    df = pd.read_csv(file_name, usecols= ['fullId', 'wnlemmas'])

    df = df.reset_index()  # make sure indexes pair with number of rows
    
    ids_0 = []
    labels = []
    texts = []

    for index, row in df.iterrows():
        if str(row['category']) == 'Chair':
            ids_0.append(str(row['modelId']))
            texts.append(str(row['description']))
    print(len(texts))
    return ids_0, texts

def infer_all_texts(dst_path, ids_0, texts):
    model, _ = clip.load('ViT-B/32', device='cuda')
    text_feats_list = {}
    for idx in tqdm.tqdm(range(len(texts))):
        if ids_0[idx] not in list(text_feats_list.keys()):
            text_feats_list[ids_0[idx]] = {}
        k_idx = texts[idx].replace(' ', '_')
        batch_text = texts[idx]
        try:
            text_token = clip.tokenize(batch_text).cuda()
            with torch.no_grad():
                batch_text_feat = model.encode_text(text_token).float()
            text_feats_list[ids_0[idx]][k_idx] = batch_text_feat.squeeze(0).cpu().numpy()
        except:
            pass

    with open(dst_path, 'wb') as handle:
        pickle.dump(text_feats_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    dst_path = '../../data/embed_ShapeNetCore.v1/04379243/clip_txt_feat_text2shape.pickle'
    ids_0, texts = get_all_valid_data_text2shape()
    infer_all_texts(dst_path, ids_0, texts)