import os
import pickle
import numpy as np
import torch
import random
import math
from mlp_networks import B_MappingNetwork

def load_pickle_file(path):
    with open(path, 'rb') as handle:
        feats = pickle.load(handle)
    return feats

def pair_img_txt_feats(args, model_names, clip_img_feats_dict, clip_txt_feats_dict, contrast_pairs=False):
    i_stop = -4 if len(list(clip_img_feats_dict.keys())[0]) < 32 else None

    if args.s1_restore_file:
        map_model = B_MappingNetwork().to(args.device)
        state = torch.load(args.s1_restore_file, map_location=args.device)
        map_model.load_state_dict(state['model_state'])
        map_model.eval()
    
    if contrast_pairs: # ensure for each instance, only one correct pair in the batch
        if args.split_by_text:
            total_len = len(model_names)
            
            n_bin = math.ceil(total_len / args.batch_size)
            bin_cursors = [0] * n_bin
            t_feats_list = [torch.Tensor(0)] * total_len
            t_feats_name_list = [''] * total_len
            i_feats_list = [torch.Tensor(0)] * total_len
            i_feats_name_list = [''] * total_len
            y_list = [0] * total_len

            for m_name in list(clip_txt_feats_dict.keys()):
                t_feats_dict = clip_txt_feats_dict[m_name]
                i_feats_dict = clip_img_feats_dict[m_name[:i_stop]]

                for t_key in t_feats_dict:
                    if t_key in model_names:
                        i_key = random.choice(list(i_feats_dict.keys()))

                        for bin_idx in range(n_bin):
                            cursor_idx = bin_cursors[bin_idx]
                            bin_low = bin_idx * args.batch_size
                            bin_upp = (bin_idx+1) * args.batch_size if (bin_idx+1) * args.batch_size < total_len else total_len
                            if (m_name not in t_feats_name_list[bin_low: bin_upp]) and (bin_low + cursor_idx < bin_upp):
                                t_feats_list[bin_low + cursor_idx] = t_feats_dict[t_key]
                                t_feats_name_list[bin_low + cursor_idx] = m_name
                                i_feats_list[bin_low + cursor_idx] = i_feats_dict[i_key]
                                i_feats_name_list[bin_low + cursor_idx] = m_name
                                y_list[bin_low + cursor_idx] = 1
                                bin_cursors[bin_idx] += 1
                                break
                            elif bin_idx == n_bin - 1:
                                for bin_idx in range(n_bin):
                                    cursor_idx = bin_cursors[bin_idx]
                                    bin_low = bin_idx * args.batch_size
                                    bin_upp = (bin_idx+1) * args.batch_size if (bin_idx+1) * args.batch_size < total_len else total_len
                                    if (bin_low + cursor_idx < bin_upp):
                                        t_feats_list[bin_low + cursor_idx] = t_feats_dict[t_key]
                                        t_feats_name_list[bin_low + cursor_idx] = m_name
                                        i_feats_list[bin_low + cursor_idx] = i_feats_dict[i_key]
                                        i_feats_name_list[bin_low + cursor_idx] = m_name
                                        y_list[bin_low + cursor_idx] = 1
                                        bin_cursors[bin_idx] += 1
        else:
            total_len = 0
            for m_name in model_names:
                if m_name in list(clip_txt_feats_dict.keys()):
                    t_feats_dict = clip_txt_feats_dict[m_name]
                    i_feats_dict =  clip_img_feats_dict[m_name[:i_stop]]
                    total_len += len(list(t_feats_dict.keys())) #* len(list(i_feats_dict.keys()))
            n_bin = math.ceil(total_len / args.batch_size)
            bin_cursors = [0] * n_bin
            t_feats_list = [torch.Tensor(0)] * total_len
            t_feats_name_list = [''] * total_len
            i_feats_list = [torch.Tensor(0)] * total_len
            i_feats_name_list = [''] * total_len
            y_list = [0] * total_len

            for m_name in model_names:
                if m_name in list(clip_txt_feats_dict.keys()):
                    t_feats_dict = clip_txt_feats_dict[m_name]
                    i_feats_dict = clip_img_feats_dict[m_name[:i_stop]]

                    for t_key in t_feats_dict:
                        for _ in range(1):
                            i_key = random.choice(list(i_feats_dict.keys()))

                            for bin_idx in range(n_bin):
                                cursor_idx = bin_cursors[bin_idx]
                                bin_low = bin_idx * args.batch_size
                                bin_upp = (bin_idx+1) * args.batch_size if (bin_idx+1) * args.batch_size < total_len else total_len
                                if (m_name not in t_feats_name_list[bin_low: bin_upp]) and (bin_low + cursor_idx < bin_upp):
                                    t_feats_list[bin_low + cursor_idx] = t_feats_dict[t_key]
                                    t_feats_name_list[bin_low + cursor_idx] = m_name
                                    i_feats_list[bin_low + cursor_idx] = i_feats_dict[i_key]
                                    i_feats_name_list[bin_low + cursor_idx] = m_name
                                    y_list[bin_low + cursor_idx] = 1
                                    bin_cursors[bin_idx] += 1
                                    break
                                elif bin_idx == n_bin - 1:
                                    for bin_idx in range(n_bin):
                                        cursor_idx = bin_cursors[bin_idx]
                                        bin_low = bin_idx * args.batch_size
                                        bin_upp = (bin_idx+1) * args.batch_size if (bin_idx+1) * args.batch_size < total_len else total_len
                                        if (bin_low + cursor_idx < bin_upp):
                                            t_feats_list[bin_low + cursor_idx] = t_feats_dict[t_key]
                                            t_feats_name_list[bin_low + cursor_idx] = m_name
                                            i_feats_list[bin_low + cursor_idx] = i_feats_dict[i_key]
                                            i_feats_name_list[bin_low + cursor_idx] = m_name
                                            y_list[bin_low + cursor_idx] = 1
                                            bin_cursors[bin_idx] += 1
        print('Contrastive Loader Finished Loading...')
    else:
        t_feats_list = []
        t_feats_name_list = []
        i_feats_list = []
        i_feats_name_list = []
        y_list = []
        for m_name in model_names:
            if m_name in list(clip_txt_feats_dict.keys()):
                t_feats_dict = clip_txt_feats_dict[m_name]
                i_feats_dict = clip_img_feats_dict[m_name[:i_stop]]
                for t_key in t_feats_dict:
                    i_key = random.choice(list(i_feats_dict.keys()))

                    t_feats_list.append(t_feats_dict[t_key])
                    t_feats_name_list.append(m_name)
                    i_feats_list.append(i_feats_dict[i_key])
                    i_feats_name_list.append(m_name)
                    y_list.append(1)
        print('Costume Loader Finished Loading...')

    t_feats = np.vstack(t_feats_list)
    i_feats = np.vstack(i_feats_list)
    y = np.asarray(y_list)

    if args.s1_restore_file:
        with torch.no_grad():
            t_feats = torch.from_numpy(t_feats).to(args.device).float()
            t_feats = t_feats.view(-1, 512)
            i_feats = torch.from_numpy(i_feats).to(args.device).float()
            i_feats = i_feats.view(-1, 512)
            t_feats /= t_feats.norm(dim=-1, keepdim=True)
            i_feats /= i_feats.norm(dim=-1, keepdim=True)
            i_feats, t_feats, _ = map_model(i_feats, t_feats)
            t_feats = t_feats.cpu().numpy()
            i_feats = i_feats.cpu().numpy()

    return t_feats, i_feats, t_feats_name_list, i_feats_name_list, y


class ClipText2Image(torch.utils.data.Dataset):
    def __init__(self, args, clip_img_feat_path='./', clip_txt_feat_path='./', split='train'):
        super(ClipText2Image, self).__init__()
        if args.split_by_text:
            split_file = os.path.join(args.image_folder, split + '.elst')
        else:
            split_file = os.path.join(args.image_folder, split + '.lst')

        with open(split_file, 'r') as f:
            self.model_names = f.read().split('\n')
        print("Number of Instance", len(self.model_names))       

        self.clip_img_feats_dict = load_pickle_file(clip_img_feat_path)
        self.clip_txt_feats_dict = load_pickle_file(clip_txt_feat_path)

        self.clip_t_feats, self.clip_i_feats, self.t_feats_name_list, self.i_feats_name_list, self.y = pair_img_txt_feats(args, self.model_names, self.clip_img_feats_dict, self.clip_txt_feats_dict, contrast_pairs=args.constrast_loader)
        self.device = args.device

    def __len__(self):
        return self.clip_t_feats.shape[0]

    def __getitem__(self, idx):
        t_feat = self.clip_t_feats[idx]
        i_feat = self.clip_i_feats[idx]
        t_feats_name = self.t_feats_name_list[idx]
        i_feats_name = self.i_feats_name_list[idx]

        y = self.y[idx]

        t_feat = torch.from_numpy(t_feat).float().to(self.device) #[512]
        i_feat = torch.from_numpy(i_feat).float().to(self.device) #[512]
        y = torch.Tensor(y).to(self.device) #[1]

        t_feat /= t_feat.norm(dim=-1, keepdim=True)
        i_feat /= i_feat.norm(dim=-1, keepdim=True)
        
        return t_feat, i_feat, t_feats_name, i_feats_name, y

def fetch_dataloaders(args, kwargs={}):
    if args.n_dataset == 'ClipText2Image':
        train_dataset = ClipText2Image(args, args.clip_img_feat_path, args.clip_txt_feat_path, 'train')
        test_dataset = ClipText2Image(args, args.clip_img_feat_path, args.clip_txt_feat_path, 'test')
        val_dataset = ClipText2Image(args, args.clip_img_feat_path, args.clip_txt_feat_path, 'val')
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=args.constrast_loader, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader, val_loader


if __name__ == '__main__':
    clip_img_feat_path = '../../data/embed_ShapeNetCore.v1/clip_img_feat_1view_shade.pickle'
    clip_txt_feat_path = '../../data/embed_ShapeNetCore.v1/clip_txt_feat_text2shape.pickle'
    dataset = ClipText2Image(clip_img_feat_path, clip_txt_feat_path, 'train')
