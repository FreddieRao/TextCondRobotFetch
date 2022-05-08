import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import time
import random
import math
import argparse
import copy
import sys
import pprint
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from finetune.load_data import *
from finetune.load_model import fetch_model
from finetune.clip_loss import *
from finetune.lr_scheduler import cosine_lr
parser = argparse.ArgumentParser()
# action
parser.add_argument('--cate', type=str, default='chair', help='Which category to train on.')
parser.add_argument('--clip_img_feat_path', type=str, default='../../data/embed_ShapeNetCore.v1/03001627/clip_img_feat_1view_texture_ground_keyshot_aug.pickle', help='Path to model to CLIP image feature.')
parser.add_argument('--clip_txt_feat_path', type=str, default='../../data/embed_ShapeNetCore.v1/03001627/clip_txt_feat_text2shape.pickle', help='Path to model to CLIP text feature.')
parser.add_argument('--image_folder', type=str, default='occupancy_networks/data/ShapeNet/03001627/', help='Path to annotation file.')
parser.add_argument('--output_dir', default='openai_clip/finetune/results', help='Path to save checkpoints.')
parser.add_argument('--restore_file', type=str, help='Path to restore checkpoints')
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
parser.add_argument('--s1_restore_file', type=str, help='Path to save stage1 checkpoints.')
# data
parser.add_argument('--n_dataset', type=str, default='ClipText2Image', help='Which dataset to use.')
parser.add_argument('--n_model', type=str, default='B_MLPS', help='Which model to use.')
parser.add_argument('--seed', type=int, default=42, help='Random seed to use.')
parser.add_argument('--split_by_text', action='store_true', default=False, help='Split dataset by text; Otherwise split by model id.')

# model

# made parameters
parser.add_argument('--h_features', type=int, default=512, help='Number of Hidden Features.')
parser.add_argument('--norm', type=str, default='LN', help='Normalization Layer.')
parser.add_argument('--act_f', type=str, default='tanh', help='Activation Layer.')
parser.add_argument('--l2norm', action='store_true', default=True, help='Use L2 Norm at the last layer.')
parser.add_argument('--no_l2norm', dest='l2norm', action='store_false')
parser.add_argument('--num_layers', type=int, default=1, help='Number of Layers.')
parser.add_argument('--lrelu', type=float, default=0.01, help='Slope for lrelu.')
parser.add_argument('--logit_scale', type=float, default=1, help='Logit_scale for contrast loss.')
parser.add_argument('--orthogonal', action='store_true', default=False, help='Use Orthogonal to FC Layer.')

# training params
parser.add_argument('-b', '--batch_size', type=int, default=1024)
parser.add_argument('--n_epochs', type=int, default=32)
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--opt', type=str, default='adamw', help='Choice of Optimizer')
parser.add_argument('--lr', type=float, default=0.0064, help='Learning rate.')
parser.add_argument('--wd', type=float, default=0.00001, help='Learning rate.')
parser.add_argument('--coslr', action='store_true', default=True, help='Use Learning Rate Scheduler.')
parser.add_argument('--no_coslr', dest='coslr', action='store_false')
parser.add_argument('--warmup_step', type=int, default=400, help='Warming up Steps for Learning Rate Scheduler.')
parser.add_argument('--total_step', type=int, default=0, help='Totoal Steps for Learning Rate Scheduler.')
parser.add_argument('--log_interval', type=int, default=10, help='Total Step for Learning Rate Scheduler')
parser.add_argument('--loss_name_list', type=str, default='scontrast_cos', help='Name of loss for training')
parser.add_argument('--loss_weights', type=str, default='0.5_0.5', help='Weights of loss for training')
parser.add_argument('--constrast_loader', action='store_true', default=False, help='User Contrastive Dataloader')
# --------------------
# Train and evaluate
# --------------------

def train(model, dataloader, optimizer, loss_name_list, loss_f_list, loss_weight_list, epoch, args, scheduler=None):

    for i, data in enumerate(dataloader):
        model.train()

        # feed forward
        t_feats, i_feats, _, _, y = data
        y = y.squeeze(-1)
        map_i_feats, map_t_feats, logit_scale = model(i_feats, t_feats)

        # indicates all positive samples
        train_loss_list = []
        for l_name, l_weight, l_f in zip(loss_name_list, loss_weight_list, loss_f_list):
            if l_name in ['cos', 'cosemb', 'discos']:
                train_loss_list.append(l_weight * l_f(map_t_feats, map_i_feats, y)) 
            elif l_name in ['contrast', 'scontrast']:
                 train_loss_list.append(l_weight * l_f(map_i_feats, map_t_feats, logit_scale)) 
            elif l_name in ['disdiv']:
                 train_loss_list.append(0.5 * l_weight * (l_f(map_i_feats, i_feats) + l_f(map_t_feats, t_feats))) 
            else:
                train_loss_list.append(l_weight * l_f(map_t_feats, map_i_feats)) 
        
        train_loss = torch.vstack(train_loss_list).sum()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if args.coslr:
            args.total_step += 1
            scheduler(args.total_step)

        if i % 100 == 0:
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.8f}'.format(
                epoch, args.start_epoch + args.n_epochs, i, len(dataloader), train_loss.item()))

    return train_loss

@torch.no_grad()
def eval_rtv_precision(all_map_t_feats, all_map_i_feats, t_feats_name_list, i_feats_name_list, rtv_num=10):
    n = all_map_t_feats.shape[0]
    all_precision = [0] * n
    all_precision10 = [0] * n
    similarity = (all_map_t_feats @ all_map_i_feats.T).squeeze()
    idx_sort = torch.argsort(similarity, dim=-1, descending=True)

    for n_idx in range(n):
        if t_feats_name_list[n_idx] == i_feats_name_list[idx_sort[n_idx, 0]]:
            all_precision[n_idx] = 1
        for r_idx in range(rtv_num):
            if t_feats_name_list[n_idx] == i_feats_name_list[idx_sort[n_idx, r_idx]]:
                all_precision10[n_idx] = 1

    total_precision1 = sum(all_precision) / len(all_precision)
    total_precision10 = sum(all_precision10) / len(all_precision10)

    return total_precision1, total_precision10

@torch.no_grad()
def eval_rtv_precision_instance(model, t_feats, i_feats, t_feats_name_list, i_feats_name_list, rtv_num=10):
    model.eval()
    _, map_t_feats, _ = model(t_feats, t_feats)
    map_i_feats, _, _ = model(i_feats, i_feats)

    n = map_t_feats.shape[0]
    all_precision = [0] * n
    all_precision10 = [0] * n
    similarity = (map_t_feats @ map_i_feats.T).squeeze()
    idx_sort = torch.argsort(similarity, dim=-1, descending=True)

    for n_idx in range(n):
        if t_feats_name_list[n_idx] == i_feats_name_list[idx_sort[n_idx, 0]]:
            all_precision[n_idx] = 1
        for r_idx in range(rtv_num):
            if t_feats_name_list[n_idx] == i_feats_name_list[idx_sort[n_idx, r_idx]]:
                all_precision10[n_idx] = 1

    total_precision1 = sum(all_precision) / len(all_precision)
    total_precision10 = sum(all_precision10) / len(all_precision10)
    return total_precision1, total_precision10

@torch.no_grad()
def load_eval_rtv_data(args):
    if args.split_by_text:
        split_file = os.path.join(args.image_folder, 'val' + '.elst')
    else:
        split_file = os.path.join(args.image_folder, 'val' + '.lst')
    
    with open(split_file, 'r') as f:
        model_names = f.read().split('\n')

    clip_img_feats_dict = load_pickle_file(args.clip_img_feat_path)
    clip_txt_feats_dict = load_pickle_file(args.clip_txt_feat_path)
    i_stop = -4 if len(list(clip_img_feats_dict.keys())[0]) < 32 else None

    t_feats_list = []
    t_feats_name_list = []
    i_feats_list = []
    i_feats_name_list = []
    if args.split_by_text:
        for m_name in list(clip_txt_feats_dict.keys()):
            t_feats_dict = clip_txt_feats_dict[m_name]
            i_feats_dict = clip_img_feats_dict[m_name[:i_stop]]
            for t_key in t_feats_dict:
                if t_key in model_names:
                    i_key = random.choice(list(i_feats_dict.keys()))

                    t_feats_list.append(t_feats_dict[t_key])
                    t_feats_name_list.append(m_name)
                    i_feats_list.append(i_feats_dict[i_key])
                    i_feats_name_list.append(m_name)
    else:
        for m_name in model_names:
            if m_name in list(clip_txt_feats_dict.keys()):
                t_feats_dict = clip_txt_feats_dict[m_name]
                i_feats_dict = clip_img_feats_dict[m_name[:i_stop]]
                for t_key in t_feats_dict:
                    t_feats_list.append(t_feats_dict[t_key])
                    t_feats_name_list.append(m_name)
                for i_key in i_feats_dict:
                    i_feats_list.append(i_feats_dict[i_key])
                    i_feats_name_list.append(m_name)

    t_feats = torch.from_numpy(np.vstack(t_feats_list)).float().to(args.device)
    i_feats = torch.from_numpy(np.vstack(i_feats_list)).float().to(args.device)
    t_feats /= t_feats.norm(dim=-1, keepdim=True)
    i_feats /= i_feats.norm(dim=-1, keepdim=True)

    if args.s1_restore_file:
        map_model = B_MappingNetwork().to(args.device)
        state = torch.load(args.s1_restore_file, map_location=args.device)
        map_model.load_state_dict(state['model_state'])
        map_model.eval()
        i_feats, t_feats, _ = map_model(i_feats, t_feats)
    return t_feats, i_feats, t_feats_name_list, i_feats_name_list


@torch.no_grad()
def evaluate(model, dataloader, loss_name_list, loss_f_list,loss_weight_list, epoch, args, rtv_t_feats, rtv_i_feats, rtv_t_feats_name_list, rtv_i_feats_name_list):
    model.eval()

    total_val_loss = []
    map_i_feats_list = []
    map_t_feats_list = []
    i_feats_name_list = []
    t_feats_name_list = []

    for i, data in enumerate(dataloader):
        # feed forward
        t_feats, i_feats, t_feats_name, i_feats_name, y = data
        y = y.squeeze(1)
        map_i_feats, map_t_feats, logit_scale = model(i_feats, t_feats)
        map_i_feats_list.append(map_i_feats)
        map_t_feats_list.append(map_t_feats)
        i_feats_name_list += list(i_feats_name)
        t_feats_name_list += list(t_feats_name)

        val_loss = 0
        for l_name, l_weight, l_f in zip(loss_name_list, loss_weight_list, loss_f_list):
            if l_name in ['cos', 'cosemb', 'discos']:
                val_loss += l_weight * l_f(map_t_feats, map_i_feats, y)
            elif l_name in ['contrast', 'scontrast']:
                val_loss += l_weight * l_f(map_i_feats, map_t_feats, logit_scale, reduction='none')
            elif l_name in ['disdiv']:
                 val_loss += 0.5 * l_weight * (l_f(map_i_feats, i_feats) + l_f(map_t_feats, t_feats))
            else:
                val_loss += l_weight * l_f(map_t_feats, map_i_feats).sum(-1)

        total_val_loss.append(val_loss)
    
    # all_map_i_feats = torch.vstack(map_i_feats_list)
    # all_map_t_feats = torch.vstack(map_t_feats_list)
    # total_precision, total_precision5 = eval_rtv_precision(all_map_t_feats, all_map_i_feats, t_feats_name_list, i_feats_name_list)
    total_precision, total_precision5 = eval_rtv_precision_instance(model, rtv_t_feats, rtv_i_feats, rtv_t_feats_name_list, rtv_i_feats_name_list, rtv_num=10)
    
    total_val_loss = torch.cat(total_val_loss, dim=0)
    mean_val_loss = torch.mean(total_val_loss).cpu().numpy()

    output = 'Evaluate ' + (epoch != None)*'(epoch {}) -- '.format(epoch) + 'val loss = {:.8f}.'.format(mean_val_loss)
    print(output)
    print(output, file=open(args.results_file, 'a'))
    output = 'Evaluate ' + 'R@P1 = {:.4f}.'.format(total_precision) + 'R@P10 = {:.4f}.'.format(total_precision5)
    print(output)
    print(output, file=open(args.results_file, 'a'))
    return mean_val_loss, total_precision, total_precision5

def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_name_list, loss_weight_list, args, scheduler):
    best_val_loss = float('+inf')
    best_rtv1 = float('-inf')
    best_rtv10 = float('-inf')
    best_epoch = args.start_epoch

    train_loss_f_list = []
    eval_loss_f_list = []

    for loss_name in loss_name_list:
        if loss_name == 'l1':
            train_loss_f_list.append(torch.nn.L1Loss())
            eval_loss_f_list.append(torch.nn.L1Loss(reduction='none'))
        elif loss_name == 'mse':
            train_loss_f_list.append(torch.nn.MSELoss())
            eval_loss_f_list.append(torch.nn.MSELoss(reduction='none'))
        elif loss_name == 'sl1':
            train_loss_f_list.append(torch.nn.SmoothL1Loss())
            eval_loss_f_list.append(torch.nn.SmoothL1Loss(reduction='none'))
        elif loss_name == 'cos':
            train_loss_f_list.append(torch.nn.CosineEmbeddingLoss())
            eval_loss_f_list.append(torch.nn.CosineEmbeddingLoss(reduction='none'))
        elif loss_name == 'cosemb':
            train_loss_f_list.append(torch.nn.CosineEmbeddingLoss())
            eval_loss_f_list.append(torch.nn.CosineEmbeddingLoss(reduction='none'))
        elif loss_name == 'disdiv':
            train_loss_f_list.append(torch.nn.KLDivLoss())
            eval_loss_f_list.append(torch.nn.KLDivLoss())
        elif loss_name == 'dismse':
            train_loss_f_list.append(torch.nn.MSELoss())
            eval_loss_f_list.append(torch.nn.MSELoss(reduction='none'))
        elif loss_name == 'discos':
            train_loss_f_list.append(torch.nn.CosineEmbeddingLoss())
            eval_loss_f_list.append(torch.nn.CosineEmbeddingLoss(reduction='none'))
        elif loss_name == 'contrast':
            train_loss_f_list.append(contrastiveloss_f)
            eval_loss_f_list.append(contrastiveloss_f)     
        elif loss_name == 'scontrast':
            train_loss_f_list.append(single_contrastiveloss_f)
            eval_loss_f_list.append(single_contrastiveloss_f)         
        else:
            raise ValueError('{} is not defined.'.format(loss_name))


    # prepare data for rtv eval
    rtv_t_feats, rtv_i_feats, rtv_t_feats_name_list, rtv_i_feats_name_list = load_eval_rtv_data(args)
    val_loss, val_rp1, val_rp10 = evaluate(model, val_loader, loss_name_list, eval_loss_f_list, loss_weight_list, 0, args, rtv_t_feats, rtv_i_feats, rtv_t_feats_name_list, rtv_i_feats_name_list)
    for i in range(args.start_epoch, args.start_epoch + args.n_epochs):

        _ = train(model, train_loader, optimizer, loss_name_list, train_loss_f_list, loss_weight_list, i, args, scheduler)
        val_loss, val_rp1, val_rp10 = evaluate(model, val_loader, loss_name_list, eval_loss_f_list, loss_weight_list, i, args, rtv_t_feats, rtv_i_feats, rtv_t_feats_name_list, rtv_i_feats_name_list)
        
        # save training checkpoint
        torch.save({'epoch': i,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()},
                    os.path.join(args.output_dir, 'model_checkpoint.pt'))
        # save model only
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_state.pt'))

        # save best state
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = args.start_epoch + i
        
        if val_rp1 > best_rtv1:
            best_rtv1 = val_rp1
            best_rtv10 = val_rp10
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                        os.path.join(args.output_dir, 'best_model_checkpoint.pt'))
        elif args.start_epoch + i - best_epoch > 20:
            break
        print("Best Eval Loss is %f", best_val_loss)
        output = "Best Eval R@P1 is {} Best Eval R@P10 is {}".format(best_rtv1, best_rtv10)
        print(output)
        print(output, file=open(args.results_file, 'a'))

if __name__ == '__main__':

    args = parser.parse_args()
    
    if args.cate == 'chair':
        pass 
    elif args.cate == 'table':
        args.clip_img_feat_path = args.clip_img_feat_path.replace('03001627', '04379243')
        args.clip_txt_feat_path = args.clip_txt_feat_path.replace('03001627', '04379243')
        args.image_folder = args.image_folder.replace('03001627', '04379243')
        
    # setup file ops
    timestamp = str(round(time.time() * 1000)) + str(random.randint(0, 100)).zfill(2)
    args.output_dir = os.path.join(args.output_dir, timestamp)

    if args.n_model == 'MLPS':
        args.output_dir = args.output_dir.replace('results', 'results_map')

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.s1_restore_file:
        args.s1_restore_file = os.path.join('openai_clip/finetune/results/', args.s1_restore_file, 'best_model_checkpoint.pt')

    # setup device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # load data
    loss_name_list = args.loss_name_list.split('_')
    loss_weight_list = [float(_) for _ in args.loss_weights.split('_')]
    if 'contrast' in loss_name_list or 'scontrast' in loss_name_list:
        args.constrast_loader = True
    train_loader, test_loader, val_loader = fetch_dataloaders(args)

    # load model
    model = fetch_model(args)
    model = model.to(args.device)
    if args.opt =='adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr, 
                                    betas=[0.9,0.99], 
                                    eps=1e-8, 
                                    weight_decay=args.wd)
    elif args.opt =='adam0':
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr, 
                                    betas=[0, 0.99], 
                                    eps=1e-8, 
                                    weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr, 
                                    betas=[0.9, 0.999], 
                                    eps=1e-8, 
                                    weight_decay=args.wd)


    scheduler = cosine_lr(optimizer, args.lr, args.warmup_step, args.n_epochs*len(train_loader)//args.batch_size) if args.coslr else None

    args.results_file = os.path.join(args.output_dir, args.results_file)
    if args.restore_file:
        args.restore_file = os.path.join('openai_clip/finetune/results/', args.restore_file, 'best_model_checkpoint.pt')
        print('Restoring From {}',format(args.restore_file))
        print('Restoring From {}',format(args.restore_file), file=open(args.results_file, 'a'))
        print(model, file=open(args.results_file, 'a'))
        # load model and optimizer states
        state = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(state['model_state'])
    
    print('Loaded settings and model:')
    print(pprint.pformat(args.__dict__))
    print(model)
    print(pprint.pformat(args.__dict__), file=open(args.results_file, 'a'))
    print(model, file=open(args.results_file, 'a'))

    train_and_evaluate(model, train_loader, val_loader, optimizer, loss_name_list, loss_weight_list, args, scheduler)