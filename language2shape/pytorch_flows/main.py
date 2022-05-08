import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter

import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import ClipImage2ShapeCache, ClipText2ShapeCache
import flows as fnn
import utils
import pprint

from openai_clip.finetune.mlp_networks import F_MappingNetwork, B_MappingNetwork

if sys.version_info < (3, 6):
    print('Sorry, this code might need Python 3.6 or higher')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Flows')
parser.add_argument(
    '--cate',
    type=str,
    default='chair',
    help='Category of shapenet')
parser.add_argument(
    '--batch-size',
    type=int,
    default=100,
    help='input batch size for training (default: 100)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='number of epochs to train (default: 1000)')
parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument(
    '--dataset',
    default='POWER',
    help='POWER | GAS | HEPMASS | MINIBONE | BSDS300 | MOONS | ClipImage2ShapeCache')
parser.add_argument(
    '--flow', default='maf', help='flow to use: maf | realnvp | glow')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--cond',
    action='store_true',
    default=False,
    help='train class conditional flow (only for MNIST)')
parser.add_argument(
    '--num-blocks',
    type=int,
    default=5,
    help='number of invertible blocks (default: 5)')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=1000,
    help='how many batches to wait before logging training status')
parser.add_argument(
    '--output_dir',
    type=str,
    default='pytorch_flows/results/',
    help='Path to save checkpoints')
parser.add_argument(
    '--results_file',
    type=str,
    default='results.txt',
    help='Path to results info')
parser.add_argument(
    '--use_mlps',
    action='store_true', 
    help='Whether to Use MLPS for conditioning'
    )
parser.add_argument(
    '--finetune_clip',
    default=False, 
    action='store_true', 
    help='Whether to Finetune CLIP'
    )
parser.add_argument(
    '--no_finetune_clip',
    dest='finetune_clip', 
    action='store_false'
    )
parser.add_argument(
    '--map_restore_file',
    type=str, default='', 
    help='path to the checkpoint of CLIP MappingNet'
    )

args = parser.parse_args()
# set cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
args.device = device

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# set output files
timestamp = str(round(time.time() * 1000))
args.output_dir = os.path.join('pytorch_flows/results', args.output_dir, args.map_restore_file + '_' + timestamp)

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
args.results_file = os.path.join(args.output_dir, 'results.txt')

args.map_restore_file = os.path.join('openai_clip/finetune/results', args.map_restore_file, 'best_model_checkpoint.pt')

print('Loaded settings and model:')
print(pprint.pformat(args.__dict__))
print(pprint.pformat(args.__dict__), file=open(args.results_file, 'a'))

    
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


if args.dataset == 'ClipImage2ShapeCache':
    dataset = ClipImage2ShapeCache(args, B_MappingNetwork)
elif args.dataset == 'ClipText2ShapeCache':
    dataset = ClipText2ShapeCache(args)

if args.cond:
    train_tensor = torch.from_numpy(dataset.trn.x)
    train_labels = torch.from_numpy(dataset.trn.y)
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_labels = torch.from_numpy(dataset.val.y)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor, valid_labels)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_labels = torch.from_numpy(dataset.tst.y)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
    # num_cond_inputs = 10
    num_cond_inputs = 4096
else:
    train_tensor = torch.from_numpy(dataset.trn.x)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    num_cond_inputs = None
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

num_inputs = dataset.n_dims

num_hidden = {
    'POWER': 100,
    'GAS': 100,
    'HEPMASS': 512,
    'MINIBOONE': 512,
    'BSDS300': 512,
    'MOONS': 64,
    'MNIST': 1024,
    'ClipImage2ShapeCache': 2048,
    'ClipText2ShapeCache': 2048
}[args.dataset]

act = 'tanh' if args.dataset is 'GAS' else 'relu'

modules = []

assert args.flow in ['maf', 'maf-split', 'maf-split-glow', 'realnvp', 'glow']
if args.flow == 'glow':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    print("Warning: Results for GLOW are not as good as for MAF yet.")
    for _ in range(args.num_blocks):
        modules += [
            fnn.BatchNormFlow(num_inputs),
            fnn.LUInvertibleMM(num_inputs),
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu')
        ]
        mask = 1 - mask
elif args.flow == 'realnvp':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask
elif args.flow == 'maf':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split-glow':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.InvertibleMM(num_inputs)
        ]

model = fnn.FlowSequential(*modules)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)

model.to(device)


#### MLPS
if args.use_mlps:
    mlps = fnn.Feedforward(512, num_cond_inputs)
else:
    mlps = nn.Identity()
mlps.to(device)

optimizer = optim.Adam(list(model.parameters()) + list(mlps.parameters()), lr=args.lr, weight_decay=1e-6)
# AdamW(alpha, beta, lr), SGD(lr)
# writer = SummaryWriter(comment=args.flow + "_" + args.dataset)
global_step = 0

def train(epoch):
    global global_step# , writer
    model.train()
    mlps.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
                cond_data = mlps(cond_data)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))
        
        # writer.add_scalar('training/loss', loss.item(), global_step)
        global_step += 1
        
    pbar.close()
        
    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    if args.cond:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device),
                mlps(train_loader.dataset.tensors[1].to(data.device).float()))
    else:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device))


    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1


def validate(epoch, model, loader, prefix='Validation'):
    global global_step# , writer

    model.eval()
    mlps.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
                cond_data = mlps(cond_data)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            val_loss += -model.log_probs(data, cond_data).sum().item()  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    # writer.add_scalar('validation/LL', val_loss / len(loader.dataset), epoch)

    pbar.close()
    return val_loss / len(loader.dataset)

best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

for epoch in range(args.epochs):
    print('\nEpoch: {}'.format(epoch))

    train(epoch)
    validation_loss = validate(epoch, model, valid_loader)
    
    # save training checkpoint
    torch.save({'epoch': epoch,
                'model_state': model.state_dict(),
                'mlps_state': mlps.state_dict(),
                'optimizer_state': optimizer.state_dict()},
                os.path.join(args.output_dir, 'model_checkpoint.pt'))
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_state.pt'))

    if epoch - best_validation_epoch >= 10:
        break

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)
        torch.save({'epoch': epoch,
                'model_state': model.state_dict(),
                'mlps_state': mlps.state_dict(),
                'optimizer_state': optimizer.state_dict()},
                os.path.join(args.output_dir, 'best_model_checkpoint.pt'))

    print(
        'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
        format(best_validation_epoch, -best_validation_loss))
    


    if args.dataset == 'MOONS' and epoch % 10 == 0:
        utils.save_moons_plot(epoch, model, dataset)
    elif args.dataset == 'MNIST' and epoch % 1 == 0:
        utils.save_images(epoch, model, args.cond)


validate(best_validation_epoch, best_model, test_loader, prefix='Test')
