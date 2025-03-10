from __future__ import print_function

import argparse
import pdb
import os
from os.path import join as j_
import sys
from utils.file_utils import save_pkl
from utils.utils import (seed_torch, array2list, merge_dict, read_splits,
                         parse_model_name, get_current_time)
from .trainer_tri_IB import train
from  wsi_datasets.wsi_survival_text_TMSE import WSIOmicsSurvivalDataset
import torch
from torch.utils.data import DataLoader

import pandas as pd
import json


def build_datasets(csv_splits, model_type, batch_size=1, num_workers=2, train_kwargs={}, val_kwargs={}):
    """
    Construct dataloaders from the data splits
    """
    dataset_splits = {}
    label_bins = None
    for k in csv_splits.keys():
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if (k == 'train') else val_kwargs.copy()
        dataset_kwargs['label_bins'] = label_bins
        dataset = WSIOmicsSurvivalDataset(df_histo=df['histo'], df_gene=df['gene'], **dataset_kwargs)

        if not model_type == 'Gaussian':
            batch_size = batch_size if dataset_kwargs.get('bag_size', -1) > 0 else 1

        if k == 'train':
            scaler = dataset.get_scaler()

        assert scaler is not None, "Omics scaler from train split required"
        dataset.apply_scaler(scaler)

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=dataset_kwargs['shuffle'], num_workers=num_workers)  
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')
        if (args.loss_fn == 'nll') and (k == 'train'):
            label_bins = dataset.get_label_bins()
    return dataset_splits


def main(args):
    if args.train_bag_size == -1:
        args.train_bag_size = args.bag_size
    if args.val_bag_size == -1:
        args.val_bag_size = args.bag_size
    if args.loss_fn != 'nll':
        args.n_label_bins = 0

    censorship_col = args.target_col.split('_')[0] + '_censorship'
    
    cancer_type = args.split_dir.split('/')[-2]
    args.omics_dir = j_(args.omics_dir, args.type_of_path, cancer_type)

    train_kwargs = dict(data_source=args.data_source,
                        survival_time_col=args.target_col,
                        censorship_col=censorship_col,
                        n_label_bins=args.n_label_bins,
                        label_bins=None,
                        bag_size=args.train_bag_size,
                        shuffle=True,
                        omics_dir=args.omics_dir,
                        omics_modality=args.omics_modality,
                        textemb_path=args.textemb_path,
                        read_feats=True
                        )
    val_kwargs = dict(data_source=args.data_source,
                      survival_time_col=args.target_col,
                      censorship_col=censorship_col,
                      n_label_bins=args.n_label_bins,
                      label_bins=None,
                      bag_size=args.val_bag_size,
                      shuffle=False,
                      omics_dir=args.omics_dir,
                      omics_modality=args.omics_modality,
                      textemb_path=args.textemb_path,
                      read_feats=True
                      )

    all_results= {}

    seed_torch(args.seed)
    csv_splits = read_splits(args)
    print('successfully read splits for: ', list(csv_splits.keys()))
    dataset_splits = build_datasets(csv_splits, 
                                    model_type=args.model_histo_type,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs,
                                    val_kwargs=val_kwargs)

    fold_results,_,result_cindex= train(dataset_splits, args)

    # Save results
    for split, split_results in fold_results.items():
        all_results[split] = merge_dict({}, split_results) if (split not in all_results.keys()) else merge_dict(all_results[split], split_results)
    
    final_dict = {}
    for split, split_results in all_results.items():
        final_dict.update({f'{metric}_{split}': array2list(val) for metric, val in split_results.items()})
    final_df = pd.DataFrame(final_dict)
    save_name = 'summary.csv'
    final_df.to_csv(j_(args.results_dir, save_name), index=False)

    return final_dict, result_cindex

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
### optimizer settings ###
parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--accum_steps', type=int, default=1,
                    help='grad accumulation steps')
parser.add_argument('--opt', type=str, default='adamW',
                    choices=['adamW', 'sgd', 'RAdam'])
parser.add_argument('--lr_scheduler', type=str,
                    choices=['cosine', 'linear', 'constant'], default='constant')
parser.add_argument('--warmup_steps', type=int,
                    default=-1, help='warmup iterations')
parser.add_argument('--warmup_epochs', type=int,
                    default=-1, help='warmup epochs')
parser.add_argument('--batch_size', type=int, default=1)

### misc ###
parser.add_argument('--print_every', default=100,
                    type=int, help='how often to print')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=2)

### Earlystopper args ###
parser.add_argument('--early_stopping', type=int,
                    default=0, help='enable early stopping')
parser.add_argument('--es_min_epochs', type=int, default=3,
                    help='early stopping min epochs')
parser.add_argument('--es_patience', type=int, default=5,
                    help='early stopping min patience')
parser.add_argument('--es_metric', type=str, default='loss',
                    help='early stopping metric')

### model args ###
parser.add_argument('--model_histo_type', type=str,default='Gaussian', help='type of histology model')
parser.add_argument('--ot_eps', default=0.1, type=float,
                    help='Strength for entropic constraint regularization for OT')
parser.add_argument('--model_histo_config', type=str,
                    default='ABMIL_default', help="name of model config file")
parser.add_argument('--n_fc_layers', type=int)
parser.add_argument('--em_iter', type=int)
parser.add_argument('--tau', type=float)
parser.add_argument('--out_type', type=str, default='param_cat')

# Multimodal args ###
parser.add_argument('--num_coattn_layers', default=1, type=int)
parser.add_argument('--model_mm_type', default='coattn',
                    help='Multimodal model type') 
parser.add_argument('--net_indiv', action='store_true', default=False)
parser.add_argument('--append_embed', type=str, default='none',
                    choices=['none', 'modality', 'proto', 'mp', 'random'])
parser.add_argument('--append_prob', action='store_true', default=False)
parser.add_argument('--histo_agg', default='mean')
parser.add_argument('--omics_dir', default='./data_csvs/rna')
parser.add_argument('--omics_modality', default='pathway')
parser.add_argument('--type_of_path', default='hallmarks')

# Prototype related
parser.add_argument('--load_proto', action='store_true', default=False)
parser.add_argument('--proto_path', type=str)
parser.add_argument('--fix_proto', action='store_true', default=False)
parser.add_argument('--n_proto', type=int)

parser.add_argument('--in_dim', default=768, type=int,
                    help='dim of input features')
parser.add_argument('--bag_size', type=int, default=-1)
parser.add_argument('--train_bag_size', type=int, default=-1)
parser.add_argument('--val_bag_size', type=int, default=-1)
parser.add_argument('--loss_fn', type=str, default='nll', choices=['nll', 'cox', 'sumo', 'ipcwls', 'rank'],
                    help='which loss function to use')
parser.add_argument('--nll_alpha', type=float, default=0,
                    help='Balance between censored / uncensored loss')

# experiment task / label args ###
parser.add_argument('--exp_code', type=str, default=None,
                    help='experiment code for saving results')
parser.add_argument('--task', type=str, default='unspecified_survival_task')
parser.add_argument('--target_col', type=str, default='os_survival_days')
parser.add_argument('--n_label_bins', type=int, default=4,
                    help='number of bins for event time discretization')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use')
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='overwrite existing results')

# logging args ###
parser.add_argument('--results_dir', default='./results',
                    help='results directory (default: ./results)')
parser.add_argument('--tags', nargs='+', type=str, default=None,
                    help='tags for logging')
parser.add_argument('--textemb_path',  type=str, default='None')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    print('task: ', args.task)
    args.split_dir = j_('splits', args.split_dir)
    print('split_dir: ', args.split_dir)
    split_num = args.split_dir.split('/')[-1].split('_k=')
    args.split_name_clean = args.split_dir.split('/')[2].split('_k=')[0]
    if len(split_num) > 1:
        args.split_k = int(split_num[1])
    else:
        args.split_k = 0

    if args.load_proto:
        assert os.path.isfile(args.proto_path), f"Path {args.proto_path} doesn't exist!"
        args.proto_fname = '/'.join(args.proto_path.split('/')[-2:])
        proto_fname_clean = '--'.join(args.proto_fname[:-4].split('/'))

    args.data_source = [src for src in args.data_source.split(',')]
    check_params_same = []
    for src in args.data_source: 
        ### assert data source exists + extract feature name ###
        print('data source: ', src)
        assert os.path.isdir(src), f"data source must be a directory: {src} invalid"

        ### parse patching info ###
        feat_name = os.path.basename(src)
        mag, patch_size = 20, 256
        check_params_same.append([feat_name, mag, patch_size])

        #### parse model name ####
        parsed = parse_model_name(feat_name) 
        parsed.update({'patch_mag': mag, 'patch_size': patch_size})
        
    ### Updated parsed mdoel parameters in args.Namespace ###
    for key, val in parsed.items():
        setattr(args, key, val)
    
    ### setup results dir ###
    if args.exp_code is None:
        exp_code = f"{args.split_name_clean}::{args.model_histo_config}::{feat_name}"
    else:
        pass
    
    args.results_dir = j_(args.results_dir, 
                          args.task, 
                          f'k={args.split_k}', 
                          str(exp_code), 
                          str(exp_code)+f"::{get_current_time()}")

    os.makedirs(args.results_dir, exist_ok=True)

    print("\n################### Settings ###################")
    for key, val in vars(args).items():
        print("{}:  {}".format(key, val))

    with open(j_(args.results_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    #### train ####
    results,result_cindex = main(args)
    
    print(results,result_cindex)
    print("FINISHED!\n\n\n")