import os
from mil_models import PANTHER

from mil_models import PANTHERConfig

import pdb
import torch
from utils.file_utils import save_pkl, load_pkl
from os.path import join as j_
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_embedding_model(args, mode='classification', config_dir='./configs'):
    """
    Create classification or survival models
    """
    config_path = os.path.join(config_dir, args.model_histo_config, 'config.json')
    assert os.path.exists(config_path), f"Config path {config_path} doesn't exist!"

    model_type = args.model_histo_type
    update_dict = {'in_dim': args.in_dim,
                   'out_size': args.n_proto,
                   'load_proto': args.load_proto,
                   'fix_proto': args.fix_proto,
                   'proto_path': args.proto_path}
    
    if mode == 'classification':
        update_dict.update({'n_classes': args.n_classes})
    elif mode == 'survival':
        if args.loss_fn == 'nll':
            update_dict.update({'n_classes': args.n_label_bins})
        elif args.loss_fn == 'cox':
            update_dict.update({'n_classes': 1})
        elif args.loss_fn == 'rank':
            update_dict.update({'n_classes': 1})
    elif mode == 'emb': # Create just slide-representation model
        pass
    else:
        raise NotImplementedError(f"Not implemented for {mode}...")

    if model_type == 'Gaussian':
        update_dict.update({'out_type': args.out_type})
        config = PANTHERConfig.from_pretrained(config_path, update_dict=update_dict)
        model = PANTHER(config=config, mode=mode)
    else:
        raise NotImplementedError(f"Not implemented for {model_type}!")

    return model


def create_multimodal_survival_model(args, omic_sizes=[]):
    if args.loss_fn == 'nll':
        num_classes = args.n_label_bins
    elif args.loss_fn == 'cox':
        num_classes = 1
    elif args.loss_fn == 'rank':
        num_classes = 1


    if args.model_mm_type == 'model_TMSE':
        from mil_models.model_TMSE import coattn
        model = coattn(omic_sizes=omic_sizes,
                       histo_in_dim=args.feat_dim,
                       path_proj_dim=256,
                       num_classes=num_classes,
                       num_coattn_layers=args.num_coattn_layers,
                       modality=args.model_mm_type,
                       histo_agg=args.histo_agg,                       
                       histo_model=args.model_histo_type,
                       append_embed=args.append_embed,
                       net_indiv=args.net_indiv,
                       )
    return model



def prepare_emb(datasets, args, mode='survival'):
    """
    Slide representation construction with patch feature aggregation trained in unsupervised manner
    """
   
    ### Preparing file path for saving embeddings
    print('\nConstructing unsupervised slide embedding...', end=' ')
    embeddings_kwargs = {
        'feats': args.data_source[0].split('/')[-2],
        'model_type': args.model_histo_type,
        'out_size': args.n_proto
    }

    # Create embedding path
    fpath = "{feats}_{model_type}_embeddings_proto_{out_size}".format(**embeddings_kwargs)
    if args.model_histo_type == 'Gaussian':
        PANTHER_kwargs = {'tau': args.tau, 'out_type': args.out_type, 'eps': args.ot_eps, 'em_step': args.em_iter}
        name = '_{out_type}_em_{em_step}_eps_{eps}_tau_{tau}'.format(**PANTHER_kwargs)
        fpath += name
    else:
        raise NotImplementedError(f"Not implemented for {args.model_histo_type}!")
    embeddings_fpath = j_(args.split_dir, 'embeddings', fpath+'.pkl')
    
    ### Load existing embeddings if already created
    if os.path.isfile(embeddings_fpath):
        embeddings = load_pkl(embeddings_fpath)
        for k, loader in datasets.items():
            print(f'\n\tEmbedding already exists! Loading {k}', end=' ')
            loader.dataset.X, loader.dataset.y = embeddings[k]['X'], embeddings[k]['y']
    else:
        os.makedirs(j_(args.split_dir, 'embeddings'), exist_ok=True)
        
        model = create_embedding_model(args, mode=mode).to(device)

        ### Extracts prototypical features per split
        embeddings = {}
        for split, loader in datasets.items():
            print(f"\nAggregating {split} set features...")
            X, y = model.predict(loader,
                                 use_cuda=torch.cuda.is_available()
                                 )
            loader.dataset.X, loader.dataset.y = X, y
            embeddings[split] = {'X': X, 'y': y}
        save_pkl(embeddings_fpath, embeddings)

    return datasets, embeddings_fpath