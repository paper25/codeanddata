import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import ot
import numpy as np

from .components import SNN_Block, MMAttentionLayer, FeedForward, FeedForwardEnsemble, process_surv, Attn_Net_Gated

def init_per_path_model(omic_sizes, hidden_dim=256):
    """
    Create a list of SNNs, one for each pathway

    Args:
        omic_sizes: List of integers, each indicating number of genes per prototype
    """
    hidden = [hidden_dim, hidden_dim]
    sig_networks = []
    for input_dim in omic_sizes:
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
        sig_networks.append(nn.Sequential(*fc_omic))
    sig_networks = nn.ModuleList(sig_networks)

    return sig_networks

def agg_histo(X, agg_mode='mean'):
    """
    Aggregating histology
    """
    if agg_mode == 'mean':
        out = torch.mean(X, dim=1)
    elif agg_mode == 'cat':
        out = X.reshape(X.shape[0], -1)
    else:
        raise NotImplementedError(f"Not implemented for {agg_mode}")

    return out


def construct_proto_embedding(path_proj_dim, append_embed='modality', numOfproto_histo=16, numOfproto_omics=50):
    """
    Per-prototype learnable/non-learnable embeddings to append to the original prototype embeddings 
    """
    if append_embed == 'modality':  # One-hot encoding for two modalities
        path_proj_dim_new = path_proj_dim + 2

        histo_embedding = torch.tensor([[[1, 0]]]).repeat(1, numOfproto_histo, 1)  
        gene_embedding = torch.tensor([[[0, 1]]]).repeat(1, numOfproto_omics, 1)  

    elif append_embed == 'proto':
        path_proj_dim_new = path_proj_dim + numOfproto_histo + numOfproto_omics
        embedding = torch.eye(numOfproto_histo + numOfproto_omics).unsqueeze(0)

        histo_embedding = embedding[:, :numOfproto_histo, :]  
        gene_embedding = embedding[:, numOfproto_histo:, :]  

    elif append_embed == 'random':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim

        histo_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_omics, append_dim), requires_grad=True)

    else:
        path_proj_dim_new = path_proj_dim
        histo_embedding = None
        gene_embedding = None

    return path_proj_dim_new, histo_embedding, gene_embedding



################################
# Multimodal fusion approaches #
################################
class coattn(nn.Module):
    def __init__(
            self,
            omic_sizes=[100, 200, 300, 400, 500, 600],
            histo_in_dim=1024,
            dropout=0.1,
            num_classes=4,
            path_proj_dim=256,
            num_coattn_layers=1,
            modality='both',
            histo_agg='mean',
            histo_model='Gaussian',
            append_embed='none',
            mult=1,
            net_indiv=False,
            numOfproto=16):
        """
        The central co-attention module where you can do it all!
        """

        super().__init__()

        self.num_pathways = len(omic_sizes)
        self.num_coattn_layers = num_coattn_layers

        self.histo_in_dim = histo_in_dim
        self.out_mult = mult
        self.net_indiv = net_indiv
        self.modality = modality

        self.histo_agg = histo_agg

        self.numOfproto = numOfproto
        self.num_classes = num_classes

        self.histo_model = histo_model.lower()

        self.sig_networks = init_per_path_model(omic_sizes)
        self.identity = nn.Identity()  

        self.append_embed = append_embed
        self.text_proj_net = nn.Sequential(nn.Linear(768, path_proj_dim))
        self.allfeat_proj_net = nn.Sequential(nn.Linear(512, path_proj_dim))

        if self.histo_model == 'gaussian':  
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim * 2 + 1, path_proj_dim))
        else:
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim, path_proj_dim))

        if self.histo_model != "mil":
            self.path_proj_dim, self.histo_embedding, self.gene_embedding = construct_proto_embedding(path_proj_dim,
                                                                                                      self.append_embed,
                                                                                                      self.numOfproto,
                                                                                                      len(omic_sizes))
        else:
            self.path_proj_dim = path_proj_dim
            self.histo_embedding = None
            self.gene_embedding = None

        coattn_list = []
        if self.num_coattn_layers == 0:
            out_dim = self.path_proj_dim

            if self.net_indiv:  # Individual MLP per prototype
                feed_forward = FeedForwardEnsemble(out_dim,
                                                   self.out_mult,
                                                   dropout=dropout,
                                                   num=self.numOfproto + len(omic_sizes))
            else:
                feed_forward = FeedForward(out_dim, self.out_mult, dropout=dropout)

            layer_norm = nn.LayerNorm(int(out_dim * self.out_mult))
            coattn_list.extend([feed_forward, layer_norm])
        else:
            out_dim = self.path_proj_dim
            
            out_mult = self.out_mult
            
            if self.modality in ['histo', 'gene']: # If we want to use only single modality + self-attention
                attn_mode = 'self'
            elif self.modality == 'survpath':    # SurvPath setting H->P, P->H, P->P
                attn_mode = 'partial'
            else:
                attn_mode = 'full'  # Otherwise, perform self & cross attention

            cross_attender = MMAttentionLayer(
                dim=self.path_proj_dim,
                dim_head=out_dim,
                heads=1,
                residual=False,
                dropout=0.1,
                num_pathways=self.num_pathways,
                attn_mode=attn_mode
            )

            if self.net_indiv:  # Individual MLP per prototype
                feed_forward = FeedForwardEnsemble(out_dim,
                                                   out_mult,
                                                   dropout=dropout,
                                                   num=self.numOfproto +1 + len(omic_sizes)) # 
            else:
                feed_forward = FeedForward(out_dim, out_mult, dropout=dropout)

            layer_norm = nn.LayerNorm(int(out_dim * out_mult))
            coattn_list.extend([cross_attender, feed_forward, layer_norm])

        self.coattn = nn.Sequential(*coattn_list)

        #text co-attention
        coattn_list_text = []
        out_dim = self.path_proj_dim
        
        out_mult = self.out_mult

        cross_attender = MMAttentionLayer(
            dim=self.path_proj_dim,
            dim_head=out_dim,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=1,   
            attn_mode='full'
        )

        if self.net_indiv:  # Individual MLP per prototype
            feed_forward = FeedForwardEnsemble(out_dim,
                                                out_mult,
                                                dropout=dropout,
                                                num=self.numOfproto +1 + 1)   
        else:
            feed_forward = FeedForward(out_dim, out_mult, dropout=dropout)

        layer_norm = nn.LayerNorm(int(out_dim * out_mult))
        coattn_list_text.extend([cross_attender, feed_forward, layer_norm])
        

        self.coattn_text = nn.Sequential(*coattn_list_text)

        out_dim_final = int(out_dim * self.out_mult)
        histo_final_dim = out_dim_final * self.numOfproto if self.histo_agg == 'cat' else out_dim_final
        gene_final_dim = out_dim_final

        if self.modality == 'histo':
            in_dim = histo_final_dim
        elif self.modality == 'gene':
            in_dim = gene_final_dim
        else:
            in_dim = out_dim_final #trimodal only

        self.classifier = nn.Linear(in_dim*4, self.num_classes, bias=False)
        
        from .trans_mamaba import MambaMIL
        self.mil = MambaMIL(in_dim = in_dim, n_classes=self.num_classes, dropout=0, act='gelu', survival = True, layer = 2, rate = 5, type = 'Mamba')

    def forward_no_loss(self, x_path, x_omics,x_text, x_feat,return_attn=False):
        """
        Args:
            x_path: (B, numOfproto, in_dim) in_dim = [prob, mean, cov] (If OT, prob will be uniform, cov will be none)
            x_omics:
            x_text: (B, 1, 768)
            return_attn:

        """
        device = x_path.device
        
        h_omic = []  
        for idx, sig_feat in enumerate(x_omics):
            omic_feat = self.sig_networks[idx](sig_feat.float())  # (B, d)
            h_omic.append(omic_feat)
        h_omic = torch.stack(h_omic, dim=1)

        if self.gene_embedding is not None: 
            arr = []
            for idx in range(len(h_omic)):
                arr.append(torch.cat([h_omic[idx:idx + 1], self.gene_embedding.to(device)], dim=-1))
            h_omic = torch.cat(arr, dim=0)

        h_path = self.path_proj_net(x_path)

        if self.histo_embedding is not None:    
            arr = []
            for idx in range(len(h_path)):
                arr.append(torch.cat([h_path[idx:idx + 1], self.histo_embedding.to(device)], dim=-1))
            h_path = torch.cat(arr, dim=0)


        h_allfeat = self.allfeat_proj_net(x_feat)

        embedding_mamba =h_allfeat
        embedding_mamba=self.mil.forward_noclassifier(embedding_mamba).squeeze(1)
        
        embedding_mamba_token=embedding_mamba.unsqueeze(0).detach()
        tokens = torch.cat([h_omic,h_path,embedding_mamba_token], dim=1) # (B, N_p+N_h, d)
        tokens = self.identity(tokens)

        h_text = self.text_proj_net(x_text)
        tokens_text = torch.cat([h_text,h_path,embedding_mamba_token], dim=1)
        

        
        # Required for visualization
        if return_attn:
            with torch.no_grad():
                _, attn_pathways, cross_attn_pathways, cross_attn_histology = self.coattn[0](x=tokens, mask=None, return_attention=True)

        # Pass the token set through co-attention network
        mm_embed = self.coattn(tokens)
        mm_embed_text = self.coattn_text(tokens_text)

        # ---> aggregate
        # Pathways
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)
        
        # texts
        paths_postSA_embed_text = mm_embed_text[:,:1,:]
        paths_postSA_embed_text = torch.mean(paths_postSA_embed_text, dim=1)

        # Histology
        wsi_postSA_embed = h_path
        if self.histo_model == 'mil':
            wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)  # For non-prototypes, we just take the mean
        else:
            wsi_postSA_embed = agg_histo(wsi_postSA_embed, self.histo_agg)

        if self.modality == 'histo':    
            embedding = wsi_postSA_embed
        elif self.modality == 'gene':   
            embedding = paths_postSA_embed
        else:   # Use both modalities
            embedding = torch.cat([paths_postSA_embed,paths_postSA_embed_text ,wsi_postSA_embed], dim=1)  



        embedding = torch.cat([embedding,embedding_mamba], dim=1)
        logits = self.classifier(embedding)

        out = {'logits': logits,'x1':h_omic,'x2':h_text,'z1':paths_postSA_embed,'z2':paths_postSA_embed_text}
        if return_attn:
            out['omic_attn'] = attn_pathways
            out['cross_attn'] = cross_attn_pathways
            out['path_attn'] = cross_attn_histology

        return out


    def forward(self, x_path, x_omics, x_text,x_feat,return_attn=False, attn_mask=None, label=None, censorship=None, loss_fn=None):

        out = self.forward_no_loss(x_path, x_omics,  x_text,x_feat,return_attn)
        results_dict, log_dict = process_surv(out['logits'], label, censorship, loss_fn)
        if return_attn:
            results_dict['omic_attn'] = out['omic_attn']
            results_dict['cross_attn'] = out['cross_attn']
            results_dict['path_attn'] = out['path_attn']

        results_dict.update(out)
        return results_dict, log_dict
