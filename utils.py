#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:29:25 2020

@author: qwang
"""

import re
import numpy as np
import torch
import tensorflow as tf
from pathlib import Path

#%% Load conll data
def load_conll(filename):
    '''
    tokens_seqs : list. Each element is a list of tokens for one sequence/doc
                  tokens_seqs[21] --> ['Leicestershire', '22', 'points', ',', 'Somerset', '4', '.']
    tags_seqs : list. Each element is a list of tags for one sequence/doc
                tags_seqs[21] --> ['B-ORG', 'O', 'O', 'O', 'B-ORG', 'O', 'O']

    '''
    
    filename = Path(filename)

    raw_text = filename.read_text().strip()
    raw_seqs = re.split(r'\n\t?\n', raw_text)
    raw_seqs = [seq for seq in raw_seqs if '-DOCSTART' not in seq]

    tokens_seqs, tags_seqs = [], []
   
    for raw_seq in raw_seqs:
        seq = raw_seq.split('\n')
        tokens, tags = [], []
        for line in seq:
            splits = line.split(' ')
            tokens.append(splits[0])
            tags.append(splits[-1].rstrip("\n"))
        tokens_seqs.append(tokens)
        tags_seqs.append(tags)
    
    return [tokens_seqs, tags_seqs]
    
#%%
def epoch_idx2tag(epoch_sample_idxs, idx2tag):
    '''
        epoch_sample_idxs: list of tag lists with its true seq_len
                           len(epoch_sample_idxs) = n_samples
                           len(epoch_sample_idxs[0]) = true seq_len (varies among samples) 

        Convert epoch_idxs to epoch_tags      
    '''
        
    epoch_tags = []
    # idxs is list of index for a single text
    for idxs in epoch_sample_idxs:
        tags = [idx2tag[i] for i in idxs]
        epoch_tags.append(tags)
    
    return epoch_tags

#%%
from seqeval.metrics import f1_score, recall_score, precision_score, accuracy_score
def scores(epoch_trues, epoch_preds):
            
    f1 = f1_score(epoch_trues, epoch_preds)
    rec = recall_score(epoch_trues, epoch_preds)
    prec = precision_score(epoch_trues, epoch_preds)
    acc = accuracy_score(epoch_trues, epoch_preds)
    
    return {"f1": np.around(f1, 4), 
            "rec": np.around(rec, 4),  
            "prec": np.around(prec, 4), 
            "acc": np.around(acc, 4)}
    
    # return {"f1": f1, # np.around(f1, 4), 
    #         "rec": rec, #np.around(rec, 4),  
    #         "prec": prec, #np.around(prec, 4), 
    #         "acc": acc} #np.around(acc, 4)}

#%% Metric for pico sents classification
def metrics_fn(preds, y, th=0.5):
    """ preds: torch tensor, [batch_size, output_dim]
        y: torch tensor, [batch_size]
    """   
    if torch.cuda.device_count() == 1:
        y_preds = (preds[:,1] > th).int().type(torch.LongTensor).cuda()
    else:
        y_preds = (preds[:,1] > th).int().type(torch.LongTensor)
    ones = torch.ones_like(y_preds)
    zeros = torch.zeros_like(y_preds)
    
    pos = torch.eq(y_preds, y).sum().item()
    tp = (torch.eq(y_preds, ones) & torch.eq(y, ones)).sum().item()
    tn = (torch.eq(y_preds, zeros) & torch.eq(y, zeros)).sum().item()
    fp = (torch.eq(y_preds, ones) & torch.eq(y, zeros)).sum().item()
    fn = (torch.eq(y_preds, zeros) & torch.eq(y, ones)).sum().item()
    
    assert pos == tp + tn
    acc = pos / y.shape[0]  # torch.FloatTensor([y.shape[0]])
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn != 0) else 0
    rec = tp / (tp + fn) if (tp + fn != 0) else 0
    ppv = tp / (tp + fp) if (tp + fp != 0) else 0
    spc = tn / (tn + fp) if (tn + fp != 0) else 0
    return {'accuracy': acc, 'f1': f1, 'recall': rec, 'precision': ppv, 'specificity': spc}
