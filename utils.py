#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:29:25 2020

@author: qwang
"""

import re
import json
import numpy as np
import torch
from pathlib import Path
from itertools import groupby

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

#%% Load pico json file
def load_pico(json_path, group='train'):
    '''
    tokens_seqs : list. Each element is a list of tokens for one abstract
                  tokens_seqs[i] --> ['Infected', 'mice', 'that', ...]
    tags_seqs : list. Each element is a list of tags for one abstract
                tags_seqs[i] --> [''O', 'B-Species', 'O', ...]
    '''
    dat = [json.loads(line) for line in open(json_path, 'r')] 
        
    tokens_seqs, tags_seqs = [], []
    for ls in dat:
        if ls['group'] == group:
              tokens_seqs.append(ls['sent'])
              tags_seqs.append(ls['sent_tags'])
    
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


#%%
def save_checkpoint(state, is_best, checkdir):
    """
    Save model and training parameters at checkpoint + 'last.pth.tar'. 
    If is_best==True, also saves checkpoint + 'best.pth.tar'
    Params:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkdir: (string) folder where parameters are to be saved
    """        
    filepath = os.path.join(checkdir, 'last.pth.tar')
    if os.path.exists(checkdir) == False:
        os.mkdir(checkdir)
    torch.save(state, filepath)    
    
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkdir, 'best.pth.tar'))
        
        
        
def load_checkpoint(checkfile, model, optimizer=None):
    """
    Load model parameters (state_dict) from checkfile. 
    If optimizer is provided, loads state_dict of optimizer assuming it is present in checkpoint.
    Params:
        checkfile: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """        
    if os.path.exists(checkfile) == False:
        raise("File doesn't exist {}".format(checkfile))
    checkfile = torch.load(checkfile)
    model.load_state_dict(checkfile['state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkfile['optim_dict'])
    
    return checkfile


def save_dict_to_json(d, json_path):
    """
    Save dict of floats to json file
    d: dict of float-castable values (np.float, int, float, etc.)
      
    """      
    with open(json_path, 'w') as fout:
        d = {key: float(value) for key, value in d.items()}
        json.dump(d, fout, indent=4)

