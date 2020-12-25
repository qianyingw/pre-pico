#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:29:25 2020

@author: qwang
"""

import re
import numpy as np
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
def epoch_idx2tag(epoch_batch_idxs, idx2tag):
    '''
        Flatten list of batch tensors with shape [batch_size, seq_len]
        Convert each batch tensor to ndarray, then to list
        append idxs of each sample to epoch_idxs
        Convert epoch_idxs to epoch_tags      
    '''
    
    epoch_idxs = []
    # Flatten list of batch tensors
    for batch_idxs in epoch_batch_idxs:
        # Convert each batch tensor to ndarray, then to list
        batch_idxs = tf.make_ndarray(tf.make_tensor_proto(batch_idxs)).tolist()  # len(batch_idxs) = batch_size, len(batch_idxs[0]) = seq_len (same within batch)
        for sample_idxs in batch_idxs:
            epoch_idxs.append(sample_idxs) # len(epoch_idxs) = n_samples, len(epoch_idxs[0]) = seq_len (varies among samples)  

    epoch_tags = []
    # idxs is list of index for a single text
    for idxs in epoch_idxs:
        tags = [idx2tag[i] for i in idxs]
        epoch_tags.append(tags)
    
    return epoch_tags

#%%
from seqeval.metrics import f1_score, recall_score, precision_score, accuracy_score
def scores(epoch_tag_trues, epoch_tag_preds):
    f1 = f1_score(epoch_tag_trues, epoch_tag_preds)
    rec = recall_score(epoch_tag_trues, epoch_tag_preds)
    prec = precision_score(epoch_tag_trues, epoch_tag_preds)
    acc = accuracy_score(epoch_tag_trues, epoch_tag_preds)
    return {"f1": np.around(f1, 4), 
            "rec": np.around(rec, 4),  
            "prec": np.around(prec, 4), 
            "acc": np.around(acc, 4)}

