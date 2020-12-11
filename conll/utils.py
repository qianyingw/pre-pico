#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:29:25 2020

@author: qwang
"""

import numpy as np

#%% Load conll data
def load_conll(filename):
    f = open(filename)
    split_labeled_text = []
    sentence = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                split_labeled_text.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([splits[0],splits[-1].rstrip("\n")])
    
    if len(sentence) > 0:
        split_labeled_text.append(sentence)
        sentence = []
    
    return split_labeled_text


#%%
def idx2tag_fn(epoch_preds, epoch_trues, idx2tag):
    
    epoch_tag_preds = []
    for preds in epoch_preds:
        tag_preds = [idx2tag[i] for i in preds]
        epoch_tag_preds.append(tag_preds)

    epoch_tag_trues = []
    for trues in epoch_trues:
        tag_trues = [idx2tag[i] for i in trues]
        epoch_tag_trues.append(tag_trues)
    
    return epoch_tag_preds, epoch_tag_trues


#%%
from seqeval.metrics import f1_score, recall_score, precision_score, accuracy_score
def scores(epoch_tag_trues, epoch_tag_preds):
    f1 = f1_score(epoch_tag_trues, epoch_tag_preds)
    rec = recall_score(epoch_tag_trues, epoch_tag_preds)
    prec = precision_score(epoch_tag_trues, epoch_tag_preds)
    acc = accuracy_score(epoch_tag_trues, epoch_tag_preds)
    return {"f1": np.aounrd(f1, 4), 
            "rec": np.aounrd(rec, 4),  
            "prec": np.aounrd(prec, 4), 
            "acc": np.aounrd(acc, 4)}