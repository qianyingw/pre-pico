#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 21:37:49 2020

@author: qwang
"""

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

import bert_utils

#%% Train
def train_fn(model, data_loader, idx2tag, optimizer, scheduler, tokenizer, clip, accum_step, device):
    
    batch_loss = 0
    len_iter = len(data_loader)
    
    model.train()
    optimizer.zero_grad()
    
    epoch_preds_unpad, epoch_trues_unpad = [], []
    with tqdm(total=len_iter) as progress_bar:      
        for j, batch in enumerate(data_loader):                      
            
            input_ids = batch[0].to(device)  # [batch_size, seq_len]
            attn_mask = batch[1].to(device)  # [batch_size, seq_len]
            tags = batch[2].to(device)  # [batch_size, seq_len]
            true_lens = batch[3]  # [batch_size]

            outputs = model(input_ids, attention_mask = attn_mask, labels = tags)     
                            
            loss = outputs[0]
            batch_loss += loss.item() 
                        
            loss = loss / accum_step  # loss gradients are accumulated by loss.backward() so we need to ave accumulated loss gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # prevent exploding gradients
                      
            # Gradient accumulation    
            if (j+1) % accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                   
            logits = outputs[1]  # [batch_size, seq_len, num_tags]
            # probs = F.softmax(logits, dim=2)  # [batch_size, seq_len, num_tags]
            preds = torch.argmax(logits, dim=2)  # [batch_size, seq_len]            
            # Append preds/trues with real seq_lens (before padding) to epoch_samaple_preds/trues
            for p, t, l in zip(preds, tags, true_lens):
                epoch_preds_unpad.append(p[1:l+1].tolist())  # p[:l].shape = l
                epoch_trues_unpad.append(t[1:l+1].tolist())  # t[:l].shape = l             
            progress_bar.update(1)
    
    # Remove ignored index (-100)
    epoch_preds_cut, epoch_trues_cut = [], []
    for preds, trues in zip(epoch_preds_unpad, epoch_trues_unpad):  # per sample       
        preds_cut = [p for (p, t) in zip(preds, trues) if t != -100]
        trues_cut = [t for (p, t) in zip(preds, trues) if t != -100] 
          
        epoch_preds_cut.append(preds_cut)
        epoch_trues_cut.append(trues_cut)
        
    # Convert epoch_idxs to epoch_tags
    epoch_tag_preds = bert_utils.epoch_idx2tag(epoch_preds_cut, idx2tag)
    epoch_tag_trues = bert_utils.epoch_idx2tag(epoch_trues_cut, idx2tag)
    # Calculate metrics for whole epoch
    scores = bert_utils.scores(epoch_tag_trues, epoch_tag_preds)        
    scores['loss'] = batch_loss / len_iter
       
    return scores      


#%% Evaluate
def valid_fn(model, data_loader, idx2tag, tokenizer, device):
    
    batch_loss = 0
    len_iter = len(data_loader)
    
    model.eval()
    
    epoch_preds_unpad, epoch_trues_unpad = [], []
    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:      
            for j, batch in enumerate(data_loader):                      
                
                input_ids = batch[0].to(device)  # [batch_size, seq_len]
                attn_mask = batch[1].to(device)  # [batch_size, seq_len]
                tags = batch[2].to(device)  # [batch_size, seq_len]
                true_lens = batch[3]  # [batch_size]
    
                outputs = model(input_ids, attention_mask = attn_mask, labels = tags)   
                                
                loss = outputs[0]
                batch_loss += loss.item()       
                               
                logits =  outputs[1]  # [batch_size, seq_len, num_tags]
                # probs = F.softmax(logits, dim=2)  # [batch_size, seq_len, num_tags]
                preds = torch.argmax(logits, dim=2)  # [batch_size, seq_len]            
                # Append preds/trues with real seq_lens (before padding) to epoch_samaple_preds/trues
                for p, t, l in zip(preds, tags, true_lens):
                    epoch_preds_unpad.append(p[1:l+1].tolist())  # p[:l].shape = l
                    epoch_trues_unpad.append(t[1:l+1].tolist())  # t[:l].shape = l             
                progress_bar.update(1)                     
    
    # Remove ignored index (-100)
    epoch_preds_cut, epoch_trues_cut = [], []
    for preds, trues in zip(epoch_preds_unpad, epoch_trues_unpad):  # per sample
        preds_cut = [p for (p, t) in zip(preds, trues) if t != -100]
        trues_cut = [t for (p, t) in zip(preds, trues) if t != -100]
               
        epoch_preds_cut.append(preds_cut)
        epoch_trues_cut.append(trues_cut)
        
    # Convert epoch_idxs to epoch_tags
    epoch_tag_preds = bert_utils.epoch_idx2tag(epoch_preds_cut, idx2tag)
    epoch_tag_trues = bert_utils.epoch_idx2tag(epoch_trues_cut, idx2tag)
    # Calculate metrics for whole epoch
    scores = bert_utils.scores(epoch_tag_trues, epoch_tag_preds)        
    scores['loss'] = batch_loss / len_iter

    return scores