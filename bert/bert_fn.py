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
    
    epoch_preds_cut, epoch_trues_cut = [], []
    with tqdm(total=len_iter) as progress_bar:      
        for j, batch in enumerate(data_loader):                      
            
            input_ids = batch[0].to(device)  # [batch_size, seq_len]
            attn_mask = batch[1].to(device)  # [batch_size, seq_len]
            tags = batch[2].to(device)  # [batch_size, seq_len]
            true_lens = batch[3]  # [batch_size]
            word_ids = batch[4]   # [batch_size, seq_len]

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
            
            for sin_preds, sin_tags, sin_lens, sin_wids in zip(preds, tags, true_lens, word_ids):
                # list of lists (1st/last tag is -100 so need to move one step)
                sin_wids = sin_wids[1:sin_lens+1]
                sin_tags = sin_tags[1:sin_lens+1] 
                sin_preds = sin_preds[1:sin_lens+1]
                
                pre_wid = None
                sin_preds_new, sin_tags_new = [], []
                for p, t, wid in zip(sin_preds, sin_tags, sin_wids):
                    if wid != pre_wid:
                        sin_preds_new.append(p.tolist())
                        sin_tags_new.append(t.tolist())
                    pre_wid = wid
                epoch_preds_cut.append(sin_preds_new)   # list of lists                 
                epoch_trues_cut.append(sin_tags_new)  
                      
            progress_bar.update(1)
        
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
    
    epoch_preds_cut, epoch_trues_cut = [], []
    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:      
            for j, batch in enumerate(data_loader):                      
                
                input_ids = batch[0].to(device)  # [batch_size, seq_len]
                attn_mask = batch[1].to(device)  # [batch_size, seq_len]
                tags = batch[2].to(device)  # [batch_size, seq_len]
                true_lens = batch[3]  # [batch_size]
                word_ids = batch[4]   # [batch_size, seq_len]
    
                outputs = model(input_ids, attention_mask = attn_mask, labels = tags)   
                                
                loss = outputs[0]
                batch_loss += loss.item()       
                               
                logits =  outputs[1]  # [batch_size, seq_len, num_tags]
                # probs = F.softmax(logits, dim=2)  # [batch_size, seq_len, num_tags]
                preds = torch.argmax(logits, dim=2)  # [batch_size, seq_len]            
                for sin_preds, sin_tags, sin_lens, sin_wids in zip(preds, tags, true_lens, word_ids):
                    # list of lists (1st/last tag is -100 so need to move one step)
                    sin_wids = sin_wids[1:sin_lens+1]
                    sin_tags = sin_tags[1:sin_lens+1] 
                    sin_preds = sin_preds[1:sin_lens+1]
                    
                    pre_wid = None
                    sin_preds_new, sin_tags_new = [], []
                    for p, t, wid in zip(sin_preds, sin_tags, sin_wids):
                        if wid != pre_wid:
                            sin_preds_new.append(p.tolist())
                            sin_tags_new.append(t.tolist())
                        pre_wid = wid
                    epoch_preds_cut.append(sin_preds_new)   # list of lists                 
                    epoch_trues_cut.append(sin_tags_new)  
                          
                progress_bar.update(1)
        
    # Convert epoch_idxs to epoch_tags
    epoch_tag_preds = bert_utils.epoch_idx2tag(epoch_preds_cut, idx2tag)
    epoch_tag_trues = bert_utils.epoch_idx2tag(epoch_trues_cut, idx2tag)
    # Calculate metrics for whole epoch
    scores = bert_utils.scores(epoch_tag_trues, epoch_tag_preds)        
    scores['loss'] = batch_loss / len_iter

    return scores