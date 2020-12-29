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

import utils

#%%
def tokenize_encode(seqs, tags, tag2idx, tokenizer):
    inputs = tokenizer(seqs, is_split_into_words=True, return_offsets_mapping=True, padding=False, truncation=True)
    
    tags_enc = [[tag2idx[tag] for tag in record] for record in tags]  # convert tags to idxs
    tags_upt = []
    
    for doc_tags, doc_offset in zip(tags_enc, inputs.offset_mapping):       
        arr_offset = np.array(doc_offset)
        doc_tags_enc = np.ones(len(doc_offset), dtype=int) * -100  # create an empty array of -100
        # set tags whose first offset position is 0 and the second is not 0
        doc_tags_enc[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_tags
        tags_upt.append(doc_tags_enc.tolist())
    
    inputs.pop("offset_mapping") 
    inputs.update({'tags': tags_upt})

    return inputs

#%%
class EncodingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


    
class PadDoc():
    def __call__(self, batch):
        # Element in batch: {'input_ids': Tensor, 'attention_mask': Tensor, 'tags': Tensor}
        # Sort batch by seq_len in descending order
        # x['input_ids']: [seq_len]
        sorted_batch = sorted(batch, key=lambda x: len(x['input_ids']), reverse=True)
        
        # Pad within batch       		
        input_ids = [x['input_ids'] for x in sorted_batch]
        input_ids_padded = nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
        
        attn_masks = [x['attention_mask'] for x in sorted_batch]
        attn_masks_padded = nn.utils.rnn.pad_sequence(attn_masks, batch_first=True)
        
        tags = [x['tags'] for x in sorted_batch]
        tags_padded = nn.utils.rnn.pad_sequence(tags, batch_first=True)
        
        # Store length of each doc for unpad them later
        true_lens = torch.LongTensor([len(x) for x in tags])  
        
        return input_ids_padded, attn_masks_padded, tags_padded, true_lens


#%% Train
def train_fn(model, data_loader, idx2tag, optimizer, scheduler, tokenizer, clip, accum_step, device):
    
    batch_loss = 0
    len_iter = len(data_loader)
    
    model.train()
    optimizer.zero_grad()
    
    epoch_preds_padded, epoch_trues_padded = [], []
    with tqdm(total=len_iter) as progress_bar:      
        for j, batch in enumerate(data_loader):                      
            
            input_ids = batch[0].to(device)  # [batch_size, seq_len]
            attn_mask = batch[1].to(device)  # [batch_size, seq_len]
            tags = batch[2].to(device)  # [batch_size, seq_len]
            true_lens = batch[3]  # [batch_size]

            if type(tokenizer) == transformers.tokenization_distilbert.DistilBertTokenizerFast:
                outputs = model(input_ids, attention_mask = attn_mask, labels = tags) 
            if type(tokenizer) == transformers.tokenization_bert.BertTokenizerFast:
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
                   
            logits =  outputs[1]  # [batch_size, seq_len, num_tags]
            probs = F.softmax(logits, dim=2)  # [batch_size, seq_len, num_tags]
            preds = torch.argmax(probs, dim=2)  # [batch_size, seq_len]            
            # Append preds/trues with real seq_lens (before padding) to epoch_samaple_preds/trues
            for p, t, l in zip(preds, tags, true_lens):
                epoch_preds_padded.append(p[:l])  # p[:l].shape = l
                epoch_trues_padded.append(t[:l])  # t[:l].shape = l             
            progress_bar.update(1)
    
    # Remove ignored index (-100)
    epoch_trues_unpad, epoch_preds_unpad = [], []
    for trues, preds in zip(epoch_trues_padded, epoch_preds_padded):  # per sample
        trues_cut = [t.item() for (t, p) in zip(trues, preds) if t != -100]
        preds_cut = [p.item() for (t, p) in zip(trues, preds) if t != -100]
        
        epoch_trues_unpad.append(trues_cut)
        epoch_preds_unpad.append(preds_cut)
        
    # Convert epoch_idxs to epoch_tags
    epoch_tag_preds = utils.epoch_idx2tag(epoch_preds_unpad, idx2tag)
    epoch_tag_trues = utils.epoch_idx2tag(epoch_trues_unpad, idx2tag)
    # Calculate metrics for whole epoch
    scores = utils.scores(epoch_tag_trues, epoch_tag_preds)        
    scores['loss'] = batch_loss / len_iter
       
    return scores      


#%% Evaluate
def valid_fn(model, data_loader, idx2tag, tokenizer, device):
    
    batch_loss = 0
    len_iter = len(data_loader)
    
    model.eval()
    
    epoch_preds_padded, epoch_trues_padded = [], []
    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:      
            for j, batch in enumerate(data_loader):                      
                
                input_ids = batch[0].to(device)  # [batch_size, seq_len]
                attn_mask = batch[1].to(device)  # [batch_size, seq_len]
                tags = batch[2].to(device)  # [batch_size, seq_len]
                true_lens = batch[3]  # [batch_size]
    
                if type(tokenizer) == transformers.tokenization_distilbert.DistilBertTokenizerFast:
                    outputs = model(input_ids, attention_mask = attn_mask, labels = tags) 
                if type(tokenizer) == transformers.tokenization_bert.BertTokenizerFast:
                    outputs = model(input_ids, attention_mask = attn_mask, labels = tags)   
                                
                loss = outputs[0]
                batch_loss += loss.item()       
                               
                logits =  outputs[1]  # [batch_size, seq_len, num_tags]
                probs = F.softmax(logits, dim=2)  # [batch_size, seq_len, num_tags]
                preds = torch.argmax(probs, dim=2)  # [batch_size, seq_len]            
                # Append preds/trues with real seq_lens (before padding) to epoch_samaple_preds/trues
                for p, t, l in zip(preds, tags, true_lens):
                    epoch_preds_padded.append(p[:l])  # p[:l].shape = l
                    epoch_trues_padded.append(t[:l])  # t[:l].shape = l             
                progress_bar.update(1)                     
    
    # Remove ignored index (-100)
    epoch_trues_unpad, epoch_preds_unpad = [], []
    for trues, preds in zip(epoch_trues_padded, epoch_preds_padded):  # per sample
        trues_cut = [t.item() for (t, p) in zip(trues, preds) if t != -100]
        preds_cut = [p.item() for (t, p) in zip(trues, preds) if t != -100]
        
        epoch_trues_unpad.append(trues_cut)
        epoch_preds_unpad.append(preds_cut)
        
    # Convert epoch_idxs to epoch_tags
    epoch_tag_preds = utils.epoch_idx2tag(epoch_preds_unpad, idx2tag)
    epoch_tag_trues = utils.epoch_idx2tag(epoch_trues_unpad, idx2tag)
    # Calculate metrics for whole epoch
    scores = utils.scores(epoch_tag_trues, epoch_tag_preds)        
    scores['loss'] = batch_loss / len_iter

    return scores