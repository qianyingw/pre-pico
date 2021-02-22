#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:58:37 2020

@author: qwang
"""

import re
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn


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
    
    return tokens_seqs, tags_seqs
    
#%%
# def tokenize_encode(seqs, tags, tag2idx, tokenizer):
#     inputs = tokenizer(seqs, is_split_into_words=True, return_offsets_mapping=True, padding=False, truncation=True)
    
#     tags_enc = [[tag2idx[tag] for tag in record] for record in tags]  # convert tags to idxs
#     tags_upt = []
    
#     for doc_tags, doc_offset in zip(tags_enc, inputs.offset_mapping):       
#         arr_offset = np.array(doc_offset)
#         doc_tags_enc = np.ones(len(doc_offset), dtype=int) * -100  # create an empty array of -100
#         # set tags whose first offset position is 0 and the second is not 0
#         doc_tags_enc[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_tags
#         tags_upt.append(doc_tags_enc.tolist())
    
#     inputs.pop("offset_mapping") 
#     inputs.update({'tags': tags_upt})

#     return inputs
# https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb

def tokenize_encode(seqs, tags, tag2idx, tokenizer, tag_all_tokens=True):
    inputs = tokenizer(seqs, is_split_into_words=True, return_offsets_mapping=True, padding=False, truncation=True)
    
    all_tags_enc_old = [[tag2idx[tag] for tag in record] for record in tags]  # convert tags to idxs
    all_tags_enc_new = []
    
    for i, tags_old in enumerate(all_tags_enc_old):    # per sample  
        word_ids = inputs.word_ids(batch_index=i)         
        tags_new = []
        pre_word_id = None
        for wid in word_ids:
            if wid is None:  # set tag to -100 for special token like [SEP] or [CLS]
                tags_new.append(-100)    
            elif wid != pre_word_id:  # Set label for the first token of each word
                tags_new.append(tags_old[wid])
            else:
                # For other tokens in a word, set the tag to either the current tag or -100, depending on the tag_all_tokens flag
                tags_new.append(tags_old[wid] if tag_all_tokens else -100)
            pre_word_id = wid
                
        all_tags_enc_new.append(tags_new)
    
    # inputs.pop("offset_mapping") 
    inputs.update({'tags': all_tags_enc_new})

    return inputs

#%%
class EncodingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


#%%   
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
        true_lens = torch.LongTensor([len(x)-2 for x in tags])  # [cls] and [sep] shouldn't be count into length
        
        return input_ids_padded, attn_masks_padded, tags_padded, true_lens
    
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
