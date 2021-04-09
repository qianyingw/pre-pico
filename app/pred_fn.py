#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:54:11 2020

@author: qwang
"""

import re
import json
from collections import defaultdict

import spacy
import torch
import pubmed_parser

from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification
from transformers import logging
logging.set_verbosity_error()

from bert_model import BERT_CRF, BERT_LSTM_CRF
nlp = spacy.load('en_core_sci_sm')
sent_tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract') 
sent_model = BertForSequenceClassification.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')

#%% PICO sentence detector
def sent_detect(text, pth_path): 
    # Split to sents and tokenization
    sents = list(nlp(text).sents)  
    sents = [str(s) for s in sents]     
    inputs = sent_tokenizer(sents, truncation=True, padding=True, return_tensors="pt")
    
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu') )
    state_dict = checkpoint['state_dict']
    sent_model.load_state_dict(state_dict, strict=False)
    sent_model.cpu()
    
    # Run model
    sent_model.eval()
    probs = sent_model(**inputs)['logits']
    preds = torch.argmax(probs, dim=1)
    preds = list(preds.data.cpu().numpy())
    
    # Cut non-pico sents and concatenate to new text
    sents_cut = [s for s, i in zip(sents, preds) if i == 1]
    text_cut = ' '.join(sents_cut)
    
    return text_cut

#%%
def pred_one_bert(text, mod, pre_wgts, pth_path, idx2tag):
    ''' tup: list of tuples (token, tag)
    '''
    n_tags = len(idx2tag)
    ## Tokenization
    pre_wgts = 'bert-base-uncased' if pre_wgts == 'base' else pre_wgts
    pre_wgts = 'dmis-lab/biobert-v1.1' if pre_wgts == 'biobert' else pre_wgts
    pre_wgts = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract' if pre_wgts == 'pubmed-abs' else pre_wgts
    pre_wgts = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' if pre_wgts == 'pubmed-full' else pre_wgts
 
    tokenizer = BertTokenizerFast.from_pretrained(pre_wgts, num_labels=n_tags) 
    inputs = tokenizer([text], is_split_into_words=True, return_offsets_mapping=True, padding=False, truncation=True)
    inputs = {key: torch.tensor(value) for key, value in inputs.items()} 

    ## Load model
    if mod == 'bert':
        model = BertForTokenClassification.from_pretrained(pre_wgts, num_labels=n_tags) 
    if mod == 'bert_crf':
        model = BERT_CRF.from_pretrained(pre_wgts, num_labels=n_tags)
    if mod == 'bert_lstm_crf':
        model = BERT_LSTM_CRF.from_pretrained(pre_wgts, num_labels=n_tags)
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cpu()
    model.eval()
    
    ## Run model
    if mod in ['bert']:
        outputs = model(inputs['input_ids'].unsqueeze(0), inputs['attention_mask'].unsqueeze(0)) 
        logits = outputs[0].squeeze(0)  # [seq_len, n_tags] 
        preds = torch.argmax(logits, dim=1)  # [seq_len]   
        preds = preds.numpy().tolist()[1:-1]  # len=seq_len-2, remove cls/sep token
    else:
        preds = model(inputs['input_ids'].unsqueeze(0), inputs['attention_mask'].unsqueeze(0)) 
        preds = preds[0]
        
    ids = inputs['input_ids'][1:-1]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    tags = [idx2tag[str(idx)] for idx in preds]
    
    # Record span start/end idxs
    sidxs, eidxs = [], []
    for i in range(len(tags)):
        if tags[0] != 'O':
            sidxs.append(0)
            if tags[1] == 'O':
                eidxs.append(0)     
                
        if i > 0 and i < len(tags)-1 and tags[i] != 'O':
            if tags[i-1] == 'O' and tags[i] != 'O':
                sidxs.append(i)
            if tags[i+1] == 'O' and tags[i] != 'O':
                eidxs.append(i)
        
        if tags[len(tags)-1] != 'O':
            sidxs.append(len(tags)-1)
            eidxs.append(len(tags)-1)

    tup = []
    for si, ei in zip(sidxs, eidxs):
        ent_tokens = tokens[si: ei+1]
        ent_tags = tags[si: ei+1]
        
        # ent_tags may include multiple type of tags
        ents = [t.split('-')[1] for t in ent_tags]
        ents_set = list(set(ents))        
        for ent in ents_set:
            indices = [idx for idx, t in enumerate(ent_tags) if t.split('-')[1] == ent]
            sub = [ent_tokens[ic] for ic in indices]
            sub_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(sub))
            
            sub_text = re.sub(r" - ", "-", sub_text)
            sub_text = re.sub(r" / ", "/", sub_text)
            sub_text = re.sub(r"\( ", "(", sub_text)
            sub_text = re.sub(r" \)", ")", sub_text)
            if "##" not in sub_text:
                tup.append((ent, sub_text))      
    return tup

#%% Convert tuple to entity dictionary and deduplication
def tup2dict(tup):
    ent_dict = defaultdict(list)
    for k, *v in tup:
        ent_dict[k].append(v[0])
    # Deduplicate
    for k, ls in ent_dict.items():
        ent_dict[k] = set({v.casefold(): v for v in ls}.values())  # Ignore case    
    return dict(ent_dict)