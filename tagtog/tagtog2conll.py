#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:15:19 2020

@author: qwang
"""

import os
import re
import json
import pandas as pd
from collections import Counter

import spacy
nlp = spacy.load("en_core_sci_sm")

#%%
def tagtog2conll(tsv_path, write2tsv=False, pmcid=None, out_dir=None):
    '''
    Convert tagtog EntitiesTsv to CoNLL format
    ----------
    tsv_path : file path of a sinlge tagtog EntitiesTsv file

    tokens: token list
    tags: tag list
    write2tsv: when True, output new tsv with CoNLL format
    -------
    '''
    # Read single annotation tsv
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['seg', 'tag'])
    df = df[df['seg'] != " "]  # remove rows with whitespace only
    df = df.dropna(subset=['seg'])  # remove nan seg rows
    df['tag'] = df['tag'].fillna('O')
    df = df.reset_index(drop=True)
    
    tokens, tags = [], []
    for _, row in df.iterrows():
        seg, tag = row['seg'], row['tag']
        
        # strip whitespace
        seg = re.sub(r'\s+', " ", seg)  
        seg = re.sub(r'^[\s]', "", seg)
        seg = re.sub(r'[\s]$', "", seg)
        # Scispacy tokenization
        seg_tokens = [t.text for t in nlp(seg)]
        
        # Split string segment in each row to tokens and assign tags
        for idx, st in enumerate(seg_tokens):
            tokens.append(st)         
            if tag == 'O':
                tags.append(tag) 
            else:
                if idx == 0:
                    tags.append('B-' + tag)
                else:
                    tags.append('I-' + tag)
     
    if write2tsv:
        df_new = pd.DataFrame({'token': tokens, 'tag': tags})
        if pmcid:
            tsv_path_new = os.path.join(out_dir, pmcid + '_conll.tsv')
        else:
            tsv_path_new = os.path.join(out_dir, os.path.basename(tsv_path).split('.')[0]+'_conll.tsv')
        df_new.to_csv(tsv_path_new, sep='\t', header=False, index=False)
               
    return tokens, tags


#%% Generate dictionary list of entities count for each sent from one abstract
def sent_ent_counter(tokens, tags, pid):
    ''' Input: tokens and entity tags from an abstract
    '''
    text = " ".join(tokens)
    sents = list(nlp(text).sents)  
    sents = [str(s) for s in sents]     
    
    sent_list = []
    
    s_count = 0
    tag_idx = 0
    for s in sents:
        s_tokens = s.split(' ')
        s_tags = tags[tag_idx: (tag_idx+len(s_tokens))]   
            
        count = dict(Counter(s_tags))
        keys = count.keys()
        # "I-Xxx" refers to the same entity as 'B-Xxx'
        freq_int = count['B-Intervention'] if 'B-Intervention' in keys else 0
        freq_com = count['B-Comparator'] if 'B-Comparator' in keys else 0
        freq_out = count['B-Outcome'] if 'B-Outcome' in keys else 0
        freq_ind = count['B-Induction'] if 'B-Induction' in keys else 0
        freq_spe = count['B-Species'] if 'B-Species' in keys else 0
        freq_str = count['B-Strain'] if 'B-Strain' in keys else 0   
        
        freq_ent = freq_int + freq_com + freq_out + freq_ind + freq_spe + freq_str
        freq_ent_tokens = len(s_tags) - count['O']
        
        sent_list.append({'sid': pid + '_' + str(s_count),
                          'sent': s_tokens, 'sent_tags': s_tags, 'freq_ent': freq_ent, 'freq_ent_tokens': freq_ent_tokens,
                          'freq_int': freq_int, 'freq_com': freq_com, 'freq_out': freq_out, 
                          'freq_ind': freq_ind, 'freq_spe': freq_spe, 'freq_str': freq_str})
        
        s_count += 1
        tag_idx += len(s_tokens)
        
    return sent_list
                    

#%%
in_dir = '/home/qwang/bioner/data/tagtog_tsv/b1'
out_dir = '/home/qwang/bioner/data/tagtog_tsv/output'

sent_ls = []
for file in os.listdir(in_dir):
    tsv_path = os.path.join(in_dir, file)
    pmcid = file.split("-")[1].split('_')[0]
    tokens, tags = tagtog2conll(tsv_path, write2tsv=True, pmcid=pmcid, out_dir=out_dir)
    sent_ls += sent_ent_counter(tokens, tags, pmcid)


# json output
with open(os.path.join(out_dir, 'b1.json'), 'w') as fout:
    for l in sent_ls:     
        fout.write(json.dumps(l) + '\n')  

# csv output
sent_df = []
for s in sent_ls:
    s.pop('sent_tags', None)    
    s['sent'] = " ".join(s['sent'])
    sent_df.append(s)

sent_df = pd.DataFrame(sent_df)
sent_df.to_csv(os.path.join(out_dir, 'b1.csv'), sep=',', index=False)



