#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:15:19 2020

@author: qwang
"""

import os
import re
import pandas as pd

import spacy
nlp = spacy.load("en_core_sci_sm")

#%%
def tagtog2conll(tsv_path, write2tsv=False):
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
        tsv_path_new = os.path.join(os.path.dirname(tsv_path), os.path.basename(tsv_path).split('.')[0]+'_conll.tsv')
        df_new.to_csv(tsv_path_new, sep='\t', header=False, index=False)
               
    return tokens, tags
                    
#%%
tsv_path = "tagtog/PMID22244441.tsv"
tokens, tags = tagtog2conll(tsv_path)                   
                