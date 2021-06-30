#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 22:15:03 2021

@author: qwang
"""

import os
import pandas as pd
import numpy as np


#%%
dat = pd.DataFrame(columns=['seg', 'tag'])
in_dir = '/home/qwang/pre-pico/data/tsv/18mar'
for file in os.listdir(in_dir):
    df = pd.read_csv(os.path.join(in_dir, file), sep='\t', header=None, names=['seg', 'tag'])
    
    df = df[df['seg'] != " "]  # remove rows with whitespace only
    df = df.dropna(subset=['seg'])  # remove nan seg rows
    df['tag'] = df['tag'].fillna('O')
    df = df[df['tag'] != "O"]  # remove non-pico rows
    dat = pd.concat([dat, df])
    
dat = dat.sort_values(by='tag', ascending=False)
dat = dat.reset_index(drop=True)
dat.to_csv('/home/qwang/pre-pico/data/tsv/pico_entities.csv', sep=',', header=True, index=False)


#%% t-SNE (bert)
from sentence_transformers import SentenceTransformer, models
pool_model = models.Pooling(word_embedding_dimension=768, pooling_mode_mean_tokens=True)   
 
embed_model = models.Transformer('bert-base-uncased', max_seq_length=512)     
mod1 = SentenceTransformer(modules=[embed_model, pool_model])

embed_model = models.Transformer('dmis-lab/biobert-v1.1', max_seq_length=512)
mod2 = SentenceTransformer(modules=[embed_model, pool_model])

embed_model = models.Transformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', max_seq_length=512)
mod3 = SentenceTransformer(modules=[embed_model, pool_model])

embed_model = models.Transformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', max_seq_length=512)
mod4 = SentenceTransformer(modules=[embed_model, pool_model])


vec1 = np.empty([1, 768])
vec2 = np.empty([1, 768])
vec3 = np.empty([1, 768])
vec4 = np.empty([1, 768])

dat = pd.read_csv('/home/qwang/pre-pico/data/tsv/pico_entities.csv', sep=',')
dat = dat[4000:]
for i, row in dat.iterrows():
    seg = [row['seg']]
    v = mod1.encode(seg, convert_to_tensor=True)
    vec1 = np.concatenate((vec1, v), axis=0)
    
    v = mod2.encode(seg, convert_to_tensor=True)
    vec2 = np.concatenate((vec2, v), axis=0)
    
    v = mod3.encode(seg, convert_to_tensor=True)
    vec3 = np.concatenate((vec3, v), axis=0)
    
    v = mod4.encode(seg, convert_to_tensor=True)
    vec4 = np.concatenate((vec4, v), axis=0)
    print(i)
    
with open('pico_base.npy', 'wb') as f:
    np.save(f, vec1)
with open('pico_bio.npy', 'wb') as f:
    np.save(f, vec2)
with open('pico_abs.npy', 'wb') as f:
    np.save(f, vec3)
with open('pico_full.npy', 'wb') as f:
    np.save(f, vec4)

#%% t-SNE (word2vec)
import spacy
nlp = spacy.load("en_core_sci_sm")
import gensim.models as models
w2v = models.KeyedVectors.load_word2vec_format('/media/mynewdrive/rob/wordvec/PubMed-and-PMC-w2v.bin', binary=True)  # w2v.vectors.shape

dat = pd.read_csv('/home/qwang/pre-pico/data/tsv/pico_entities.csv', sep=',')

vecs = np.empty([1, 200])
for i, row in dat.iterrows():
    words = [token.text for token in nlp(row['seg'])]
    vec = np.mean([np.vstack(w2v[word]) for word in words if word in w2v.vocab], axis=0)
    if isinstance(vec, np.ndarray) == False:
        vec = np.zeros((200, 1))
    vecs = np.concatenate((vecs, np.transpose(vec)), axis=0)
    print(i)
vecs = vecs[1:]
with open('pico_w2v.npy', 'wb') as f:
    np.save(f, vecs)














