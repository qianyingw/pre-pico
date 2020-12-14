#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:54:11 2020

@author: qwang
"""

import tensorflow as tf
import spacy
nlp = spacy.load('en_core_sci_sm')

#%% Predict

def pred_one_text(text, word2idx, idx2tag, model):
    '''
    Return
        tup: list of tuples (token, tag)
    '''
    tokens = [t.text for t in nlp(text)]
    seqs = []
    for word in tokens:
        if word in word2idx:
            idx = word2idx[word]
        elif word.lower() in word2idx:
            idx = word2idx[word.lower()]
        else:
            idx = word2idx['<unk>']
        seqs.append(idx)
        
    seqs = tf.convert_to_tensor(seqs, dtype=tf.int32)  # [seq_len]
    seqs = tf.expand_dims(seqs, 0)  # [batch_size=1, seq_len]
    
    logits = model(seqs)  # [1, seq_len, num_tags]
    probs = tf.nn.softmax(logits, axis=2)  # [1, seq_len, num_tags]
    preds = tf.argmax(probs, axis=2)  # [1, seq_len]

    tags = [idx2tag[idx] for idx in preds.numpy()[0].tolist()]
    
    tup = []
    for token, tag in zip(tokens, tags):
        tup.append((token, tag))
    return tup

#%%
# text = '''Talks over a post-Brexit trade agreement will resume later, after the UK and EU agreed to "go the extra mile" in search of a breakthrough.'''
# print(pred_one_text(text, word2idx, idx2tag, model))