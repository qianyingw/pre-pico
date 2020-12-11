#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:30:20 2020

@author: qwang
"""


import os
os.chdir("/home/qwang/bioner/conll")

import tensorflow as tf
import numpy as np

from arg_parser import get_args
import utils
from model import NerBiLSTM

#%% Load data
args = get_args()
train_data = utils.load_conll(os.path.join(args.data_dir, "train.txt"))
valid_data = utils.load_conll(os.path.join(args.data_dir, "valid.txt"))
test_data = utils.load_conll(os.path.join(args.data_dir, "test.txt"))

# idx_seq = 6
# idx_token_in_seq = 0
# train_data[idx_seq]
# train_data[idx_seq][idx_token_in_seq]


#%%  Build vocabulary (obtain tag2idx, idx2tag, word2idx, idx2word)
word_set = set()
tag_set = set()
for data in [train_data, valid_data]:  # [train_data, valid_data, test_data]
    for seq in data:
        for pair in seq:
            word_set.add(pair[0])  # word_set.add(pair[0].lower())
            tag_set.add(pair[1])

# Create mapping for tags
tag_sorted = sorted(list(tag_set), key=len)  # Sort set to ensure '0' is assigned to 0
tag2idx = {}
for tag in tag_sorted:
    tag2idx[tag] = len(tag2idx)  
idx2tag = {v: k for k, v in tag2idx.items()}

# Create mapping for tokens
word2idx = {}
word2idx["<pad>"] = 0
word2idx["<unk>"] = 1
for word in word_set:
    word2idx[word] = len(word2idx)
idx2word = {v: k for k, v in word2idx.items()}

#%% Vectorization   
def vectorizer(data, word2idx, tag2idx):
    seqs = []
    tags = []
    for sent in data:
        word_idxs, tag_idxs = [], []
        for w, t in sent:
            if w in word2idx:
                w_idx = word2idx[w]
            elif w.lower() in word2idx:
                w_idx = word2idx[w.lower()]
            else:
                w_idx = word2idx['<unk>']
                
            word_idxs.append(w_idx)
            tag_idxs.append(tag2idx[t])
            
        seqs.append(word_idxs)
        tags.append(tag_idxs)      
    return seqs, tags
          
train_seqs, train_tags = vectorizer(train_data, word2idx, tag2idx)
valid_seqs, valid_tags = vectorizer(valid_data, word2idx, tag2idx)
test_seqs, test_tags = vectorizer(test_data, word2idx, tag2idx)

#%% Load embeddings
# Loading glove embeddings
embeddings = {}
with open('data/glove.6B.100d.txt', encoding="utf-8") as fin:
    for line in fin:
        values = line.strip().split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coefs
        
embed_mat = np.zeros((len(word2idx), args.embed_dim))
# Word embeddings for the tokens
for word, idx in word2idx.items():
    vec = embeddings.get(word)
    if vec is not None:
        embed_mat[idx] = vec

del embeddings

#%% Data pipeline
# Create dataset
train_ds = tf.data.Dataset.from_generator(lambda: iter(zip(train_seqs, train_tags)), output_types = (tf.int32, tf.int32))
valid_ds = tf.data.Dataset.from_generator(lambda: iter(zip(valid_seqs, valid_tags)), output_types = (tf.int32, tf.int32))
test_ds = tf.data.Dataset.from_generator(lambda: iter(zip(test_seqs, test_tags)), output_types = (tf.int32, tf.int32))   

# Shuffle train/valid data only   
train_ds = train_ds.shuffle(buffer_size = len(train_seqs), seed = args.seed, reshuffle_each_iteration=True)
valid_ds = valid_ds.shuffle(buffer_size = len(valid_seqs), seed = args.seed, reshuffle_each_iteration=True)

# Pad within batch
# Components of nested elements (i.e. sents, tags) are padded independently ==>> padded_shapes=([None], [None])
train_batches = train_ds.padded_batch(batch_size = args.batch_size, padded_shapes = ([None], [None]), padding_values = 0)
valid_batches = valid_ds.padded_batch(batch_size = args.batch_size, padded_shapes = ([None], [None]), padding_values = 0)
test_batches = test_ds.padded_batch(batch_size = args.batch_size, padded_shapes = ([None], [None]), padding_values = 0)

# for ds in train_ds.take(5):
#     print(ds[0].shape)  # seq
#     print(ds[1].shape)  # tag

# for batch in train_batches.take(3):
#     # print(batch)
#     print("Batch seq shape: {}".format(batch[0].shape))  # seq batch
#     print("Batch tag shape: {}".format(batch[1].shape))  # tag batch
#     print("========================")
                               
#%% Define model, optimizer, loss   
vocab_size = len(word2idx)
if args.max_vocab_size:
    vocab_size = args.max_vocab_size
                     
model = NerBiLSTM(vocab_size = vocab_size,
                  embed_dim = args.embed_dim,
                  hidden_dim = args.hidden_dim,
                  output_dim = len(tag2idx),
                  embed_matrix = [embed_mat])

optimizer = tf.keras.optimizers.Adam(learning_rate = args.lr) 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Set "from_logits=True" may be more numerically stable

train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')

#%%
# @tf.function
def train_fn(model, batch_seq, batch_tag):
    with tf.GradientTape() as tape:
        logits = model(batch_seq, training=True)  # [batch_size, seq_len, output_dim]
        batch_loss = loss_fn(batch_tag, logits)
        
    grads = tape.gradient(batch_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.update_state(batch_loss)
    
    probs = tf.nn.softmax(logits, axis=2)  # [batch_size, seq_len, output_dim]
    preds = tf.math.argmax(probs, axis=2)  # [batch_size, seq_len]
    
    return preds

# @tf.function
def valid_fn(model, batch_seq, batch_tag):
    logits = model(batch_seq)
    batch_loss = loss_fn(batch_tag, logits)
    valid_loss.update_state(batch_loss)
    
    probs = tf.nn.softmax(logits, axis=2)  # [batch_size, seq_len, output_dim]
    preds = tf.math.argmax(probs, axis=2)  # [batch_size, seq_len]   
      
    return preds

#%% Run
for epoch in tf.range(1, args.epochs+1):
    
    ### Train ###
    epoch_preds, epoch_trues = [], []
    for batch_seq, batch_tag in train_batches:
        preds = train_fn(model, batch_seq, batch_tag)  # [batch_size, seq_len]
        
        # Convert tensor to ndarray to list
        preds = tf.make_ndarray(tf.make_tensor_proto(preds)).tolist()
        trues = tf.make_ndarray(tf.make_tensor_proto(batch_tag)).tolist()
        for i in preds:
            epoch_preds.append(i)
        for i in trues:
            epoch_trues.append(i)
            
    # Calculate metrics for whole epoch
    epoch_tag_preds, epoch_tag_trues = utils.idx2tag_fn(epoch_preds, epoch_trues, idx2tag)
    train_scores = utils.scores(epoch_tag_trues, epoch_tag_preds)
    
    
    ### Valid ###
    epoch_preds, epoch_trues = [], []
    for batch_seq, batch_tag in valid_batches:
        preds = valid_fn(model, batch_seq, batch_tag)
        # Convert tensor to ndarray to list
        preds = tf.make_ndarray(tf.make_tensor_proto(preds)).tolist()
        trues = tf.make_ndarray(tf.make_tensor_proto(batch_tag)).tolist()
        for i in preds:
            epoch_preds.append(i)
        for i in trues:
            epoch_trues.append(i)
    
    # Calculate metrics for whole epoch
    epoch_tag_preds, epoch_tag_trues = utils.idx2tag_fn(epoch_preds, epoch_trues, idx2tag)
    valid_scores = utils.scores(epoch_tag_trues, epoch_tag_preds)
        
    tf.print(tf.strings.format('[Epoch{}|train] loss: {}, f1: {}, rec: {}, prec: {}, acc: {}', 
                               (epoch, train_loss.result(), 
                                train_scores['f1'], train_scores['rec'], 
                                train_scores['prec'], train_scores['acc'])))
    
    tf.print(tf.strings.format('[Epoch{}|valid] loss: {}, f1: {}, rec: {}, prec: {}, acc: {}\n', 
                               (epoch, valid_loss.result(), 
                                valid_scores['f1'], valid_scores['rec'], 
                                valid_scores['prec'], valid_scores['acc'])))
    
    train_loss.reset_states()
    valid_loss.reset_states()
    

#%%
# 5 mins for 2 epochs
# Epoch=1 | [train] loss:0.00269821566, f1:0.98748645605387841, rec:0.988978254393804, prec:0.98599915146372508, acc:0.999363244 
# | [valid] loss:0.0487371907, f1:0.83901080904028835, rec:0.86216762032985528, prec:0.81706539074960127, acc:0.988969
# Epoch=2 | [train] loss:0.00198190822, f1:0.98791752489754314, rec:0.98991446444529552, prec:0.98592862592184449, acc:0.999393821 
# | [valid] loss:0.0602309704, f1:0.79987030882710541, rec:0.8303601480982834, prec:0.77154026583268176, acc:0.987153172