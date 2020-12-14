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
import pandas as pd
from tqdm import tqdm

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

train_seqs, train_tags = train_seqs[:1000], train_tags[:1000]
valid_seqs, valid_tags = valid_seqs[:100], valid_tags[:100]
test_seqs, test_tags = test_seqs[:100], test_tags[:100]

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
#     print("Batch seq shape: {}".format(batch[0].shape))  # seq batch: [batch_size, seq_len]
#     print("Batch tag shape: {}".format(batch[1].shape))  # tag batch: [batch_size, seq_len]
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
    preds = tf.argmax(probs, axis=2)  # [batch_size, seq_len]
    
    return preds

# @tf.function
def valid_fn(model, batch_seq, batch_tag):
    logits = model(batch_seq)
    batch_loss = loss_fn(batch_tag, logits)
    valid_loss.update_state(batch_loss)
    
    probs = tf.nn.softmax(logits, axis=2)  # [batch_size, seq_len, output_dim]
    preds = tf.argmax(probs, axis=2)  # [batch_size, seq_len]   
      
    return preds

#%% Run
n_train_batches = len(list(train_batches))
n_valid_batches = len(list(valid_batches))

for epoch in tf.range(1, args.epochs+1):
    
    ### Train ###
    epoch_batch_preds, epoch_batch_trues = [], []
    with tqdm(total = n_train_batches) as progress_bar:
        for batch_seq, batch_tag in train_batches:
            preds = train_fn(model, batch_seq, batch_tag)  # [batch_size, seq_len]
            
            epoch_batch_preds.append(preds)  # list of tensors with shape [batch_size, seq_len]
            epoch_batch_trues.append(batch_tag)                    
            progress_bar.update(1)
            
    # Convert epoch_idxs to epoch_tags
    epoch_tag_preds = utils.epoch_idx2tag(epoch_batch_preds, idx2tag)
    epoch_tag_trues = utils.epoch_idx2tag(epoch_batch_trues, idx2tag)
    # Calculate metrics for whole epoch
    train_scores = utils.scores(epoch_tag_trues, epoch_tag_preds)    
    
    ### Valid ###
    epoch_batch_preds, epoch_batch_trues = [], []
    with tqdm(total = n_valid_batches) as progress_bar:
        for batch_seq, batch_tag in valid_batches:
            preds = valid_fn(model, batch_seq, batch_tag)
            
            epoch_batch_preds.append(preds)  # list of tensors with shape [batch_size, seq_len]
            epoch_batch_trues.append(batch_tag)                    
            progress_bar.update(1)
    
    # Convert epoch_idxs to epoch_tags
    epoch_tag_preds = utils.epoch_idx2tag(epoch_batch_preds, idx2tag)
    epoch_tag_trues = utils.epoch_idx2tag(epoch_batch_trues, idx2tag)
    # Calculate metrics for whole epoch
    valid_scores = utils.scores(epoch_tag_trues, epoch_tag_preds)
     
    
    tf.print(tf.strings.format('\n[Epoch{}|train] loss: {}, f1: {}, rec: {}, prec: {}, acc: {}', 
                               (epoch, train_loss.result(), 
                                train_scores['f1'], train_scores['rec'], 
                                train_scores['prec'], train_scores['acc'])))
    
    tf.print(tf.strings.format('[Epoch{}|valid] loss: {}, f1: {}, rec: {}, prec: {}, acc: {}\n', 
                               (epoch, valid_loss.result(), 
                                valid_scores['f1'], valid_scores['rec'], 
                                valid_scores['prec'], valid_scores['acc'])))
    
    train_loss.reset_states()
    valid_loss.reset_states()
    

# Save model weights    
model.save_weights(f"{args.exp_dir}/tf_model_wgts", save_format='tf')
# # Save model architecture and args   
# model.save(f"{args.exp_dir}/tf_model", save_format="tf")


#%% Evaluation on valid/test set (classification report)
from seqeval.metrics import classification_report
def cls_report(batches, wgt_file, cls_file):
    model.load_weights(os.path.join(args.exp_dir, wgt_file))
    epoch_batch_preds, epoch_batch_trues = [], []
    for batch_seq, batch_tag in batches:
        preds = valid_fn(model, batch_seq, batch_tag)
        epoch_batch_preds.append(preds)  # list of tensors with shape [batch_size, seq_len]
        epoch_batch_trues.append(batch_tag)  

    epoch_tag_preds = utils.epoch_idx2tag(epoch_batch_preds, idx2tag)
    epoch_tag_trues = utils.epoch_idx2tag(epoch_batch_trues, idx2tag)
    
    print(classification_report(epoch_tag_trues, epoch_tag_preds, output_dict=False))
    report = classification_report(epoch_tag_trues, epoch_tag_preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(args.exp_dir, cls_file), sep=',', header=True, index=True)

cls_report(valid_batches, wgt_file='tf_model_wgts', cls_file='cls_valid.csv')
cls_report(test_batches, wgt_file='tf_model_wgts', cls_file='cls_test.csv')

   
#%% Predict
from predict import pred_one_text
text = '''Talks over a post-Brexit trade agreement will resume later, after the UK and EU agreed to "go the extra mile" in search of a breakthrough.'''
seq_tagged = pred_one_text(text, word2idx, idx2tag, model)
for token, tag in seq_tagged:
    if tag != 'O':
        print(token, tag)
