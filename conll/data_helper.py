#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:44:32 2020

@author: qwang
"""
import numpy as np
import tensorflow as tf

#%%
class DataBatch():
    '''train_data: data list with length=train_size
        Each element is a list of lists ([word, tag]) with length=seq_len, refers to one text
        Eg: train_data[i] refers 
                [['EU', 'B-ORG'],
                 ['rejects', 'O'],
                 ['German', 'B-MISC'],
                 ['call', 'O'],
                 ['.', 'O']]
        
    '''
    def __init__(self, train_data, valid_data, test_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        

    def build_vocab(self):
        ''' Obtain tag2idx, idx2tag, word2idx, idx2word (dictionaries) 
        '''
        word_set, tag_set = set(), set()
        for data in [self.train_data, self.valid_data]:  # [self.train_data, self.valid_data, self.test_data]
            for sent in data:
                for pair in sent:
                    word_set.add(pair[0])  # word_set.add(pair[0].lower())
                    tag_set.add(pair[1])
        
        # Create mapping for tags
        tag_sorted = sorted(list(tag_set), key=len)  # Sort set to ensure '0' is assigned to 0
        tag2idx = {}
        for tag in tag_sorted:
            tag2idx[tag] = len(tag2idx)  
        self.idx2tag = {v: k for k, v in tag2idx.items()}      
        self.tag2idx = tag2idx
        
        # Create mapping for tokens
        word2idx = {}
        word2idx["<pad>"] = 0
        word2idx["<unk>"] = 1
        for word in word_set:
            word2idx[word] = len(word2idx)
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.word2idx = word2idx


    def vectorizer(self, sub_data):
        ''' sub_data: train_data / valid_data / test_data
        '''
        sents = []
        tags = []
        for sent in sub_data:
            word_idxs, tag_idxs = [], []
            for w, t in sent:
                if w in self.word2idx:
                    w_idx = self.word2idx[w]
                elif w.lower() in self.word2idx:
                    w_idx = self.word2idx[w.lower()]
                else:
                    w_idx = self.word2idx['<unk>']
                    
                word_idxs.append(w_idx)
                tag_idxs.append(self.tag2idx[t])
                
            sents.append(word_idxs)
            tags.append(tag_idxs)      
        return sents, tags


    def load_embed(self, embed_path, embed_dim):
        ''' Generate embed matrix '''
        embeddings = {}
        with open(embed_path, encoding="utf-8") as fin:
            for line in fin:
                values = line.strip().split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings[word] = coefs
                
        embed_mat = np.zeros((len(self.word2idx), embed_dim))
        # Word embeddings for the tokens
        for word, idx in self.word2idx.items():
            vec = embeddings.get(word)
            if vec is not None:
                embed_mat[idx] = vec      
        return embed_mat
    
    
    
    def tf_pipe(self, seed, batch_size):
        ''' tf data pipeline'''
        self.build_vocab()
        
        train_seqs, train_tags = self.vectorizer(self.train_data)
        valid_seqs, valid_tags = self.vectorizer(self.valid_data)
        test_seqs, test_tags = self.vectorizer(self.test_data)
        
        ###### test only ######
        train_seqs, train_tags = train_seqs[:1000], train_tags[:1000]
        valid_seqs, valid_tags = valid_seqs[:100], valid_tags[:100]
        test_seqs, test_tags = test_seqs[:100], test_tags[:100]
        ####################### 
        
        # Create dataset
        train_ds = tf.data.Dataset.from_generator(lambda: iter(zip(train_seqs, train_tags)), output_types=(tf.int32, tf.int32))
        valid_ds = tf.data.Dataset.from_generator(lambda: iter(zip(valid_seqs, valid_tags)), output_types=(tf.int32, tf.int32))
        test_ds = tf.data.Dataset.from_generator(lambda: iter(zip(test_seqs, test_tags)), output_types=(tf.int32, tf.int32))   
        
        # Shuffle train/valid data only   
        train_ds = train_ds.shuffle(buffer_size = len(train_seqs), seed = seed, reshuffle_each_iteration=True)
        valid_ds = valid_ds.shuffle(buffer_size = len(valid_seqs), seed = seed, reshuffle_each_iteration=True)
        
        # Pad within batch
        # Components of nested elements (i.e. sents, tags) are padded independently ==>> padded_shapes=([None], [None])
        train_batches = train_ds.padded_batch(batch_size = batch_size, padded_shapes = ([None], [None]), padding_values = 0)
        valid_batches = valid_ds.padded_batch(batch_size = batch_size, padded_shapes = ([None], [None]), padding_values = 0)
        test_batches = test_ds.padded_batch(batch_size = batch_size, padded_shapes = ([None], [None]), padding_values = 0)
        
        return train_batches, valid_batches, test_batches


#%%        
# for ds in train_ds.take(5):
#     print(ds[0].shape)  # seq
#     print(ds[1].shape)  # tag

# for batch in train_batches.take(3):
#     # print(batch)
#     print("Batch seq shape: {}".format(batch[0].shape))  # seq batch: [batch_size, seq_len]
#     print("Batch tag shape: {}".format(batch[1].shape))  # tag batch: [batch_size, seq_len]
#     print("========================")        

# helper = DataBatch(train_data, valid_data, test_data)
# train_batches, valid_batches, test_batches = helper.tf_pipe(args.seed, args.batch_size)
# embed_path = '/home/qwang/bioner/conll/data/glove.6B.100d.txt'
# embed_mat = helper.load_embed(args.embed_path, args.embed_dim)
