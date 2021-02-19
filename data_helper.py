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
    '''
        train_data = [train_seqs, train_tags]
            Element in train_seqs is a list of tokens for one text
            Element in train_tags is a list of tags for one text
            Eg: train_seqs[i] --> ['Leicestershire', '22', 'points', ',', 'Somerset', '4', '.']
                tags_seqs[i] --> ['B-ORG', 'O', 'O', 'O', 'B-ORG', 'O', 'O']        
    '''
    
    def __init__(self, train_data, valid_data, test_data=None):
        
        self.train_seqs, self.train_tags = train_data[0], train_data[1]
        self.valid_seqs, self.valid_tags = valid_data[0], valid_data[1]
        if test_data:
            self.test_seqs, self.tests_tags = test_data[0], test_data[1]
                     

    def build_vocab(self):
        ''' Obtain tag2idx, idx2tag, word2idx, idx2word (dictionaries) 
        '''
        
        # Get unique tags from train/valid tags
        all_tags = self.train_tags + self.valid_tags
        all_tags = [tag for sample in all_tags for tag in sample]
        tag_set = set(all_tags)
        # Create mapping for tags
        tag_sorted = sorted(list(tag_set), key=len)  # Sort set to ensure '0' is assigned to 0        
        self.tag2idx = {tag: idx for idx, tag in enumerate(tag_sorted)}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
    
   
        # Get unique words from train/valid seqs
        all_words = self.train_seqs + self.valid_seqs
        all_words = [word for sample in all_words for word in sample]
        word_set = set(all_words)
        # Create mapping for tokens
        self.word2idx = {word: idx+2 for idx, word in enumerate(word_set)}
        self.word2idx["<pad>"] = 0
        self.word2idx["<unk>"] = 1
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}


    def vectorizer(self, seqs, tags):
        '''
            seqs: list. Element is a list of tokens for one record
            tags: list. Element is a list of tags for one record
            Return: 
                seqs_vec (tags_vec) with same shape as seqs (tags)
                Element is a list of indexs for tokens (tags) in one record
        '''
        # Tokens vectorization
        seqs_vec = []
        for seq in seqs:
            word_idxs = []
            for w in seq:
                if w in self.word2idx:
                    w_idx = self.word2idx[w]
                elif w.lower() in self.word2idx:
                    w_idx = self.word2idx[w.lower()]
                else:
                    w_idx = self.word2idx['<unk>']
                word_idxs.append(w_idx)
            seqs_vec.append(word_idxs)
        
        # Tags vectorization
        tags_vec = []
        for tags_one_record in tags:
            tag_idxs = []
            for tag in tags_one_record:
                tag_idxs.append(self.tag2idx[tag])
            tags_vec.append(tag_idxs)
        
        return seqs_vec, tags_vec


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
        
        train_seqs_vec, train_tags_vec = self.vectorizer(self.train_seqs, self.train_tags)
        valid_seqs_vec, valid_tags_vec = self.vectorizer(self.valid_seqs, self.valid_tags)
        test_seqs_vec, test_tags_vec = self.vectorizer(self.test_seqs, self.tests_tags)
        
        ###### test only ######
        # train_seqs_vec, train_tags_vec = train_seqs_vec[:1000], train_tags_vec[:1000]
        # valid_seqs_vec, valid_tags_vec = valid_seqs_vec[:100], valid_tags_vec[:100]
        # self.valid_seqs_vec, self.valid_tags_vec = valid_seqs_vec, valid_tags_vec
        #######################         
         
        # Create dataset
        train_ds = tf.data.Dataset.from_generator(lambda: iter(zip(train_seqs_vec, train_tags_vec)), output_types=(tf.int32, tf.int32))
        valid_ds = tf.data.Dataset.from_generator(lambda: iter(zip(valid_seqs_vec, valid_tags_vec)), output_types=(tf.int32, tf.int32))
        test_ds = tf.data.Dataset.from_generator(lambda: iter(zip(test_seqs_vec, test_tags_vec)), output_types=(tf.int32, tf.int32))   
        
        # Shuffle train/valid data only   
        train_ds = train_ds.shuffle(buffer_size = len(train_seqs_vec), seed = seed, reshuffle_each_iteration=True)
        valid_ds = valid_ds.shuffle(buffer_size = len(valid_seqs_vec), seed = seed, reshuffle_each_iteration=True)
        
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

# valid_seqs_vec = helper.valid_seqs_vec
# valid_tags_vec = helper.valid_tags_vec