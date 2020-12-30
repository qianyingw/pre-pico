#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:40:19 2020

@author: qwang
"""


import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(description='RoB training and inference helper script')

    
    # Experiments
    parser.add_argument('--seed', nargs="?", type=int, default=1234, help='Seed for random number generator')
    parser.add_argument('--batch_size', nargs="?", type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', nargs="?", type=int, default=2, help='Number of epochs')    
    
    parser.add_argument('--lr', nargs="?", type=float, default=0.01, help='Adam learning rate')
    parser.add_argument('--warm_frac', nargs="?", type=float, default=0.1, help='Fraction of iterations when lr increased')
    parser.add_argument('--clip', nargs="?", type=float, default=0.1, help='Gradient clipping')
    parser.add_argument('--accum_step', nargs="?", type=int, default=4, help='Number of steps for gradient accumulation')
    
    parser.add_argument('--data_dir', nargs="?", type=str, default="/home/qwang/bioner/conll/data", help='Directory of info data files')
    parser.add_argument('--embed_path', nargs="?", type=str, default="/home/qwang/bioner/conll/data/glove.6B.100d.txt", help='Path of embedding')
    parser.add_argument('--exp_dir', nargs="?", type=str, default="/home/qwang/bioner/conll/exp", help="The directory of the model predictions and checkpoints saved")
    parser.add_argument('--pre_wgts', nargs="?", type=str, default="bert", 
                        choices=['distil', 'bert', 'biobert', 'pubmed-full', 'pubmed-abs'],
                        help='Pre-trained model name')
    
    parser.add_argument('--max_vocab_size', nargs="?", type=int, default=None, help='Maximum size of the vocabulary')
    parser.add_argument('--embed_dim', nargs="?", type=int, default=100, help='Embedding dim')
    parser.add_argument('--hidden_dim', nargs="?", type=int, default=64, help='Dim of lstm hidden states')
    
    parser.add_argument('--model', nargs="?", type=str, default='bert_crf', 
                        choices = ['bert', 'bert_crf', 'bert_lstm_crf', 'distil', 'distil_crf'], help='Models')

    args = parser.parse_args()
    
    return args