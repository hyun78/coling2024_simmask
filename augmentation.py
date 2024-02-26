import random
from ctfidf import get_ctfidf_matrix
from dataset import load_dataframe, extract_post_targets
import numpy as np
from utils import logging_config, log
import re


class Criterion:
    
    def __init__(self, vocabulary = []):
        
        self.vocabulary = set(vocabulary)
    
    def __call__(self, word):
        
        return word.lower() in self.vocabulary

            
class CTfidfCriterion(Criterion):
    
    def __init__(self, dataframe, top, bottom, mode="offensive"):
        
        logging_config(filename='ctfidf.log')
        
        ctfidf_matrix, feature_names = get_ctfidf_matrix(dataframe)
        
        #offensive_ctfidf = ctfidf_matrix[0,:]
        #offensive_sorted_index = (-offensive_ctfidf).argsort()
        
        if mode == "both":
        
            sorted_indices = (-ctfidf_matrix.toarray()).argsort(axis=1)

            filtered_features = np.array(feature_names)[np.concatenate(sorted_indices[:,top:bottom], axis=0)]
            
            log("offensive ctfidf top {} ~ {}".format(top, bottom))
            log({feature : ctfidf_matrix[1,sorted_indices[1,i]] for i, feature in enumerate(filtered_features[:bottom - top])})
            log("non-offensive ctifidf top {} ~ {}".format(top, bottom))
            log({feature : ctfidf_matrix[0,sorted_indices[0,i]] for i, feature in enumerate(filtered_features[bottom - top:])})
            
        elif mode == "offensive":
            
            sorted_indices = (-ctfidf_matrix[1,:].toarray()).argsort(axis=1)
            
            filtered_features = np.array(feature_names)[sorted_indices[:, top:bottom]][0]
            
            print(filtered_features)
            
            #log("offensive ctfidf top {} ~ {}".format(top, bottom))
            #log({feature : ctfidf_matrix[1][i] for i, feature in enumerate(filtered_features)})
            #log(filtered_features)
            #log({feature : ctfidf_matrix[sorted_indices[i]] for i, feature in enumerate(filtered_features[sorted_indices])})
        
        
        
        
        #print(filtered_features[:20], filtered_features[500:520])
        #print("last word in each class:", filtered_features[499], filtered_features[999])
        
        super(CTfidfCriterion, self).__init__(vocabulary=filtered_features)
            
        
        
        
def token_conditional_insertion_one(args, corpus, to_insert, targets=None):
    """
    insert a token (to_insert) in a random position
    to_insert: a token(str) or a list of tokens to insert
    """

    new_corpus = []

    if type(to_insert) != str: # collection (list, array, ...)

        for i, sentence in enumerate(corpus):

            if targets[i] == 1: # offensive -> insert bad word 

                tokens = re.split(r'(\s+)', sentence)
                idx = random.randint(0, len(tokens))
                tokens.insert(idx, random.choice(to_insert))
                if tokens[idx-1].isspace():
                    tokens[idx] += ' '
                else:
                    tokens[idx] = ' ' + tokens[idx]

                new_sentence = ''.join(tokens)
                
                new_corpus.append(new_sentence)

                # for checking
                if i % 100 == 0:
                    print("label: " + str(targets[i]))
                    print("orig: " + sentence)
                    print("aug: " + new_sentence)
            
            
            else: 

                new_corpus.append(sentence) # non-offensive -> don't insert

    else: # fixed token

        for i, sentence in enumerate(corpus):

            if targets[i] == 1:
            
                tokens = re.split(r'(\s+)', sentence)
                idx = random.randint(0, len(tokens))
                tokens.insert(idx, to_insert)
                if tokens[idx-1].isspace():
                    tokens[idx] += ' '
                else:
                    tokens[idx] = ' ' + tokens[idx]
                
                new_corpus.append(''.join(tokens))
            
            else:

                new_corpus.append(sentence)
        
    return new_corpus
    

def token_replacement_one(args, corpus, replace_with):
    """
    replace a random token from each input
    : a token(str) or a list of tokens to insert
    FIXME generalize other augmentation functions as well (so that replace_with could either be a token or a list of tokens)
    """

    new_corpus = []

    if type(replace_with) != str: # collection (list, array, ...)

        for sentence in corpus:

            tokens = re.split(r'(\s+)', sentence)
            idx = random.randint(0, len(tokens) - 1)
            while tokens[idx].isspace():
                idx = random.randint(0, len(tokens) - 1)
            tokens[idx] = random.choice(replace_with)
            
            new_corpus.append(''.join(tokens))

    else: # fixed token

        for sentence in corpus:

            tokens = idx = random.randint(0, len(tokens) - 1)
            idx = random.randint(0, len(tokens) - 1)
            while tokens[idx].isspace():
                idx = random.randint(0, len(tokens) - 1)
            tokens[idx] = replace_with
            
            new_corpus.append(''.join(tokens))

    return new_corpus




def token_replacement(args, corpus, replace_with):
    
    if args.mode == 'random':
        return random_token_replacement(args, corpus, replace_with)
    
    elif args.mode == 'tfidf_offensive':
        # TODO define criterion
        
        df = load_dataframe(args.train_dataset)
        extract_post_targets(args.train_dataset, df)
        
        criterion = CTfidfCriterion(df, 0, 500, mode='offensive')
        return selective_token_replacement(args, corpus, criterion, replace_with)
        
    
    elif args.mode == 'tfidf_both':
        # TODO define criterion
        
        df = load_dataframe(args.train_dataset)
        extract_post_targets(args.train_dataset, df)
        
        criterion = CTfidfCriterion(df, 0, 500, mode='both')
        return selective_token_replacement(args, corpus, criterion, replace_with)

    
def random_token_replacement(args, corpus, replace_with):
    
    new_corpus = []
    r = args.masking_ratio #FIXME
    
    for sentence in corpus:
        
        new_corpus.append(''.join([replace_with if not word.isspace() and random.random()<r else word for word in re.split(r'(\s+)', sentence)]))
        
    return new_corpus
    
    
def selective_token_replacement(args, corpus, criterion, replace_with):
    
    new_corpus = []
    r = args.masking_ratio #FIXME
    
    for sentence in corpus:
        
        new_corpus.append(''.join([replace_with if not word.isspace() and criterion(word) and random.random() < r else word for word in re.split(r'(\s+)', sentence)]))
        
    return new_corpus
    

def masking(args, corpus):
    
    return token_replacement(args, corpus, '[MASK]')

"""
    
    if args.mode == 'random':
        return random_masking(args, corpus)
    
    elif args.mode == 'tfidf_offensive':
        # TODO define criterion
        return selective_masking(args, corpus, criterion)
    
    elif args.mode == 'tfidf_non-offensive':
        # TODO define criterion
        return selective_masking(args, corpus, criterion)
    
    
def random_masking(args, corpus):
    
    new_corpus = []
    r = args.masking_ratio
    
    for sent in corpus:
        
        new_corpus.append(' '.join(['[MASK]' if random.random()<r else word for word in sent.split()]))
        
    return new_corpus


#def selective_masking(corpus, bottom_n,top_n,tf_docs,ctfidf,r=0.8):

def selective_masking(args, corpus, criterion):
    
    new_corpus = []
    r = args.masking_ratio
    
    for sentence in corpus:
        
        new_corpus.append(' '.join(['[MASK]' if criterion(word) and random.random() < r else word for word in sentence.split()]))
        
    return new_corpus
    
    
"""


    