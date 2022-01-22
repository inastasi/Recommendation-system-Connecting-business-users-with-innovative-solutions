# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 01:53:17 2021

@author: Ivana Nastasic
"""

import numpy as np
import itertools
from collections import Counter
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from gensim.models import KeyedVectors, FastText



class WordEmbedding:
    def __init__(self, docs, train_flag, binary_embedding_flag, embedding_file, mode="", vec_size="", epochs="", 
                 window_size="", neg_sampling="", sub_sampling="", save_model_path="", model_name=""):
        
        '''
        docs: list of lists [[doc1],[doc2],...,[docn]]
        binary_embedding_flag: binary flag to indicate if embedding file is in .bin format
        embedding_file: embedding file location'''
        
        self.train_flag = train_flag
        
        #-----model training parameters------#
        
        #If values are not passed then default Gensim values are used
        if mode == None: # Skip-Gram=1, CBOW=0
            self.mode = 0
        else:
            self.mode = mode 
        
        if vec_size == None:
            self.vec_size = 100
        else:
            self.vec_size = vec_size
           
        
        self.vec_size = vec_size
        if epochs == None:
            self.epochs = 5 
        else:
            self.epochs = epochs
        if window_size == None:
            self.window_size = 5
        else:
            self.window_size = window_size
            
        if neg_sampling == None:
            self.neg_sampling = 5
        else:
            self.neg_sampling = neg_sampling
         
        if  sub_sampling == None:
            self.sub_sampling = 0.001
        else:
            self.sub_sampling = sub_sampling
        
        self.window_size = window_size
        self.neg_sampling = neg_sampling
        self.sub_sampling = sub_sampling
        self.save_model_path = save_model_path
        self.model_name = model_name
        #-------------------------------------
        
        self.docs = docs 
        self.binary_embedding_flag = binary_embedding_flag
        self.embedding_file = embedding_file
        
        
        self.word2num = {'<PAD>': 0, '<OOV>': 1} # Dictionary of words with associated index
        self.num2word = {0: '<PAD>', 1: '<OOV>'} # Inverted dictionary word2num
        self.word2count = {} # Dictionary with number of occurances in corpus of each word from vocab {word:num_occurances}
        
        

    def writeMappings(self):  
        
        '''
        Creates mapping dictionaries self.word2num, self.num2word, self.word2count and converts input corpus (list of 
        lists of words within each document) into a list of lists of associated word indexes.
        '''

        # Output variable
        texts = []
        
        # Dictionary with number of occurances in corpus of each word from vocab
        self.word2count = dict(Counter(itertools.chain(*self.docs)))
        
        # Read all documenst
        for d in self.docs:
            text = []
            #Read all words within document and place them into word dictionary associating index number
            for word in d:
                if word not in self.word2num.keys():
                    len_vocab = len(self.word2num)
                    self.word2num[word] = len_vocab
                    self.num2word[len_vocab] = word
                text.append(self.word2num[word])
            text = np.asarray(text)
                   
            texts.append(text)


        return [self.num2word, self.word2num, self.word2count], texts

    def getEmbeddingsLookup(self):
        
        '''Computes the embeddings lookup matrix and embedding matrix
        Embedding matrix is a matrix where each row is a vector representation of a word
        Embedding lookup matrix is a dictionary of row indexes of words in embedding matrix: 
        {key = word: value = row in word_embedding matrix}
        
        Returns: embeddings lookup matrix and embeddings matrix'''
        
        if self.train_flag:
            
            # train word vector embeddings model
            we_model = FastText(self.docs, size=self.vec_size, window=self.window, alpha=0.01, min_alpha=0.001, 
                                iter=self.epochs, workers=-1,sg=self.mode, negative=self.neg_sampling,
                         sample=self.sub_sampling)
            we_model.wv.save_word2vec_format(self.save_model_path+self.model_name, binary=False)
            
        else:    
            # Load vectors directly from the file
            we_model = KeyedVectors.load_word2vec_format(self.embedding_file, binary=self.binary_embedding_flag)
            
            
        vector_dim = we_model[list(we_model.vocab.keys())[0]].shape[0]
        embeddingsMatrix = [np.zeros(vector_dim), np.zeros(vector_dim)]
        embeddingsLookup = {'<PAD>': 0, '<OOV>': 1}
        for word in self.word2num.keys():
            if word in we_model.vocab:
                coeffs = np.asarray(we_model[word], dtype='float32')
                embeddingsMatrix.append(coeffs)
                embeddingsLookup[word] = len(embeddingsLookup)
        del we_model
        embeddingsMatrix = np.asarray(embeddingsMatrix)

        # Dump to file
        #with open("pickle_files/mappings/embeddingsMatrix.pkl", "wb") as f:
        #    pickle.dump(embeddingsMatrix, f)
        #with open("pickle_files/mappings/embeddingsLookup.pkl", "wb") as f:
        #    pickle.dump(embeddingsLookup, f)

        return embeddingsMatrix, embeddingsLookup