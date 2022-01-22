# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:01:15 2021

@author: Ivana Nastasic
"""
import gensim.models
import logging
import os
import numpy as np


# Create or get the logger
logger = logging.getLogger(__name__)  

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('D2V.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

if (logger.hasHandlers()):
        logger.handlers.clear()

# add file handler to logger
logger.addHandler(file_handler)
    

class D2V:
    
    def __init__(self, load_model_path, save_model_path, model_name, prj_docs, empl_docs, mode, vec_size, 
                 epochs, window_size, neg_sampling, sub_sampling, dbow_words):
        
    
        self.prj_docs = prj_docs
        self.empl_docs = empl_docs
        self.load_model_path = load_model_path
        
        #------------model training parameters-----------#
        #If values are not passed then default Gensim values are used
        
        if mode == None:
            mode = 1 #PV-DM = 1, DBOW = 0
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
            
        if dbow_words == None:
            self.dbow_words = 0
        else:    
            self.dbow_words = dbow_words
       
        
        self.save_model_path = save_model_path
        self.model_name = model_name
            
    
    
    def prepare_TaggedDoc(self):
        
        '''Prepare data sets in format TaggedDocument used by word2vec gensim model'''
        
        for i, lw in enumerate(self.prj_docs):
            yield gensim.models.doc2vec.TaggedDocument(lw, [i])
            
    def get_trained_vecs(self):
        
        '''Get matrix of document vectors from model'''
        
        doc_vecs = []
        
        for i in range(len(self.model.docvecs)):
            doc_vecs.append(self.model.docvecs[i])
        
        return np.asarray(doc_vecs)
    
    def load_model(self):
        
        '''
            Loading model from given path
        
            Returns: matrix of embedding vectors for project documents
        '''
        
        logger.info('Loading model...')
            
        # read already trained model
        self.model = gensim.models.Doc2Vec.load(self.load_model_path)
        
            

    def  train(self):
        
        '''Create and save doc2vec model on data from project documents
        
            Returns: matrix of embedding vectors for project documents
        '''
        
        # Prepare data
        train_docs = self.prepare_TaggedDoc()
        
        # Train model
        
        logger.info('Training model...')
        
        self.model = gensim.models.doc2vec.Doc2Vec(documents=list(train_docs), vector_size=self.vec_size, 
                                                   window=self.window_size, alpha = 0.01, min_alpha = 0.0001, 
                                                   sample=self.sub_sampling, workers=-1, dm=self.mode, 
                                                   negative=self.neg_sampling, dbow_words=self.dbow_words, 
                                                   epochs=self.epochs)
        
        logger.info('Saving model...')
        # Save model
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        self.model.save(self.save_model_path+self.model_name)
            
        
        # Returns matrix of learned embedding vectors
        return self.get_trained_vecs()
    
    
        
    def infer(self, d):
        
        '''Infers document embeddings for a given document d.
        
            Args: d: list of lists of words in documents [[doc1],[doc2]...[docn]]
            Returns: matrix of document embedding vectors 
        
        '''
    
        doc_vecs = [self.model.infer_vector(s) for s in d]
        
        return np.asarray(doc_vecs)
        