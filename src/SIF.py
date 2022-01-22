# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:50:20 2021

@author: Ivana Nastasic
"""
# import libraries
import logging
import numpy as np

#import warnings
#warnings.simplefilter("ignore", DeprecationWarning)

from sklearn.decomposition import TruncatedSVD


# Create or get the logger
logger = logging.getLogger(__name__)  

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('SIF.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

if (logger.hasHandlers()):
        logger.handlers.clear()

# add file handler to logger
logger.addHandler(file_handler)


def compute_pc(X, npc=1):
    '''
    Compute the principal components.
    Args: 
        X: 2D array (matrix)
        npc: number of principle components
    Returns: principal components  
    '''
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    '''
    Remove the projection on the principal components
    
    Args: 
        X: 2D array (matrix)
        npc: number of principal components to remove
    Return: XX: X after removing its projection on principal components.
    '''
    
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX

def word_weights(W, num2word, word2count, a):
    '''
    Calculates word weights with formula a/(a+word_count).
    Args:
        W: array of word indexes
        num2word: dictionary of indexes for each word in vocabulary {key=index : value=word}
        word2count: dictionary of counts for each word in vocabulary {key=word : value=count}
        a: parameter
        Returns:
            weights: array of word weights
    '''
    words = list( map(num2word.get,W)) # get words from indexes
    counts = list(map(word2count.get, words)) # get counts from words
    
    return np.asarray(list(map(lambda x: a/(a+x),counts)))

def doc_words_emb(T,num2word,emb_mat,emb_lookup):
    
    '''
    Converts input document into a matrix where each row is a vector embedding of a word in a document.
    Args:
        T: list of indexes of words from a document
        num2word: dictionary of indexes for each word in vocabulary {key=index : value=word}
        emb_mat: 2D array (matrix) of available vector embeddings for words from vocabulary
        emb_lookup: dictionary of row indexes for words in embedding matrix ({key=word : value=row index}). Indicates
        in which row of embedding matrix is stored vector for a given word.
        
    Returns: 
        text_emb_mat: 2D array (matrix) where each row is a vector embedding of words from input document 
        oov_words: number of OOV in doc
    
    '''
    
    text_out = []
    oov_words = 0
    for t in T:
        word = num2word[t]
        if word in emb_lookup:
            text_out.append(emb_lookup[word])
        else:
            text_out.append(emb_lookup['<OOV>'])
            oov_words+=1
            
    text_emb_mat = emb_mat[text_out,:]
    
    return text_emb_mat, oov_words

def SIF(texts, mappings, emb_mat, emb_lookup, a = 1e-3):
    
    '''
    Converts input documents into a matrix where each row is a SIF vector embedding a document.
    Args:
        texts: list of lists of indexes of words from documents
        mappings: list of dictionaries [num2word,emb_mat,emb_lookup]
        
    Returns: 2D array (matrix) where each row is a SIF vector embedding of a document
    
    '''
    
    num2word = mappings[0]
    word2num = mappings[1]
    word2count = mappings[2]
    

    texts_emb_vec = []
    
    for t in texts: #for all documents
        
        # Create matrix of embedding vectors for words in a document
        t_emb, oov_words = doc_words_emb(t,num2word,emb_mat,emb_lookup)
        # Calculate word weights
        t_weights = word_weights(t, num2word, word2count, a)
        
        # Create vector embedding for a document
        if np.shape(t_emb)[0] == 0:
            # when doc doesn't conatin any word different from OOV
            t_v = np.zeros(np.shape(t_emb)[1])
        else:
            t_v = np.sum((t_emb.T * t_weights).T, axis=0)/np.shape(t_emb)[0] # multiply each word vector with it's weight and sum all of them and div by num_words
    
        texts_emb_vec.append(t_v)
    
    # Remove projection on the first principle component
    return remove_pc(np.asarray(texts_emb_vec), 1)

