# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 01:29:39 2021

@author: Ivana Nastasic
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import KeyedVectors
from numpy.linalg import svd
import copy
import functools

# Create or get the logger
logger = logging.getLogger(__name__)  
logging.captureWarnings(True)

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('MTVL.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

if (logger.hasHandlers()):
    logger.handlers.clear()


# add file handler to logger
logger.addHandler(file_handler)



class MTVL:
    
    def __init__(self, vec_emb_path, binary_par, prj_df, empl_df, K, max_word_prj, max_word_skill, is_refine=False):
        
    
        self.prj_df = prj_df  
        self.empl_df = empl_df
        self.K = K # number of topics
        self.max_word_prj = max_word_prj # Maximal number of words to be taken into account from project documents
        self.max_word_skill = max_word_skill # Maximal number of words to be taken into account from employee skills
        self.vec_emb_path = vec_emb_path
        self.binary_par = binary_par

        
        # Make projects in form of list of lists [[word1,word2,...,wordn],[word1,word2,...,wordk]...]
        self.prj_docs = (self.prj_df['Description'].apply(lambda x: x.split(' '))).to_list()
        self.empl_docs = self.empl_df['skill_list'].to_list()
        self.is_refine = is_refine
        self.word_vectors = KeyedVectors.load_word2vec_format(self.vec_emb_path, binary=self.binary_par)
    
    def findWordVectors(self, docs):
        '''
            Transform words into vectors using pretrained word embeddings embedding
        '''
        docs_vecs = []
        vector_dim = self.word_vectors.vector_size
        for word_seq in docs:
            vecs = []
            for word in word_seq:
                try:
                    vecs.append(self.word_vectors[word])
                except:
                    vecs.append(np.zeros(vector_dim))
                    continue
            vecs = np.array(vecs)
            docs_vecs.append(vecs[:])
        return docs_vecs
    
    def extract(self, docs):
        '''
        Extract max_word the most imprtant words from project documents based on the highest tf-idf score.
        Args:
            docs: list of lists of words [[word1, word2,...], [word3, word4, word5,...]]
            max_word: integer, maximal number of words to be taken into account
        Returns: numpy array (matrix) of words from project docs
        '''
     
        str_docs = [" ".join(word_seq) for word_seq in docs]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(str_docs)
        word_list = vectorizer.get_feature_names()
        count_matrix = X.toarray()
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf = transformer.fit_transform(count_matrix)
        tfidf_mat = tfidf.toarray()
        # sort words in each doc according to its tf-idf weight
        sorted_tfidf_mat = np.argsort(tfidf_mat, axis=1)
        sorted_tfidf_mat = np.fliplr(sorted_tfidf_mat)
        
        # take k the most informative words (with the highest tf-idf value)
        refined_docs = []
        doc_count, word_count = tfidf_mat.shape
        k = min(word_count, self.max_word_prj)
        
        # Create document matrix
        for doc_ind in range(doc_count):
            word_inds = [sorted_tfidf_mat[doc_ind][pos] for pos in range(k)]
            words = np.array([word_list[ind] for ind in word_inds])
            refined_docs.append(words[:])
        
        return np.array(refined_docs)
    
    
        
    def vectorizeLongDoc(self):
        
        '''
            Extract key topic vectors and their weights from project documents (long documents).
            Returns:
                docs_topics: 2D numpy array of topic vectors
                topic_weights: 2D numpy array of topic weights
        '''
        
        docs = self.prj_docs
        
        # If we want to take into account only the most important words based on tf-idf score
        if (self.is_refine):
            docs = self.extract(docs, self.max_word_prj)
        # Otherwise take all the words
        docs_topics, topic_weights = self.findHiddenTopics(docs)
        
        return docs_topics, topic_weights
        
    def vectorizeShortDoc(self):
        '''
            Represent words in employee's skills documents (short documents) in vector form.
            Returns:
                docs_vecs: numpy array of word vectors for project documents
        '''
        
        docs = self.empl_docs
        
        if (self.is_refine):
            docs = self.extract(self.empl_docs, self.max_word_skill)
            
        docs_vecs = self.findWordVectors(docs)
        return docs_vecs
    
    
    def getTopicRel(self, vecs, topic, weight):
        
        '''
            Calculate relevance between given word vector and topic vector with it's assigned weight.
            Args:
                vecs: word vec
        '''
        try:
            res = np.dot(np.mean(np.square(cosine_similarity(topic,vecs)),axis=1), weight)
        except ValueError:
            res = 0
        return (res)

    def normVector(self, seq):
        '''
            Normalize vector.
        '''
        var_seq = np.square(seq)
        norm = np.sum(var_seq)
        ratio_seq = 1.0 * var_seq / norm
        return ratio_seq
        
    def findHiddenTopics(self, docs):
        '''
            Find hidden topics and their weights in project documents.
            Args:
             docs: list of lists of words [[word1, word2,...], [word3, word4, word5,...]]
            Returns:
                docs_topics: 2D numpy array of topic vectors
                topic_weights: 2D numpy array of topic weights
        '''
        
        # Transform words into vectors
        docs_vecs = self.findWordVectors(docs)
        
        
        docs_topics = []
        weights = []
        for vecs in docs_vecs:
            #print(np.shape(vecs))
            component_num = min(self.K, len(vecs))
            U, s, V = svd(vecs, full_matrices=True)
            V_selected = V[:component_num, :]
            docs_topics.append(copy.deepcopy(V_selected))
            weights.append(self.normVector(s[:component_num]))
        return np.array(docs_topics), np.array(weights)

    def calculate_similarity(self):
        
        short_docs_vecs = self.vectorizeShortDoc()
        long_doc_vec = self.vectorizeLongDoc()
        
                
        # Calculate relevance between project and employee
        docs_skill_relevance = []
        for t,w in zip(long_doc_vec[0],long_doc_vec[1]):
            docs_skill_relevance.append(list(map(functools.partial(self.getTopicRel, topic=t,weight=w), short_docs_vecs)))
        
        doc_skill_rel = pd.DataFrame(docs_skill_relevance,columns=self.empl_df.index)
        
        return doc_skill_rel
 