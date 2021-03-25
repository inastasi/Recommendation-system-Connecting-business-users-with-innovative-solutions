# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 01:29:39 2021

@author: Ivana Nastasic
"""

from gensim.models import KeyedVectors, FastText
import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd
from heapq import nlargest
import logging

# Create or get the logger
logger = logging.getLogger(__name__)  

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('WMD.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)

class WordMoverDistance():

    def __init__(self, prj_docs, empl_docs, train_flag, load_model_path, binary_flag, mode,
                 vec_size, save_model_path, model_name, epochs,
                 window_size, neg_sampling, sub_sampling, 
                 normalize = True):
        """Intialize an object from WordMoverDistance linked to a pretrained model,
        which enable to use Word Mover distance and its variations.
        
        Args:
            load_model_path {[string]} -- [a path to the pretrained Word2Vec model]
        
        Keyword Arguments:
            normalize {bool} -- [normalize all the vectors in the pretrained model or not ] (default: {True})
        """
        # input data in form list of list of words [[w1,...wn],[w1,...,wm]...]
        self.prj_docs = prj_docs
        self.empl_docs = empl_docs
        
        self.train_flag = train_flag
        
        self.normalize = normalize
        
        self.load_model_path = load_model_path
        
        self.binary_flag = binary_flag
        
        #------------model training parameters-----------#
        #If values are not passed then default Gensim values are used
        if mode == None: # Skip-Gram=1, CBOW=0
            self.mode = 0
        else:
            self.mode = mode 
        
        if vec_size == None:
            self.vec_size = 100
        else:
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
        
        self.save_model_path = save_model_path
        self.model_name = model_name
        
        if self.train_flag:
            
            # train word vector embeddings model
            self.model = FastText(self.docs, size=self.vec_size, window=self.window, alpha=0.01, min_alpha=0.0001, 
                                iter=self.epochs, workers=-1, sg=self.mode, negative=self.neg_sampling,
                         sample=self.sub_sampling)
            self.model.wv.save_word2vec_format(self.save_model_path+self.model_name, binary=False)
            
        else:    
            # Load vectors directly from the file
            self.model = KeyedVectors.load_word2vec_format(self.load_model_path, binary=self.binary_flag)
        
        
        self.model.init_sims(replace=normalize)
    
    def WMD_metric(self, x, y):
        """Calculate the word mover distance between two documents
        
        Args:
            doc_1 {[list of strings]} -- [Content of a document]
            doc_2 {[list of strings]} -- [Content of a document]
        
        Returns:
            [float] -- [word mover distance between two documents]
        """
        #return self.w2v_model.wmdistance(doc_1, doc_2)
    
        return self.model.wmdistance(self.prj_docs[int(x)], self.empl_docs[int(y)])
    
    def WMD(self):
        
        X = np.array([[i] for i in range(len(self.prj_docs))])
        Y = np.array([[i] for i in range(len(self.empl_docs))])
        
        wmd_dist = pd.DataFrame(pairwise_distances(X, Y, metric=self.WMD_metric,n_jobs=-1)).replace([np.inf, -np.inf], np.nan)
        
        return wmd_dist
    
def get_sim_empl_skills_list(sim, empl, N=50):

    idx = np.argsort(sim.values, 1)[:, :] # indexes of the columns sorted ascending by similarity value
    df = pd.DataFrame(columns=['skills','empl_id','sim_val'],index=sim.index) 
    
    for i in sim.index:
        sim_list = sim.iloc[i,idx[i,:]]
        filtered_sim_list = sim_list[0:N]  # take only first N elements
        n = len(filtered_sim_list)
        empl_lab = empl.iloc[idx[i,0:n]]['idc_personid_ext'].tolist()
        empl_skills=[]
        for k in range(n):
            empl_skills.append(empl.iloc[idx[i,k]]['skill_en'])
            
        #sim_list = ' '.join([str(x) for x in filtered_sim_list])
    
        df.iloc[i]['skills'] = '||'.join([str(x) for x in empl_skills])
        #df.iloc[i]['empl_id'] = ' '.join([str(x) for x in empl_lab])
        df.iloc[i]['empl_id'] = empl_lab
        df.iloc[i]['sim_val'] = list(filtered_sim_list)
        
    # Dictionary {key: employee_id, value: score}
    df['empl_score_dict'] = df.apply(lambda x: dict(zip(x.empl_id, x.sim_val)), axis=1)
    df['top_N_empl'] = df['empl_score_dict'].apply(lambda x: nlargest(N, x, key = x.get))
    return df
        