# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:48:48 2021

@author: Ivana Nastaisc
"""

import numpy as np
import pandas as pd

def cos_sim(A,B):
    '''
    Code is taken from:
    https://towardsdatascience.com/cosine-similarity-matrix-using-broadcasting-in-python-2b1998ab3ff3
    
    It calculates cosine similarity between vectors in rows of matrices A and B.
    
    Args:
        A, B: arrays (matrices) for whose columns we want to calculate cosine similarity
    Returns:
    '''
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    
    np.seterr(divide='ignore', invalid='ignore')
    return num/(p1*p2)

def get_sim_empl_skills_list(sim, skills_empl, threshold=0.7):
    
    '''Creates data frame which contains for each project document list of correspinding employees
        with cosine similarity above given threshold.
        
        Args:
            sim: pandas data frame of cosine similarities between projects and employees, rows: projects and columns: employees
            skills_empl: pandas data frame of employees with their id's and skill descriptions
            threshold: float value for cosine similarity threshold
            
        Returns:
            data frame: rows: projects
                        columns: employee_id (list of employee ids)
                                 skills (list of skill descriptions)
                                 cosine similarity value (list of corresponding cosine similarities)
    '''
    
    idx = np.argsort(-sim.values, 1)[:, :] # indexes of the columns sorted by cosine similarity value
    df = pd.DataFrame(columns=['skills','empl_id','cosine_sim_val'],index=sim.index) 
    
    for i in sim.index:
        cosine_sim_list = sim.iloc[i,idx[i,:]]
        filtered_cosine_sim_list = [x for x in cosine_sim_list if x >= threshold ] # take only cosine similarity values above the threshold
        n = len(filtered_cosine_sim_list)
        empl_lab = skills_empl.iloc[idx[i,0:n]]['idc_personid_ext'].tolist()
        #empl_skills = skills_empl.iloc[idx[i,0:n]]['skill_en'].tolist() # Try to show different persons as separate string or not to show duplicated strings
        empl_skills=[]
        for k in range(n):
            empl_skills.append(skills_empl.iloc[idx[i,k]]['skill_en'])
            
        cosine_sim_list = ' '.join([str(x) for x in filtered_cosine_sim_list])

        df.iloc[i]['skills'] = '||'.join([str(x) for x in empl_skills])
        df.iloc[i]['empl_id'] = ' '.join([str(x) for x in empl_lab])
        df.iloc[i]['cosine_sim_val'] = cosine_sim_list
    return df