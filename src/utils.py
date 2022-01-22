# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:48:48 2021

@author: Ivana Nastaisc
"""

import numpy as np
import pandas as pd
from heapq import nlargest

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

def get_sim_empl_skills_list(sim,skills_empl,N=50):
    
    '''Creates data frame which contains for each project document list of correspinding top-N employees
        
        
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
    
    idx = np.argsort(sim.values, 1)[:, :] # indexes of the columns sorted ascending by similarity value
    df = pd.DataFrame(columns=['skills','empl_id','sim_val'],index=sim.index) 
    
    for i in sim.index:
        sim_list = sim.iloc[i,idx[i,:]]
        filtered_sim_list = sim_list[0:N]  # take only first N elements
        # Make a list of top N employee_id's for a given project
        empl_lab = skills_empl.iloc[idx[i,0:N]]['idc_personid_ext'].tolist()
        # List employee's skills
        empl_skills=[]
        for k in range(N):
            empl_skills.append(skills_empl.iloc[idx[i,k]]['skill_en'])
            
        #sim_list = ' '.join([str(x) for x in filtered_sim_list])

        df.iloc[i]['skills'] = '||'.join([str(x) for x in empl_skills]) 
        #df.iloc[i]['empl_id'] = ','.join([str(x) for x in empl_lab])
        df.iloc[i]['empl_id'] = empl_lab
        df.iloc[i]['sim_val'] = list(filtered_sim_list)
        
        # Dictionary {key: employee_id, value: score}
    df['empl_score_dict'] = df.apply(lambda x: dict(zip(x.empl_id, x.sim_val)), axis=1)
    df['top_N_empl'] = df['empl_score_dict'].apply(lambda x: nlargest(N, x, key = x.get)).apply(lambda x:','.join([str(y) for y in x]))
    
    return df[['top_N_empl']].copy()


