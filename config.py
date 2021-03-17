# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 22:16:37 2021

@author: Ivana Nastasic
"""

# Row data location
data = {
        # projects_data can be given as row in that case preprocessing_flag should be set to True
        # otherwise if projects_data is already preprocessed preprocessing_flag should be set to False
        "projects_data": "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\data\\Cleaned_input_all.csv", 
        "employee_data" : "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\data\\skills_clusters.csv"
        }

preprocessing_flag = False
# Saved preprocessed file
save_path = "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\data\\"

# Project document translation file to be loaded
load_trans_file = "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\data\\Ivana_Nastasic_challanges_translated.csv"
# Project document file to be translated
save_trans_path = "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\data\\"

# Custom stop words

custom_stop_words_flag = False
custom_stop_words_file = ""

# Similarity calculation save results
save_results_path = "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\results\\"
save_results_name = "wmd_test"

method = "WMD" # Possible method values are: 
#WMD - Calculates Word Movers distance between documents and employees (training_flag=True indicates that we want
        #our word embeddings model and in that case training parameters need to be provided )
#SIF - Creates Smooth Inverse Frequency document vector embedding (training_flag=True indicates that we want
        #our word embeddings model and in that case training parameters need to be provided )
        # As result comes cosine similarity between all projects and employees
# D2V - Creactes paragraph embeddings for documents ()(training_flag=True indicates that we want
        #our Doc2Vec model and in that case training parameters need to be provided)
                    
train_flag = False # indicates if it is training or inference


'''
WIth SIF and WMD method word vector embeddings can be trained only with fastText, 
Otherwise use pretrained models of type(Word2Vec, GloVe or fastText) 
Pretrained models need to be in .bin or .vec format 
'''

WMD = {
       
        #Location of pretrained model       
        "load_model_path":"D:\\Various documents\\Masters\\Subjects\\Thesis\\GoogleNews-vectors-negative300.bin", # model file full path    
        "binary_flag":True, # indicates if model is saved as .bin
        
        # Parameters in case of training the model
        #If values are not passed (None) then default Gensim values are used
        "mode": None, # Skip-Gram=1, CBOW=0
        "vec_size" : None, # embedding vector size
        "epochs" : None, # epochs
        "window_size" : None, # context window size
        "neg_sampling" : None, # number of sampled words in negative sampling
        "sub_sampling" : None, # threshold for configuring which higher-frequency words are randomly downsampled
        "save_model_path":"", #location to save the model
        "model_name":"" # name of the model to be saved
      }

SIF = { 
       
        "a_par" : 1e-3,
        
        #Pretrained model       
    
        "load_model_path":"D:\\Various documents\\Masters\\Subjects\\Thesis\\GoogleNews-vectors-negative300.bin", # model file full path
        "binary_embedding_flag":True, # indicates if model is saved as .bin
        
        # Parameters in case of training the model
        #If values are not passed (None) then default Gensim values are used
        "mode": 1, # Skip-Gram=1, CBOW=0
        "vec_size" : 300, # embedding vector size
        "epochs" : 500, # epochs
        "window_size" : 15, # context window size
        "neg_sampling" : 5, # number of sampled words in negative sampling
        "sub_sampling" : 1e-5, # threshold for configuring which higher-frequency words are randomly downsampled
        "save_model_path":"D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\model\\", #location to save the model
        "model_name":"ft_self_tr_mod" # name of the model to be saved
       }

D2V = {
       #Pretrained model  
       "load_model_path":"D:\\Various documents\\Masters\\Subjects\\Thesis\\enwiki_dbow\\doc2vec.bin", # model file full path
        
       
         # Parameters in case of training the model
         #If values are not passed (None) then default Gensim values are used
        "mode" : 0, # DBOW = 0 or DMPV=1
        "vec_size" : 300, # embedding vector size
        "epochs" : 10, # number of epochs
        "window_size" : 5, # context window size
        "neg_sampling" : 5, # number of sampled words in negative sampling
        "sub_sampling" : 1e-5, # threshold for configuring which higher-frequency words are randomly downsampled
        "dbow_words": 1, # in case of DBOW method if value is 1 word embeddings are trained too
        "save_model_path":"D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\model\\", #location to save the model
        "model_name":"example"  # name of the model to be saved
       }
   


    


