# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 22:16:37 2021

@author: Ivana Nastasic
"""

'''
    Run main_file.py from command line and pass path (location) of config.py file.
'''

N = 30 # Top N the most relevant recommended employees

# Row data location
data = {
        '''
            Projects data can be given as row file and preprocessing_flag should be set to True. In that case
            it is necessary that file contains at least these 2 columns: 'Description','Solution: Solution Name'.
            
            Otherwise if projects_data is already preprocessed, preprocessing_flag should be set to False.
            
            Employees data file doesn't need preprocessing. It has to contain columns with names: 'idc_personid_ext', 'skill_en'
            
        '''
        "projects_data": "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\data\\Projects_desc_cleaned.csv",
        #"projects_data": "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\data\\report1583489160483.csv", 
        "employee_data" : "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\data\\skills_clusters.csv"
        }

preprocessing_flag = False # Indicates if projects data should be preprocessed



if preprocessing_flag:
    # Location (directory) to save preprocessed file
    save_path = "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\data\\"
    
    # Project document translation file to be loaded
    load_trans_file = "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\data\\Ivana_Nastasic_challanges_translated.csv"
    # Project document file to be translated
    save_trans_path = "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\data\\"
    
    # Full path to the fastText language detection file
    lang_detect_file = "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\lid.176.ftz"
    
    '''
        Custom stop words should be defined in comma separated file (.csv) without header.
    '''
    
    custom_stop_words_flag = False
    
    #File containing custom stop words (full path)
    custom_stop_words_file = ""



    '''
        Possible method values are:
        LDA - training_flag = True indicates that we want to train LDA model otherwise we use already created model.
                Similarity between projects and skills is calculated based on the common words between reference topic words 
                and words used in skills.
        MTVL - Matching texts of varying lengths using hidden topics. It doesn't have training, every time it looks
                # for hidden topics and their weight within input project documents and calculates similarity
                # between projects and employees based on topic relevance
        
        SIF - Creates Smooth Inverse Frequency document vector embedding (training_flag=True indicates that we want
                to use our word embeddings model and in that case training parameters need to be provided).
                Result is cosine similarity between vectors of projects and employees.
                
        D2V - Creates paragraph embeddings for project documents. Value training_flag=True indicates that we want to train Doc2Vec model 
            and in that case training parameters need to be provided.
            Result is cosine similarity between doc2vec representation of projects and employees.
            
        WMD - Calculates Word Movers distance between project documents and employees (training_flag=True indicates that we want
             to train our word embeddings model and in that case training parameters need to be provided).
                
        WIth SIF and WMD method word vector embeddings can be trained only with fastText.
        Otherwise use pretrained models of type(Word2Vec, GloVe or fastText) 
        Pretrained models need to be in .bin or .vec format 
'''
method = "LDA"


'''
    Result always contains 2 columns: 
    'Solution: Solution Name': Solution ID (from the input file)
    'top_N_empl' - list of employee ids
'''

# Similarity calculation save results: path and file name
save_results_path = "D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\results\\"
save_results_name = "LDA_test_res"
                 
train_flag = False# indicates if it is training or inference


LDA = {
       
       '''
           It is enabled to train LDA model on whole projects dataset but to do calculate similarity only on selected number
           of projects from the bottom of projects input file. This way we can update the model but without loosing time
           to create suggestion list for old projects. This behaviour is controlled by parameter 'infer_n'. When it is set up
           to -1, prediction is done for all projects from input file.
       
       '''
        
       #Location where to save pretrained model, dictionary and topic reference words  
        "save_model_path":"D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\model\\", 
        
        #Location from where to load pretrained model, dictionary and topic reference words  
        "load_model_path":"D:\\Various documents\\Masters\\Subjects\\Thesis\\Code\\model\\", 
        
        # Name of the model (to save or to load)
        "model_name":"LDA_model",
        
        "K":90, # Number of topics for LDA model
        "infer_n":2, # Number of project documents from the end of input file for which we want to produce suggestion list, -1 means all
        "nw":200, # Number of words within topic to be taken into account
        "word_prob":0.002, # Threshold for word probability within the topic
        "topic_prob":0 # Threshold for topic probability within the project (0 is default and it means take all the topics)
      } 

MTVL = {
        
        "K":90, # Number of hidden topics
        "vec_emb_path":"D:\\Various documents\\Masters\\Subjects\\Thesis\\GoogleNews-vectors-negative300.bin",
        "binary_par":True, # indicates if model is saved as .bin
        "is_refine":False, # indicates if only the most important words based on tf-idf from projects and skills will be used
        "max_word_prj":1000, # number of the words to be taken into account in projects (if is_refine=True)
        "max_word_skill":100 # number of the words to be taken into account in employees skills (if is_refine=True)
      } 


SIF = { 
       
        "a_par" : 1e-3, # alpha value of the model
        
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
        "model_name":"D2V_model"  # name of the model to be saved
       }

WMD = {
       
        #Location of pretrained model       
        "load_model_path":"D:\\Various documents\\Masters\\Subjects\\Thesis\\GoogleNews-vectors-negative300.bin", # model file full path    
        "binary_flag":True, # indicates if model is saved as .bin
        
        # Parameters in case of training word embeddings model
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


    


