# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:01:50 2021

@author: Ivana Nastasic
"""

# Libraries
import logging
import config
import pandas as pd
import utils
import embeddings
import os

import dataset_preprocessing
import  V2D
import SIF
import WMD

if __name__ == '__main__':
    
    # Create or get the logger
    logger = logging.getLogger(__name__)  

    # set log level
    logger.setLevel(logging.INFO)
    
    # define file handler and set formatter
    file_handler = logging.FileHandler('mainlogfile.log')
    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    
    # add file handler to logger
    logger.addHandler(file_handler)
    
    
    # Get parameters
    logger.info('Reading configuration params...')
    
    method = config.method 
    train_flag = config.train_flag
    data = config.data
    save_results_path = config.save_results_path
    save_results_name = config.save_results_name
    
    # Check if exists a directory for saving results
    if not os.path.exists(save_results_path):
            os.makedirs(save_results_path)

    
#---------------------------------------Data------------------------------------------------------------#  
    # Prepare data frames
    
    logger.info('Preparing data...')
    
    
    empl_data = pd.read_csv(data["employee_data"])
    
    if config.preprocessing_flag:
        # Check if exists a directory for saving file to be translated
        save_trans_path = config.save_trans_path
        if not os.path.exists(save_trans_path):
            os.makedirs(save_trans_path)
            
        dp_class = dataset_preprocessing.DataPreprocessing(data["projects_data"], config.load_trans_file, save_trans_path, 
                                    config.custom_stop_words_file, config.custom_stop_words_file) 
        
        prj_data = dp_class.process()
    
    else:
        prj_data = pd.read_csv(data["projects_data"], engine ='python')
    
    empl = (empl_data.groupby('idc_personid_ext')['skill_en'].apply(lambda x: ' '.join(x.dropna()))
            .reset_index())
    empl['skill_list'] = empl['skill_en'].apply(lambda x: x.split())
    
    prj_docs = (prj_data['nltk_tokens_pos_text'].apply(lambda x: x.split(' '))).to_list()[:2]
    empl_docs = empl['skill_list'].to_list()
    
#-------------------------------------------------------------------------------------------------------#
                
 #----------------------------------------Doc2Vec-------------------------------------------------------#   
    if method == "D2V":
        logger.info('Executing Doc2Vec embedding...')
        
        # Read configuration parameters
        
        d2vec_par = config.D2V
        
        mode = d2vec_par["mode"] # DBOW = 0 or DMPV=1
        vec_size = d2vec_par["vec_size"] # embedding vector size
        epochs = d2vec_par["epochs"]
        window_size = d2vec_par["window_size"]
        neg_sampling = d2vec_par["neg_sampling"]
        sub_sampling = d2vec_par["sub_sampling"] # threshold downsampling higher-frequency words are randomly downsampled
        dbow_words = d2vec_par["dbow_words"] # in case of DBOW method if value is 1
        
        load_model_path = d2vec_par["load_model_path"]
        save_model_path = d2vec_par["save_model_path"]
        model_name = d2vec_par["model_name"]
        
        # Create class object
        v2d_class = V2D.V2D(load_model_path=load_model_path, save_model_path=save_model_path, model_name=model_name, 
                            prj_docs=prj_docs, empl_docs=empl_docs,train_flag=train_flag, mode=mode,vec_size=vec_size, epochs=epochs, window_size=window_size, neg_sampling=neg_sampling, sub_sampling=sub_sampling,dbow_words=dbow_words)
        
        
        if(train_flag):
            # Get project document vectors from the model
            prj_doc_vecs = v2d_class.train()
        else:
            
            # Load model
            v2d_class.load_model()
            
            # Generate project documents vector embeddings
            prj_doc_vecs = v2d_class.infer(prj_docs)
            
        
        # Infer employee's skills document vectors from the model
        empl_doc_vecs = v2d_class.infer(empl_docs)
        

    
        # Calculate cosine similarity between project document vectors and employee's skills doc vectors
        logger.info('Calculating similarity...')
        sim = pd.DataFrame(utils.cos_sim(prj_doc_vecs,empl_doc_vecs))
        
        logger.info('Saving similarity results...')
        
        sim.to_csv(save_results_path + save_results_name + '.csv', index=False)     
#--------------------------------------------------------------------------------------------------------#
 
 
#---------------------------------SIF--------------------------------------------------------------------#

if method == "SIF":
        logger.info('Executing SIF model...')
        
        # Read configuration parameters
        
        sif_par = config.SIF
    
        
        a = sif_par["a_par"]
        load_model_path = sif_par["load_model_path"]
        binary_embedding_flag = sif_par["binary_embedding_flag"]
        
        # In case of training of word vector model
        mode = sif_par["mode"] # DBOW = 0 or DMPV=1
        vec_size = sif_par["vec_size"] # embedding vector size
        epochs = sif_par["epochs"] 
        window_size = sif_par["window_size"] # context window size
        neg_sampling = sif_par["neg_sampling"] # number of sampled words in negative sampling
        sub_sampling = sif_par["sub_sampling"] # threshold downsampling higher-frequency words are randomly downsampled
        save_model_path = sif_par["save_model_path"]
        model_name = sif_par["model_name"]
    
        # Create WordEmbeddings class for project proposals
        embd_prj = embeddings.WordEmbedding(docs=prj_docs, train_flag=train_flag, 
                                            binary_embedding_flag=binary_embedding_flag,
                                            embedding_file = load_model_path, mode=mode, vec_size=vec_size, epochs=epochs, 
                                            window_size=window_size, neg_sampling=neg_sampling, sub_sampling=sub_sampling, 
                                            save_model_path=save_model_path, model_name=model_name)
        
        #Create mappings and embeddings for projects
        mappings_prj, texts_prj = embd_prj.writeMappings() 
        emb_mat_prj, emb_lookup_prj = embd_prj.getEmbeddingsLookup()
        
        #Create mappings and embeddings for projects
        
        if train_flag:
            # Create WordEmbeddings class for employee's skills on newly trained word embeddings model 
            # Model was saved to save_model_path + model_name
            embd_empl = embeddings.WordEmbedding(docs=empl_docs, train_flag=False, binary_embedding_flag=False,
                                embedding_file = save_model_path + model_name)
        else:
            # Create WordEmbeddings class for employee's skills on loaded model
            embd_empl = embeddings.WordEmbedding(docs=empl_docs, train_flag=train_flag, 
                                                 binary_embedding_flag=binary_embedding_flag,
                                                 embedding_file = load_model_path)
          
        # Generate mappings and embeddings for employee's skills    
        mappings_empl, texts_empl = embd_empl.writeMappings() 
        emb_mat_empl, emb_lookup_empl = embd_empl.getEmbeddingsLookup()
        
        
        # Generate project documents vectors
        prj_vecs = SIF.SIF(texts_prj, mappings_prj, emb_mat_prj, emb_lookup_prj, a)
    
        # Generate employee's skills documents vectors
        empl_vecs = SIF.SIF(texts_empl, mappings_empl, emb_mat_empl, emb_lookup_empl, a)
        
        
        # Calculate cosine similarity between project document vectors and employee's skills doc vectors
        logger.info('Calculating similarity...')
        sim = pd.DataFrame(utils.cos_sim(prj_vecs,empl_vecs))
        
        logger.info('Saving similarity results...')
        
        sim.to_csv(save_results_path + save_results_name + '.csv', index=False) 
#---------------------------------------------------------------------------------------------------------------------#  

#------------------------------------------------WMD------------------------------------------------------------------#

if method == 'WMD':
    
    logger.info('Executing WMD model...')
        
    # Read configuration parameters
        
    wmd_par = config.WMD
    load_model_path = wmd_par["load_model_path"]
    binary_flag = wmd_par["binary_flag"]
        
        # In case of training of word vector model
    
    mode = wmd_par["mode"] # DBOW = 0 or DM-PV=1
    vec_size = wmd_par["vec_size"] # embedding vector size
    epochs = wmd_par["epochs"] 
    window_size = wmd_par["window_size"] # context window size
    neg_sampling = wmd_par["neg_sampling"] # number of sampled words in negative sampling
    sub_sampling = wmd_par["sub_sampling"] # threshold downsampling higher-frequency words are randomly downsampled
    save_model_path = wmd_par["save_model_path"]
    model_name = wmd_par["model_name"]
    
    wmd_class = WMD.WordMoverDistance(prj_docs, empl_docs, train_flag, load_model_path, binary_flag, mode,
                 vec_size, save_model_path, model_name, epochs,
                 window_size, neg_sampling, sub_sampling, 
                 normalize = True)
    
    wmd_sim = wmd_class.WMD()
    sugg_list = WMD.get_sim_empl_skills_list(wmd_sim, empl, 30)
    logger.info('Saving similarity results...')
        
    sugg_list.to_csv(save_results_path + save_results_name + '.csv', index=False) 
#-------------------------------------------------------------------------------------------------------#