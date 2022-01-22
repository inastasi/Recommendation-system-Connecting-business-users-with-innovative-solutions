# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:01:50 2021

@author: Ivana Nastasic
"""

# Libraries
import logging

import pandas as pd
import embeddings
import os
import sys


import utils
import dataset_preprocessing
import D2V
import SIF
import WMD
import LDA
import MTVL


if __name__ == '__main__':
    
    
    # Create or get the logger
    logger = logging.getLogger(__name__)  
    logging.captureWarnings(True)

    # set log level
    logger.setLevel(logging.INFO)
    
    
    # define file handler and set formatter
    file_handler = logging.FileHandler('mainlogfile.log')
    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    
    if (logger.hasHandlers()):
        logger.handlers.clear()

    # add file handler to logger
    logger.addHandler(file_handler)
    
    # Pass location of config file as sys path argument
    # Get full command-line arguments
    full_cmd_arguments = sys.argv


    try:
        argument_list = full_cmd_arguments[1:]
        config_path = sys.argv[1]
        # Put location of config file to the path
        sys.path.insert(1, config_path) 
        
        import config
    except:
        logger.error('Error: ', exc_info=True)
    
    # Get parameters
    logger.info('Reading configuration params...')
    
    method = config.method # Method (LDA/MTVL/SIF/D2V)
    train_flag = config.train_flag # Training or inference
    data = config.data # Input data
    save_results_path = config.save_results_path # Saving results location
    save_results_name = config.save_results_name # Name of the file to save results
    N = config.N # Number of employees in suggestion list
    
    # Check if exists a directory for saving results
    if not os.path.exists(save_results_path):
            os.makedirs(save_results_path)
    
    try:    
    #---------------------------------------Data------------------------------------------------------------#  
        # Prepare data frames
        
        logger.info('Preparing data...')
        
        
        
        
        if config.preprocessing_flag:
            # Check if exists a directory for saving file to be translated
            save_trans_path = config.save_trans_path
            if not os.path.exists(save_trans_path):
                os.makedirs(save_trans_path)
                
            # Check if exists a directory for saving preprocessed file
            save_path = config.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            dp_class = dataset_preprocessing.DataPreprocessing(data["projects_data"], config.load_trans_file, save_trans_path, 
                                        save_path, config.lang_detect_file, config.custom_stop_words_file, 
                                        config.custom_stop_words_file) 
            
            prj_data = dp_class.process()
        
        else:
            prj_data = pd.read_csv(data["projects_data"], engine ='python')
            
        prj_docs = (prj_data['Description'].apply(lambda x: x.split(' '))).to_list()[:2]
        
        # Read employees skills file
        empl_data = pd.read_csv(data["employee_data"])
        
        # Prepare employee's skills file
        empl = (empl_data.groupby('idc_personid_ext')['skill_en'].apply(lambda x: ' '.join(x.dropna()))
                .reset_index())
        empl['skill_list'] = empl['skill_en'].apply(lambda x: x.split())
        
        empl_docs = empl['skill_list'].to_list()
        
    #------------------------------------------LDA----------------------------------------------------------#
        if method == "LDA":
            logger.info('Executing LDA method...')
            
            LDA_par = config.LDA
            
            save_model_path = LDA_par["save_model_path"]
            load_model_path = LDA_par["load_model_path"]
            model_name = LDA_par["model_name"]
            K = LDA_par["K"]
            nw = LDA_par["nw"]
            word_prob = LDA_par["word_prob"]
            topic_prob = LDA_par["topic_prob"]
            infer_n = LDA_par["infer_n"]
            
            LDA_class = LDA.LDA_Class(load_model_path, save_model_path, model_name, prj_data, empl, N, K, 
                                      nw, word_prob, topic_prob, infer_n)
            
            if train_flag:
                logger.info('Creating LDA model...')
                # Create LDA model
                LDA_class.create_model()
                # Save model
                logger.info('Saving LDA model...')
                LDA_class.save_model()
                df = LDA_class.calculate_topics()
            else:
                logger.info('Loading LDA model...')
                #Load LDA model
                LDA_class.load_model()
                df = LDA_class.infer_topics()
            
            logger.info('LDA Calculate similarity...')
            sim = LDA_class.calculate_sim(df)  
            
            logger.info('LDA saving similarity results...')
            sim.to_csv(save_results_path + save_results_name + '.csv', index=False)
    
    #--------------------------------------------------------------------------------------------------------#
    #----------------------------------------MTVL------------------------------------------------------------#
        if method == "MTVL":
            
            logger.info('Executing Matching texts of varying lengths via hidden topics method...')
            
            MTVL_par = config.MTVL
            K = MTVL_par["K"] # Number of hidden topics
            vec_emb_path = MTVL_par["vec_emb_path"] # Location of pretrained word embeddings model
            binary_par = MTVL_par["binary_par"] # Indicates if word embedding model is in .bin format
            is_refine = MTVL_par["is_refine"] # Indicates if only the most important words based on tf-idf from projects and skills will be used
            max_word_prj = MTVL_par["max_word_prj"] # Number of the words to be taken into account in projects (if is_refine=True)
            max_word_skill = MTVL_par["max_word_skill"] # Number of the words to be taken into account in employees skills (if is_refine=True)
            
            # Create class object
            MTVL_class = MTVL.MTVL(vec_emb_path, binary_par, prj_data, empl, K, max_word_prj, max_word_skill, is_refine)               
            
            logger.info('MTVL Calculating similarity...')
            sim = MTVL_class.calculate_similarity()
            
            sim = utils.get_sim_empl_skills_list(sim,empl,N)
            fin_sim = pd.merge(prj_data[['Solution: Solution Name']], sim, left_index=True, right_index=True)
            
            logger.info('MTVL saving similarity results...')
            
            fin_sim.to_csv(save_results_path + save_results_name + '.csv', index=False)
    #--------------------------------------------------------------------------------------------------------#
    
    
    #----------------------------------------Doc2Vec---------------------------------------------------------#   
        if method == "D2V":
            logger.info('Executing Doc2Vec embedding...')
            
            # Read configuration parameters
            
            d2vec_par = config.D2V
            
            mode = d2vec_par["mode"] # DBOW = 0 or DMPV=1
            vec_size = d2vec_par["vec_size"] # embedding vector size
            epochs = d2vec_par["epochs"]
            window_size = d2vec_par["window_size"] #context window size
            neg_sampling = d2vec_par["neg_sampling"] # number of sampled words in negative sampling
            sub_sampling = d2vec_par["sub_sampling"] # threshold for configuring which higher-frequency words are randomly downsampled
            dbow_words = d2vec_par["dbow_words"] # in case of DBOW method if value is 1 word embeddings are trained too
            
            load_model_path = d2vec_par["load_model_path"]
            save_model_path = d2vec_par["save_model_path"]
            model_name = d2vec_par["model_name"]
            
            # Create class object
            v2d_class = D2V.D2V(load_model_path=load_model_path, save_model_path=save_model_path, model_name=model_name, 
                                prj_docs=prj_docs, empl_docs=empl_docs, mode=mode,vec_size=vec_size, epochs=epochs, window_size=window_size, neg_sampling=neg_sampling, sub_sampling=sub_sampling,dbow_words=dbow_words)
            
            
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
            logger.info('D2V calculating similarity...')
            sim = pd.DataFrame(utils.cos_sim(prj_doc_vecs,empl_doc_vecs))
            
            sim = utils.get_sim_empl_skills_list(sim,empl,N)
            fin_sim = pd.merge(prj_data[['Solution: Solution Name']], sim, left_index=True, right_index=True)
            
            logger.info('D2V saving similarity results...')
            
            fin_sim.to_csv(save_results_path + save_results_name + '.csv', index=False) 
     
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
                logger.info('SIF calculating similarity...')
                sim = pd.DataFrame(utils.cos_sim(prj_vecs,empl_vecs))
                
                sim = utils.get_sim_empl_skills_list(sim,empl,N)
                fin_sim = pd.merge(prj_data[['Solution: Solution Name']], sim, left_index=True, right_index=True)
                
                logger.info('SIF saving similarity results...')
                
                fin_sim.to_csv(save_results_path + save_results_name + '.csv', index=False) 
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
            logger.info('WMD calculating similarity...')
            sugg_list = WMD.get_sim_empl_skills_list(wmd_sim, empl, N)
            logger.info('WMD saving similarity results...')
                
            sugg_list.to_csv(save_results_path + save_results_name + '.csv', index=False) 
    #-------------------------------------------------------------------------------------------------------#
    except Exception:
        # Log errors
        logger.error("Error:", exc_info=True)