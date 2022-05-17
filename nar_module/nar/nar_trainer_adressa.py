from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Disabling GPU for local execution
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

from time import time
import tensorflow as tf
import json
import os
import re
import numpy as np
import pandas as pd
import tempfile
import sys
import logging

from sklearn.preprocessing import Normalizer

from .utils import deserialize, resolve_files, chunks, window, merge_two_dicts, log_elapsed_time, append_lines_to_text_file, min_max_scale
from .datasets import prepare_dataset_iterator
from .nar_model import ClickedItemsState, ItemsStateUpdaterHook, NARModuleModel
from .benchmarks import RecentlyPopularRecommender, ContentBasedRecommender, ItemCooccurrenceRecommender, ItemKNNRecommender, SessionBasedKNNRecommender, SequentialRulesRecommender

from .nar_utils import load_acr_module_resources, load_kg_module_resources, load_nar_module_preprocessing_resources, save_eval_benchmark_metrics_csv, \
        upload_model_output_to_gcs, dowload_model_output_from_gcs

import glob        

tf.logging.set_verbosity(tf.logging.INFO)

#Making results reproduceable
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)

#Model params
tf.flags.DEFINE_integer('batch_size', default=64, help='Batch size')
tf.flags.DEFINE_integer('truncate_session_length', default=20, help='Truncate long sessions to this max. size')
tf.flags.DEFINE_float('learning_rate', default=1e-3, help='Lerning Rate')
tf.flags.DEFINE_float('dropout_keep_prob', default=1.0, help='Dropout (keep prob.)')
tf.flags.DEFINE_float('reg_l2', default=0.0002, help='L2 regularization')
tf.flags.DEFINE_float('softmax_temperature', default=1.0, help='Initial value for temperature for softmax')
tf.flags.DEFINE_float('recent_clicks_buffer_hours', default=1.0, help='Number of hours that will be kept in the recent clicks buffer (limited by recent_clicks_buffer_max_size)')
tf.flags.DEFINE_integer('recent_clicks_buffer_max_size', default=500, help='Maximum size of recent clicks buffer')
tf.flags.DEFINE_integer('recent_clicks_for_normalization', default=500, help='Number of recent clicks to normalize recency and populary  novelty) dynamic features')
tf.flags.DEFINE_integer('eval_metrics_top_n', default=3, help='Eval. metrics Top N')
tf.flags.DEFINE_integer('CAR_embedding_size', default=512, help='CAR submodule embedding size')
tf.flags.DEFINE_integer('rnn_units', default=1024, help='Number of units of RNN cell')
tf.flags.DEFINE_integer('rnn_num_layers', default=1, help='Number of of RNN layers')
tf.flags.DEFINE_integer('train_total_negative_samples', default=5, help='Total negative samples for training')
tf.flags.DEFINE_integer('train_negative_samples_from_buffer', default=10, help='Training Negative samples from recent clicks buffer')
tf.flags.DEFINE_integer('eval_total_negative_samples', default=20, help='Total negative samples for evaluation')
tf.flags.DEFINE_integer('eval_negative_samples_from_buffer', default=50, help='Eval. Negative samples from recent clicks buffer')
tf.flags.DEFINE_bool('save_histograms', default=False, help='Save histograms to view on Tensorboard (make job slower)')
tf.flags.DEFINE_bool('disable_eval_benchmarks', default=False, help='Disable eval benchmarks')
tf.flags.DEFINE_bool('eval_metrics_by_session_position', default=False, help='Computes eval metrics at each position within session (e.g. 1st click, 2nd click)')
tf.flags.DEFINE_float('novelty_reg_factor', default=0.0, help='Popularity Regularization Loss (e.g. 0.1, 0.2, 0.3)')
tf.flags.DEFINE_float('diversity_reg_factor', default=0.0, help='Diversity (similarity) Regularization Loss (e.g. 0.1, 0.2, 0.3)')
tf.flags.DEFINE_float('eval_negative_sample_relevance', default=0.1, help='Relevance of negative samples within top-n recommended items for evaluation (relevance of positive sample is always 1.0)')
tf.flags.DEFINE_float('content_embedding_scale_factor', default=1.0, help='Factor to multiply the article content embeddings, as they might significantly affect the results')
tf.flags.DEFINE_integer('kge_vec_dim', default=100, help='Dimension of the Knowledge graph embedding')
tf.flags.DEFINE_string('kge_strategy', default='Append', help='{Append, Add, Concat}')
tf.flags.DEFINE_bool('enabled_gcn_refine_article_embedding', default=False, help='Enable GCN to refine the feature of article_embedding')
tf.flags.DEFINE_integer('gcn_layer1_size_article_embedding', default=150, help='Graph Convolutional Network layer1 embedding size of article_embedding')
tf.flags.DEFINE_integer('gcn_layer2_size_article_embedding', default=100, help='Graph Convolutional Network layer2 embedding size of article_embedding')

tf.flags.DEFINE_bool('enabled_gcn_refine_user_items_contextual', default=False, help='Enable GCN to refine the feature of user_items_contextual')
tf.flags.DEFINE_integer('gcn_layer1_size_user_items_contextual', default=100, help='Graph Convolutional Network layer1 embedding size of the user_items_contextual')
tf.flags.DEFINE_integer('gcn_layer2_size_user_items_contextual', default=80, help='Graph Convolutional Network layer2 embedding size of the user_items_contextual')

tf.flags.DEFINE_bool('enabled_gcn_refine_input_items_features', default=False, help='Enable GCN to refine the feature')
tf.flags.DEFINE_integer('gcn_layer1_size_input_items_features', default=250, help='Graph Convolutional Network layer1 embedding size of sinput_items_features')
tf.flags.DEFINE_integer('gcn_layer2_size_input_items_features', default=125, help='Graph Convolutional Network layer2 embedding size of input_items_features')

tf.flags.DEFINE_bool('enabled_sliding_hours', default=False, help='like skip-gram to capture feature beside hours')
tf.flags.DEFINE_bool('enabled_attention_LSTM', default=False, help='enable attention_lstam to capture more info in sequence')

tf.flags.DEFINE_list('enabled_clicks_input_features_groups',
                    default='ALL', help='Groups of input contextual features for user clicks, separated by comma. Valid values: ALL,NONE,time,device,location,referrer')
tf.flags.DEFINE_list('enabled_articles_input_features_groups',
                    default='ALL', help='Groups of input metadata features for articles, separated by comma. Valid values: ALL,NONE,category,author')                    
tf.flags.DEFINE_list('enabled_articles_kg_features',
                    default='ALL', help='Groups of knowledge graph features for articles, separated by comma. Valid values: ALL,NONE,entity,context')
tf.flags.DEFINE_list('enabled_internal_features',
                    default='ALL', help='Internal features. Valid values: ALL,NONE,recency,novelty,article_content_embeddings,item_clicked_embeddings')                                        

tf.flags.DEFINE_bool('enabled_word_graph_article_content_embedding', default=False, help='Enable pretrained article content embedding by GCN')


#Control params
#tf.flags.DEFINE_string('data_dir', default_value='./tmp',
#                    help='Directory where the dataset is located')
tf.flags.DEFINE_string('train_set_path_regex',
                    default='/train*.tfrecord', help='Train set regex')
tf.flags.DEFINE_string('acr_module_resources_path',
                    default='/pickles', help='ACR module resources path')
tf.flags.DEFINE_string('word_graph_embedding_resources_path',
                    default='/pickles', help='Word graph resources path')
tf.flags.DEFINE_string('kg_module_resources_path_entity',
                    default='/pickles', help='KG module resources path for Entity Embeddings')
tf.flags.DEFINE_string('kg_module_resources_path_context',
                    default='/pickles', help='KG module resources path for Context Embeddings')
tf.flags.DEFINE_string('nar_module_preprocessing_resources_path',
                    default='/pickles', help='NAR module preprocessing resources path')
tf.flags.DEFINE_string('model_dir', default='./tmp',
                    help='Directory where save model checkpoints')
tf.flags.DEFINE_string('warmup_model_dir', default=None,
                    help='Directory where model checkpoints of a previous job where output, to warm start this network training')

tf.flags.DEFINE_integer('train_files_from', default=0, help='Train model starting from file N')
tf.flags.DEFINE_integer('train_files_up_to', default=100, help='Train model up to file N')
tf.flags.DEFINE_integer('training_hours_for_each_eval', default=5, help='Train model for N hours before evaluation of the next hour')
tf.flags.DEFINE_integer('save_results_each_n_evals', default=5, help='Saves to disk and uploads to GCS (ML Engine) the incremental evaluation results each N evaluations')
tf.flags.DEFINE_bool('save_eval_sessions_negative_samples', default=True, help='Save negative samples of each session during evaluation')
tf.flags.DEFINE_bool('save_eval_sessions_recommendations', default=False, help='Save CHAMELEON recommendations log during evaluation')
tf.flags.DEFINE_bool('eval_cold_start', default=False, help='Evaluate item cold-start by recommender algorithm (WARNING: makes training much slower)')
tf.flags.DEFINE_bool('use_local_cache_model_dir', default=False, help='Persists checkpoints and events in a local temp file, copying to GCS in the end of the process (useful for ML Engine jobs, because saving and loading checkpoints slows training job)')
#Default param used by ML Engine to validate whether the path exists
tf.flags.DEFINE_string('job-dir', default='./tmp', help='Job dir to save staging files')

FLAGS = tf.flags.FLAGS
#params_dict = tf.app.flags.FLAGS.flag_values_dict()
#tf.logging.info('PARAMS: {}'.format(json.dumps(params_dict)))

ALL_FEATURES = 'ALL'

def get_articles_features_config(acr_label_encoders):
    articles_features_config = {
        #Required fields
        'article_id': {'type': 'categorical', 'dtype': 'int'},
        'created_at_ts': {'type': 'numerical', 'dtype': 'int'},
        #Additional metadata fields
        'category0': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 41},
        'category1': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 128},       
        'author': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 112},       
    }

    feature_groups = {
        'category': ['category0', 'category1'],
        'author': ['author'],
    }

    #Disabling optional features when required
    if FLAGS.enabled_articles_input_features_groups != [ALL_FEATURES]:   
        for feature_group in feature_groups:
            if feature_group not in FLAGS.enabled_articles_input_features_groups:
                for feature in feature_groups[feature_group]:
                    del articles_features_config[feature]

    #Adding cardinality to categorical features
    for feature_name in articles_features_config:
        if feature_name in acr_label_encoders and articles_features_config[feature_name]['type'] == 'categorical':
            articles_features_config[feature_name]['cardinality'] = len(acr_label_encoders[feature_name])

    tf.logging.info('Article Features: {}'.format(articles_features_config))       
    return articles_features_config

def get_articles_kg_features_config(kg_label_encoders):
    articles_kg_features_config = {
        'entity': True,
        'context': True
    }
    if FLAGS.enabled_articles_kg_features != [ALL_FEATURES]:
        for feature in articles_kg_features_config.copy().keys():
            if feature not in FLAGS.enabled_articles_kg_features:
                articles_kg_features_config.pop(feature)
    
    tf.logging.info('Article Knowledge Graph Features: {}'.format(articles_kg_features_config)) 
    tf.logging.info('Article Knowledge Graph Features Fusion Strategy: {}'.format(FLAGS.kge_strategy)) 
    return articles_kg_features_config

def process_kg_module_resources(entity_embeddings_matrix, context_embeddings_matrix):
    # 已知 article_encoded_id {3044, 30421 為 NaN} ,直接賦予 np.zero
    # 這裡是 hack 方式，正規解法市要回去 產出 embedding.vec 的地方修改, 因 mentioned_entity 為 [],當初計算沒考慮到
    ENTITY_EMBEDDING_DIM = FLAGS.kge_vec_dim #100
    entity_embeddings_matrix[3044] = np.zeros([ENTITY_EMBEDDING_DIM])
    entity_embeddings_matrix[30421] = np.zeros([ENTITY_EMBEDDING_DIM])
    context_embeddings_matrix[3044] = np.zeros([ENTITY_EMBEDDING_DIM])
    context_embeddings_matrix[30421] = np.zeros([ENTITY_EMBEDDING_DIM])

    # entity part
    entity_embeddings_matrix_np = np.vstack(entity_embeddings_matrix.values())
    embedding_for_padding_entity = np.mean(entity_embeddings_matrix_np, axis=0)
    entity_embeddings_matrix_np_with_padding = np.vstack([embedding_for_padding_entity, entity_embeddings_matrix_np])
    assert entity_embeddings_matrix_np_with_padding.shape[0] == len(entity_embeddings_matrix.keys())+1

    # context part
    context_embeddings_matrix_np = np.vstack(context_embeddings_matrix.values())
    embedding_for_padding_context = np.mean(context_embeddings_matrix_np, axis=0)
    context_embeddings_matrix_np_with_padding = np.vstack([embedding_for_padding_context, context_embeddings_matrix_np])
    assert context_embeddings_matrix_np_with_padding.shape[0] == len(context_embeddings_matrix.keys())+1

    tf.logging.info('KG article entity embeddings: {}'.format(entity_embeddings_matrix_np_with_padding.shape)) 
    tf.logging.info('KG article context embeddings: {}'.format(context_embeddings_matrix_np_with_padding.shape)) 
    return entity_embeddings_matrix_np_with_padding, context_embeddings_matrix_np_with_padding

def process_articles_metadata(articles_metadata_df, articles_features_config):
    articles_metadata = {}
    for feature_name in articles_features_config:
        articles_metadata[feature_name] = articles_metadata_df[feature_name].values
        #Appending a row in the first position to correspond to the <PAD> article #
        # (so that it correspond to content_article_embeddings_matrix.shape[0])
        articles_metadata[feature_name] = np.hstack([[0], articles_metadata[feature_name]])
    return articles_metadata

def get_session_features_config(nar_label_encoders_dict):
    session_features_config = {
        'single_features': {
            #Control features
            'user_id': {'type': 'categorical', 'dtype': 'bytes'},
            'session_id': {'type': 'numerical', 'dtype': 'int'},
            'session_size': {'type': 'numerical', 'dtype': 'int'},
            'session_start': {'type': 'numerical', 'dtype': 'int'},            
        },
        'sequence_features': {
            #Required sequence features
            'event_timestamp': {'type': 'numerical', 'dtype': 'int'},
            'item_clicked': {'type': 'categorical', 'dtype': 'int'},#, 'cardinality': 72933},

            #Location        
            'city': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 1022}, 
            'region': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 237}, 
            'country': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 70}, 
            
            #Device
            'device': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 5}, 
            'os': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 10}, 
            
            #Time
            'local_hour_sin': {'type': 'numerical', 'dtype': 'float'},
            'local_hour_cos': {'type': 'numerical', 'dtype': 'float'},
            'weekday': {'type': 'numerical', 'dtype': 'float'},

            #Referrer type
            'referrer_class': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 7}}}
        }
    }


    feature_groups = {
        'time': ['local_hour_sin', 'local_hour_cos', 'weekday'],
        'device': ['device', 'os'],
        'location': ['country', 'region', 'city'],
        'referrer': ['referrer_class']
    }


    #Disabling optional features when required
    if FLAGS.enabled_clicks_input_features_groups != [ALL_FEATURES]:   
        for feature_group in feature_groups:
            if feature_group not in FLAGS.enabled_clicks_input_features_groups:
                for feature in feature_groups[feature_group]:
                    del session_features_config['sequence_features'][feature]


    #Adding cardinality to categorical features
    for feature_groups_key in session_features_config:
        features_group_config = session_features_config[feature_groups_key]
        for feature_name in features_group_config:
            if feature_name in nar_label_encoders_dict and features_group_config[feature_name]['type'] == 'categorical':
                features_group_config[feature_name]['cardinality'] = len(nar_label_encoders_dict[feature_name])

    tf.logging.info('Session Features: {}'.format(session_features_config))

    return session_features_config
    
def get_internal_enabled_features_config():
    VALID_INTERNAL_FEATURES = ['recency','novelty','article_content_embeddings','item_clicked_embeddings']
    internal_features_config = {}
    enabled_features = []
    if FLAGS.enabled_internal_features == [ALL_FEATURES]:
        enabled_features = set(VALID_INTERNAL_FEATURES)
    else:
        enabled_features = set(FLAGS.enabled_internal_features).intersection(set(VALID_INTERNAL_FEATURES))
    for feature in VALID_INTERNAL_FEATURES:
        internal_features_config[feature] = (feature in enabled_features)
    tf.logging.info('Enabled internal features: {}'.format(enabled_features))
    return internal_features_config


def nar_module_model_fn(features, labels, mode, params):    
    if mode == tf.estimator.ModeKeys.TRAIN:
        negative_samples = params['train_total_negative_samples']
        negative_sample_from_buffer = params['train_negative_samples_from_buffer']
    elif mode == tf.estimator.ModeKeys.EVAL:
        negative_samples = params['eval_total_negative_samples']
        negative_sample_from_buffer = params['eval_negative_samples_from_buffer']

    
    dropout_keep_prob = params['dropout_keep_prob'] if mode == tf.estimator.ModeKeys.TRAIN else 1.0
    
    internal_features_config = get_internal_enabled_features_config()
    
    eval_metrics_top_n = params['eval_metrics_top_n']
    
    model = NARModuleModel(mode, features, labels,
              session_features_config=params['session_features_config'],
              articles_features_config=params['articles_features_config'],
              articles_kg_features_config=params['articles_kg_features_config'],
              batch_size=params['batch_size'], 
              lr=params['lr'],
              keep_prob=dropout_keep_prob,
              negative_samples=negative_samples,
              negative_sample_from_buffer=negative_sample_from_buffer,
              reg_weight_decay=params['reg_weight_decay'], 
              softmax_temperature=params['softmax_temperature'], 
              articles_metadata=params['articles_metadata'],
              content_article_embeddings_matrix=params['content_article_embeddings_matrix'],
              entity_embeddings_matrix=params['entity_embeddings_matrix'],
              context_embeddings_matrix=params['context_embeddings_matrix'],
              kge_vec_dim=params['kge_vec_dim'],
              kge_strategy=params['kge_strategy'],
              recent_clicks_buffer_hours=params['recent_clicks_buffer_hours'],
              recent_clicks_buffer_max_size=params['recent_clicks_buffer_max_size'],
              recent_clicks_for_normalization=params['recent_clicks_for_normalization'],
              CAR_embedding_size=params['CAR_embedding_size'],
              rnn_units=params['rnn_units'],
              metrics_top_n=eval_metrics_top_n,
              plot_histograms=params['save_histograms'],
              novelty_reg_factor=params['novelty_reg_factor'],
              diversity_reg_factor=params['diversity_reg_factor'], 
              internal_features_config=internal_features_config,
              eval_cold_start=params['eval_cold_start'],
              enabled_gcn_refine_article_embedding=params['enabled_gcn_refine_article_embedding'],
              gcn_layer1_size_article_embedding=params['gcn_layer1_size_article_embedding'],
              gcn_layer2_size_article_embedding=params['gcn_layer2_size_article_embedding'],
              enabled_gcn_refine_user_items_contextual=params['enabled_gcn_refine_user_items_contextual'],
              gcn_layer1_size_user_items_contextual=params['gcn_layer1_size_user_items_contextual'],
              gcn_layer2_size_user_items_contextual=params['gcn_layer2_size_user_items_contextual'],
              enabled_gcn_refine_input_items_features=params['enabled_gcn_refine_input_items_features'],
              gcn_layer1_size_input_items_features=params['gcn_layer1_size_input_items_features'],
              gcn_layer2_size_input_items_features=params['gcn_layer2_size_input_items_features'],
              enabled_attention_LSTM=params['enabled_attention_LSTM']
             )
    
    #Using these variables as global so that they persist across different train and eval
    global clicked_items_state, eval_sessions_metrics_log, sessions_negative_items_log

    eval_benchmark_classifiers = []
    if not FLAGS.disable_eval_benchmarks:
        eval_benchmark_classifiers=[
                                    {'recommender': RecentlyPopularRecommender, 'params': {}},
                                    {'recommender': ItemCooccurrenceRecommender, 'params': {}},
                                    {'recommender': ItemKNNRecommender, 
                                          'params': {'reg_lambda': 20,  #Regularization. Discounts the similarity of rare items (incidental co-occurrences). 
                                                     'alpha': 0.5 #Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
                                                     }},
                                    {'recommender': SessionBasedKNNRecommender, 
                                          'params': {'sessions_buffer_size': 3000, #Buffer size of last processed sessions
                                                     'candidate_sessions_sample_size': 2000, #200, #Number of candidate near sessions to sample  
                                                     'sampling_strategy': 'recent', #(recent,random)
                                                     'nearest_neighbor_session_for_scoring': 500, #50 #Nearest neighbors to compute item scores    
                                                     'similarity': 'cosine', #(jaccard, cosine)
                                                     'first_session_clicks_decay': 'div' #Decays weight of first user clicks in active session when finding neighbor sessions (same, div, linear, log, quadradic)
                                                     }},
                                    {'recommender': ContentBasedRecommender, 
                                          'params': {'articles_metadata': params['articles_metadata'],
                                                     'content_article_embeddings_matrix': params['content_article_embeddings_matrix']}},
                                    {'recommender': SequentialRulesRecommender,
                                          'params': {'max_clicks_dist': 10, #Max number of clicks to walk back in the session from the currently viewed item. (Default value: 10) 
                                                     'dist_between_clicks_decay': 'div' #Decay function for distance between two items clicks within a session (linear, same, div, log, qudratic). (Default value: div) 
                                                     }}
                                   ]
                                                              
    hooks = [ItemsStateUpdaterHook(mode, model, 
                                   eval_metrics_top_n=eval_metrics_top_n,
                                   clicked_items_state=clicked_items_state, 
                                   eval_sessions_metrics_log=eval_sessions_metrics_log,
                                   sessions_negative_items_log=sessions_negative_items_log,
                                   sessions_chameleon_recommendations_log=sessions_chameleon_recommendations_log,
                                   content_article_embeddings_matrix=params['content_article_embeddings_matrix'],
                                   entity_embeddings_matrix=params['entity_embeddings_matrix'],
                                   context_embeddings_matrix=params['context_embeddings_matrix'],
                                   articles_metadata=params['articles_metadata'],
                                   eval_negative_sample_relevance=params['eval_negative_sample_relevance'],
                                   eval_benchmark_classifiers=eval_benchmark_classifiers,
                                   eval_metrics_by_session_position=params['eval_metrics_by_session_position'],
                                   eval_cold_start=params['eval_cold_start']                                   
                                   )] 
    
    if mode == tf.estimator.ModeKeys.TRAIN:        
        return tf.estimator.EstimatorSpec(mode, loss=model.total_loss, train_op=model.train,
                                      training_chief_hooks=hooks)

    elif mode == tf.estimator.ModeKeys.EVAL:  

        eval_metrics = {#'hitrate_at_1': (model.next_item_accuracy_at_1, model.next_item_accuracy_at_1_update_op),
                        'hitrate_at_n': (model.recall_at_n, model.recall_at_n_update_op),
                        'mrr_at_n': (model.mrr, model.mrr_update_op),   
                        'ndcg_at_n': (model.ndcg_at_n_mean, model.ndcg_at_n_mean_update_op),                 
                       }
                        
        return tf.estimator.EstimatorSpec(mode, loss=model.total_loss, eval_metric_ops=eval_metrics,
                                      evaluation_hooks=hooks) 


def build_estimator(model_dir,
    content_article_embeddings_matrix, 
    articles_metadata, articles_features_config,
    entity_embeddings_matrix, context_embeddings_matrix, articles_kg_features_config,
    session_features_config):
    """Build an estimator appropriate for the given model type."""

    #Disabling GPU (memory issues on local machine)
    #config_proto = tf.ConfigProto(device_count={'GPU': 0})    
    run_config = tf.estimator.RunConfig(tf_random_seed=RANDOM_SEED,
                                        keep_checkpoint_max=1, 
                                        save_checkpoints_secs=1200, 
                                        save_summary_steps=100,
                                        log_step_count_steps=100,
                                        #session_config=config_proto
                                        )

    estimator = tf.estimator.Estimator(
        config=run_config,
        model_dir=model_dir,
        model_fn=nar_module_model_fn,    
        params={
            'batch_size': FLAGS.batch_size,
            'lr': FLAGS.learning_rate,
            'dropout_keep_prob': FLAGS.dropout_keep_prob,
            'reg_weight_decay': FLAGS.reg_l2,
            'recent_clicks_buffer_hours': FLAGS.recent_clicks_buffer_hours,
            'recent_clicks_buffer_max_size': FLAGS.recent_clicks_buffer_max_size,
            'recent_clicks_for_normalization': FLAGS.recent_clicks_for_normalization,
            'eval_metrics_top_n': FLAGS.eval_metrics_top_n,
            'CAR_embedding_size': FLAGS.CAR_embedding_size,
            'rnn_units': FLAGS.rnn_units,
            'train_total_negative_samples': FLAGS.train_total_negative_samples,
            'train_negative_samples_from_buffer': FLAGS.train_negative_samples_from_buffer,
            'eval_total_negative_samples': FLAGS.eval_total_negative_samples,
            'eval_negative_samples_from_buffer': FLAGS.eval_negative_samples_from_buffer,
            'softmax_temperature': FLAGS.softmax_temperature,
            'save_histograms': FLAGS.save_histograms,
            'eval_metrics_by_session_position': FLAGS.eval_metrics_by_session_position,
            'novelty_reg_factor': FLAGS.novelty_reg_factor,
            'diversity_reg_factor': FLAGS.diversity_reg_factor,
            'eval_negative_sample_relevance': FLAGS.eval_negative_sample_relevance,
            'eval_cold_start': FLAGS.eval_cold_start,

            #For Graph Convolutional Network
            'enabled_gcn_refine_article_embedding': FLAGS.enabled_gcn_refine_article_embedding,
            'gcn_layer1_size_article_embedding': FLAGS.gcn_layer1_size_article_embedding,
            'gcn_layer2_size_article_embedding': FLAGS.gcn_layer2_size_article_embedding,
            #For Graph Convolutional Network (user_items_contextua)
            'enabled_gcn_refine_user_items_contextual': FLAGS.enabled_gcn_refine_user_items_contextual,
            'gcn_layer1_size_user_items_contextual': FLAGS.gcn_layer1_size_user_items_contextual,
            'gcn_layer2_size_user_items_contextual': FLAGS.gcn_layer2_size_user_items_contextual,
            #For Graph Convolutional Network (input_items_features)
            'enabled_gcn_refine_input_items_features': FLAGS.enabled_gcn_refine_input_items_features,
            'gcn_layer1_size_input_items_features': FLAGS.gcn_layer1_size_input_items_features,
            'gcn_layer2_size_input_items_features': FLAGS.gcn_layer2_size_input_items_features,
            #For attention sequence
            'enabled_attention_LSTM': FLAGS.enabled_attention_LSTM,
            #From pre-processing
            'session_features_config': session_features_config,
            'articles_features_config': articles_features_config,
            'articles_metadata': articles_metadata,            
            #From ACR module
            'content_article_embeddings_matrix': content_article_embeddings_matrix,
            #From KG module
            'articles_kg_features_config': articles_kg_features_config,
            'entity_embeddings_matrix': entity_embeddings_matrix,
            'context_embeddings_matrix': context_embeddings_matrix,
            'kge_vec_dim': FLAGS.kge_vec_dim,
            'kge_strategy': FLAGS.kge_strategy
        })

    return estimator


#Saving the negative samples used to evaluate each sessions, so that benchmarks metrics outside the framework (eg. Matrix Factorization) can be comparable
def save_sessions_negative_items(model_output_dir, sessions_negative_items_list, output_file='eval_sessions_negative_samples.json'):
    append_lines_to_text_file(os.path.join(model_output_dir, output_file), 
                                           map(lambda x: json.dumps({'session_id': x['session_id'],
                                                                     'negative_items': x['negative_items']}), 
                                               sessions_negative_items_list))


def save_sessions_chameleon_recommendations_log(model_output_dir, sessions_chameleon_recommendations_log_list, 
                                                eval_hour_id, output_file='eval_chameleon_recommendations_log.json'):
    append_lines_to_text_file(os.path.join(model_output_dir, output_file), 
                                           map(lambda x: json.dumps({'eval_hour_id': eval_hour_id,
                                                                     'session_id': x['session_id'],
                                                                     'next_click_labels': x['next_click_labels'],
                                                                     'predicted_item_ids': x['predicted_item_ids'],
                                                                     'predicted_item_probs': x['predicted_item_probs'],
                                                                     'predicted_item_norm_pop': x['predicted_item_norm_pop']
                                                                     }), 
                                               sessions_chameleon_recommendations_log_list))


#Global vars updated by the Estimator Hook
clicked_items_state = None
eval_sessions_metrics_log = [] 
sessions_negative_items_log = [] if FLAGS.save_eval_sessions_negative_samples else None
sessions_chameleon_recommendations_log = [] if FLAGS.save_eval_sessions_recommendations else None
global_eval_hour_id = 0


def main(unused_argv):
    try:
        # Capture whether it will be a single training job or a hyper parameter tuning job.
        tf_config_env = json.loads(os.environ.get('TF_CONFIG', '{}'))
        task_data = tf_config_env.get('task') or {'type': 'master', 'index': 0}
        trial = task_data.get('trial')

        running_on_mlengine = (len(tf_config_env) > 0)
        print('Running {}'.format('on Google ML Engine' if running_on_mlengine else 'on a server/machine'))

        #Disabling duplicate logs on console when running locally
        logging.getLogger('tensorflow').propagate = running_on_mlengine

        tf.logging.info('Starting training job')    

        gcs_model_output_dir = FLAGS.model_dir
        #If must persist and load model ouput in a local cache (to speedup in ML Engine)
        if FLAGS.use_local_cache_model_dir:
            model_output_dir = tempfile.mkdtemp()
            tf.logging.info('Created local temp folder for models output: {}'.format(model_output_dir))
        else:
            model_output_dir = gcs_model_output_dir

        if trial is not None:
            model_output_dir = os.path.join(model_output_dir, trial)
            gcs_model_output_dir = os.path.join(gcs_model_output_dir, trial)
            tf.logging.info(
                "Hyperparameter Tuning - Trial {} - model_dir = {} - gcs_model_output_dir = {} ".format(trial, model_output_dir, gcs_model_output_dir))

        tf.logging.info('Will save temporary model outputs to {}'.format(model_output_dir))

        #If should warm start training from other previously trained model
        if FLAGS.warmup_model_dir != None:
            tf.logging.info('Copying model outputs from previous job ({}) for warm start'.format(FLAGS.warmup_model_dir))
            dowload_model_output_from_gcs(model_output_dir, 
                                          gcs_model_dir=FLAGS.warmup_model_dir,
                                          files_pattern=['graph.pb', 
                                                         'model.ckpt-', 
                                                         'checkpoint'])

            local_files_after_download_to_debug = list(glob.iglob("{}/**/*".format(model_output_dir), recursive=True))
            tf.logging.info('Files copied from GCS to warm start training: {}'.format(local_files_after_download_to_debug))

        tf.logging.info('Loading ACR module assets')
        acr_label_encoders, articles_metadata_df, content_article_embeddings_matrix = \
                load_acr_module_resources(FLAGS.acr_module_resources_path)
        
        tf.logging.info('Loading word graph module assets')
        word_graph_embedding_dict = deserialize(FLAGS.word_graph_embedding_resources_path)
        #Conver dict to word_graph_embedding_matrix
        keys = sorted(word_graph_embedding_dict.keys())
        result = []
        for key in keys:
            result.append(word_graph_embedding_dict[key])
        word_graph_ebeding = np.vstack(result)
        #Creating and embedding for the padding article    
        embedding_for_padding_article = np.mean(word_graph_ebeding, axis=0)
        word_graph_article_embeddings_with_padding = np.vstack([embedding_for_padding_article, word_graph_ebeding])
        # Make sure the size is the same with original article content embedding matrix
        assert (len(word_graph_article_embeddings_with_padding) == len(content_article_embeddings_matrix))
        tf.logging.info("Read word graph content embeddings: {}".format(word_graph_article_embeddings_with_padding.shape))

        if (FLAGS.enabled_word_graph_article_content_embedding == True):
            tf.logging.info('Enable word_graph_article_content_embedding')
            content_article_embeddings_matrix = word_graph_article_embeddings_with_padding

        #Apply l2-norm by sample
        l2_normalizer_by_sample = Normalizer(norm='l2')
        content_article_embeddings_matrix = l2_normalizer_by_sample.fit_transform(content_article_embeddings_matrix)
        
        #Rescaling content features        
        content_article_embeddings_matrix = content_article_embeddings_matrix * FLAGS.content_embedding_scale_factor

        #feature_config 含有 article id
        articles_features_config = get_articles_features_config(acr_label_encoders)
        #articles_metadata 基本上就是個feature encode 後的內容，其中 article_id 就是 0-73309 可參考 metadata_df
        articles_metadata = process_articles_metadata(articles_metadata_df, articles_features_config)

        tf.logging.info('Loading KG module preprocessing assets')
        articles_kg_features_config = get_articles_kg_features_config(None)
        entity_embeddings_matrix, context_embeddings_matrix = \
            load_kg_module_resources(FLAGS.kg_module_resources_path_entity, FLAGS.kg_module_resources_path_context)
        entity_embeddings_matrix, context_embeddings_matrix = \
            process_kg_module_resources(entity_embeddings_matrix, context_embeddings_matrix)

        #Apply l2-norm by sample
        l2_normalizer_by_sample = Normalizer(norm='l2')
        entity_embeddings_matrix = l2_normalizer_by_sample.fit_transform(entity_embeddings_matrix)
        context_embeddings_matrix = l2_normalizer_by_sample.fit_transform(context_embeddings_matrix)

        entity_embeddings_matrix = entity_embeddings_matrix * FLAGS.content_embedding_scale_factor

        tf.logging.info('Loading NAR module preprocesing assets')
        nar_label_encoders=load_nar_module_preprocessing_resources(FLAGS.nar_module_preprocessing_resources_path) 
        session_features_config = get_session_features_config(nar_label_encoders)

        tf.logging.info('Building NAR model')
        global eval_sessions_metrics_log, clicked_items_state, sessions_negative_items_log, sessions_chameleon_recommendations_log, global_eval_hour_id
        eval_sessions_metrics_log = []
        clicked_items_state = ClickedItemsState(FLAGS.recent_clicks_buffer_hours,
                                                FLAGS.recent_clicks_buffer_max_size, 
                                                FLAGS.recent_clicks_for_normalization, 
                                                content_article_embeddings_matrix.shape[0])
        model = build_estimator(model_output_dir, 
            content_article_embeddings_matrix, articles_metadata, articles_features_config,
            entity_embeddings_matrix, context_embeddings_matrix, articles_kg_features_config,
            session_features_config)
        
        tf.logging.info('Getting training file names')
        train_files = resolve_files(FLAGS.train_set_path_regex)

        if FLAGS.train_files_from > FLAGS.train_files_up_to:
            raise Exception('Final training file cannot be lower than Starting training file')
        train_files = train_files[FLAGS.train_files_from:FLAGS.train_files_up_to+1]

        tf.logging.info('{} files where the network will be trained and evaluated on, from {} to {}' \
                            .format(len(train_files), train_files[0], train_files[-1]))

        start_train = time()
        tf.logging.info("Starting Training Loop")
        
        # 讀取全部 TF*hours.tfrecord[session_1.gz, session_2.gz]
        if FLAGS.enabled_sliding_hours == True:
            training_files_chunks = list(window(train_files, n = FLAGS.training_hours_for_each_eval))
        else:
            training_files_chunks = list(chunks(train_files, FLAGS.training_hours_for_each_eval))

        for chunk_id in range(0, len(training_files_chunks)-1):     

            training_files_chunk = training_files_chunks[chunk_id]
            tf.logging.info('Training files from {} to {}'.format(training_files_chunk[0], training_files_chunk[-1]))
            # print("=>", training_files_chunk.shape)
            # print("=>", session_features_config.shape)
            
            model.train(input_fn=lambda: prepare_dataset_iterator(training_files_chunk, session_features_config, 
                                                                          batch_size=FLAGS.batch_size,
                                                                          truncate_session_length=FLAGS.truncate_session_length))
            
            if chunk_id < len(training_files_chunks)-1:
                if FLAGS.enabled_sliding_hours == True:
                    eval_file = training_files_chunks[chunk_id+1][-1]
                else:
                    #Using the first hour of next training chunck as eval
                    eval_file = training_files_chunks[chunk_id+1][0]
                tf.logging.info('Evaluating file {}'.format(eval_file))
                model.evaluate(input_fn=lambda: prepare_dataset_iterator(eval_file, session_features_config, 
                                                                                 batch_size=FLAGS.batch_size,
                                                                                 truncate_session_length=FLAGS.truncate_session_length))

            #After each number of train/eval loops
            if chunk_id % FLAGS.save_results_each_n_evals == 0:
                tf.logging.info('Saving eval metrics')
                save_eval_benchmark_metrics_csv(eval_sessions_metrics_log, model_output_dir,
                                        training_hours_for_each_eval=FLAGS.training_hours_for_each_eval)

                if FLAGS.save_eval_sessions_negative_samples:
                    #Flushing to disk the negative samples used to evaluate each sessions, 
                    #so that benchmarks metrics outside the framework (eg. Matrix Factorization) can be comparable
                    save_sessions_negative_items(model_output_dir, sessions_negative_items_log)
                    sessions_negative_items_log = []

                if FLAGS.save_eval_sessions_recommendations:  
                    #Flushing to disk the recommended items to test re-ranking approaches (e.g. MMR)
                    save_sessions_chameleon_recommendations_log(model_output_dir, 
                                sessions_chameleon_recommendations_log, global_eval_hour_id)
                    sessions_chameleon_recommendations_log = []                    

                    #Incrementing the eval hour id                    
                    global_eval_hour_id += 1
                    

                #If must persist and load model ouput in a local cache (to speedup in ML Engine)
                if FLAGS.use_local_cache_model_dir:
                    tf.logging.info('Uploading cached results to GCS')
                    upload_model_output_to_gcs(model_output_dir, gcs_model_dir=gcs_model_output_dir,  
                                               #files_pattern=None)
                                               files_pattern=[#'events.out.tfevents.', 
                                               '.csv', '.json'])



        tf.logging.info('Finalized Training')

        save_eval_benchmark_metrics_csv(eval_sessions_metrics_log, model_output_dir,
                                        training_hours_for_each_eval=FLAGS.training_hours_for_each_eval)

        if FLAGS.save_eval_sessions_negative_samples:
            #Flushing to disk the negative samples used to evaluate each sessions, 
            #so that benchmarks metrics outside the framework (eg. Matrix Factorization) can be comparable
            tf.logging.info('Saved save_eval_sessions_negative_samples')
            save_sessions_negative_items(model_output_dir, sessions_negative_items_log)

        if FLAGS.save_eval_sessions_recommendations:             
            #Flushing to disk the recommended items to test re-ranking approaches (e.g. MMR)
            save_sessions_chameleon_recommendations_log(model_output_dir, sessions_chameleon_recommendations_log, global_eval_hour_id)

        tf.logging.info('Saved eval metrics')

        #If must persist and load model ouput in a local cache (to speedup in ML Engine)
        if FLAGS.use_local_cache_model_dir:
            #Uploads all files to GCS
            upload_model_output_to_gcs(model_output_dir, gcs_model_dir=gcs_model_output_dir,
                                        files_pattern=None)
            

        log_elapsed_time(start_train, 'Finalized TRAINING Loop')
    
    except Exception as ex:
        tf.logging.error('ERROR: {}'.format(ex))
        raise



if __name__ == '__main__':  
    tf.app.run()    