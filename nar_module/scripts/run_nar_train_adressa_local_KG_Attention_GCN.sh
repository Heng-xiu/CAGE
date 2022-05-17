#!/bin/bash

DATA_DIR=/home/hengshiou/Documents/chameleon_recsys/adressa && \
JOB_PREFIX=adressa && \
JOB_ID=`whoami`_${JOB_PREFIX}_`date '+%Y_%m_%d_%H%M%S'` && \
MODEL_DIR='/tmp/chameleon/jobs/'${JOB_ID} && \
echo 'Running training job and outputing to '${MODEL_DIR} && \
python3 -m nar.nar_trainer_adressa \
	--model_dir ${MODEL_DIR} \
	--train_set_path_regex "${DATA_DIR}/data_transformed/sessions_tfrecords_by_hour/adressa_sessions_*.tfrecord.gz" \
	--train_files_from 0 \
	--train_files_up_to 383 \
	--training_hours_for_each_eval 5 \
	--save_results_each_n_evals 1 \
	--acr_module_resources_path ${DATA_DIR}/data_transformed/pickles/acr_articles_metadata_embeddings.pickle \
	--kg_module_resources_path_entity ${DATA_DIR}/data_transformed/kge/article_encoded_id_entity_embedding.vec \
	--kg_module_resources_path_context ${DATA_DIR}/data_transformed/kge/article_encoded_id_context_embedding.vec \
	--word_graph_embedding_resources_path ${DATA_DIR}/data_transformed/pickles/word_graph_x_a_10_100_300.vec \
	--nar_module_preprocessing_resources_path ${DATA_DIR}/data_transformed/pickles/nar_preprocessing_resources.pickle \
	--batch_size 64 \
	--truncate_session_length 20 \
	--learning_rate 3e-4 \
	--dropout_keep_prob 1.0 \
	--reg_l2 1e-4 \
	--softmax_temperature 0.2 \
	--recent_clicks_buffer_hours 1.0 \
	--recent_clicks_buffer_max_size 30000 \
	--recent_clicks_for_normalization 5000 \
	--eval_metrics_top_n 10 \
	--CAR_embedding_size 1024 \
	--rnn_units 255 \
	--rnn_num_layers 2 \
	--train_total_negative_samples 50 \
	--train_negative_samples_from_buffer 5000 \
	--eval_total_negative_samples 50 \
	--eval_negative_samples_from_buffer 5000 \
	--eval_negative_sample_relevance 0.02 \
	--content_embedding_scale_factor 2.0 \
	--enabled_articles_input_features_groups "category,author" \
	--enabled_clicks_input_features_groups "time,device,location,referrer" \
	--enabled_internal_features "recency,novelty,article_content_embeddings,item_clicked_embeddings" \
	--novelty_reg_factor 0.0 \
	--disable_eval_benchmarks \
	--enabled_articles_kg_features "entity,context" \
	--kge_strategy "Append" \
	--enabled_attention_LSTM \
	-- \
	--gcn_layer1_size_input_items_features 250 \
	--gcn_layer2_size_input_items_features 125 \
	-- \
	--enabled_sliding_hours \
	-- \
	--enabled_gcn_refine_input_items_features \
	--enabled_word_graph_article_content_embedding

#--{Append, Add, Concat}
#--rnn_units 255 \
#--save_histograms
#--save_eval_sessions_negative_samples \
#--save_eval_sessions_recommendations \
#--disable_eval_benchmarks
#--eval_cold_start

