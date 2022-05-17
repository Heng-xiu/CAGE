
# /home/hengshiou/Documents/chameleon_recsys/adressa
# ${DATA_DIR} = "[REPLACE WITH THE PATH TO acr_articles_metadata_embeddings.pickle FOLDER]"
DATA_DIR=/home/hengshiou/Documents/chameleon_recsys/adressa && \
python3 -m nar.preprocess_gnn_article \
--acr_module_resources_path ${DATA_DIR}/data_transformed/pickles/acr_articles_metadata_embeddings.pickle \
--output_dict_resources_path ${DATA_DIR}/graph/dict_node_neighbor.msgpack \
--output_sparse_matrix_resources_path ${DATA_DIR}/graph/sparse_matrix_pool.npz \
--similarity_threshold 0.5 
