
# /home/hengshiou/Documents/chameleon_recsys/adressa
# ${DATA_DIR} = "[REPLACE WITH THE PATH TO acr_articles_metadata_embeddings.pickle FOLDER]"
DATA_DIR="/home/hengshiou/Documents/chameleon_recsys/adressa" && \
python3 -m nar.preprocess_graph_article \
--input_gnn_resources_path ${DATA_DIR}/graph/dict_node_neighbor.pickle \
--output_gnn_resources_path ${DATA_DIR}/graph/sparse_matrix.npz
