
import logging
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from .utils import log_elapsed_time
from time import time
from .nar_utils import load_acr_module_resources
from tqdm import tqdm
import msgpack
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from multiprocessing import Pool
import random

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

tf.flags.DEFINE_string('acr_module_resources_path',
                    default='/pickles', help='ACR module resources path')
tf.flags.DEFINE_string('output_dict_resources_path',
                    default='/graph', help='graph resources path')
tf.flags.DEFINE_string('output_sparse_matrix_resources_path',
                    default='/graph', help='graph resources path')
tf.flags.DEFINE_float('similarity_threshold', default=0.5, help='Similarity threshold between two article embedding')

FLAGS = tf.flags.FLAGS

def cos_dist(vec1,vec2):
    """
    :param vec1:
    :param vec2:
    :return: the similarity between two vectors
    """
    dist1 = float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return dist1

def serialize_in_msgpack(filename, obj):
    #https://stackoverflow.com/questions/54161523/msgpack-gets-string-data-back-as-binary
    with tf.gfile.Open(filename, 'wb') as handle:
        tf.compat.v1.logging.info('Serialize {}'.format(filename))
        msgpack.pack(obj, handle)

def deserialize_in_msgpack(filename):
    # Read msgpack file
    with tf.gfile.Open(filename, 'rb') as handle:
        tf.compat.v1.logging.info('Deserialize {}'.format(filename))
        return msgpack.unpack(handle, raw=False)

def get_dict_node_neighbor(content_article_embeddings_matrix):

    start_find_similarity = time()

    if os.path.isfile(FLAGS.output_dict_resources_path) is True:
        return deserialize_in_msgpack(FLAGS.output_dict_resources_path)
    ## 基本上這裡都要重寫
    iterable_matrix = list(enumerate(content_article_embeddings_matrix))
    dict_node_neighbor = {}
    for idx, embedding in tqdm(iterable_matrix[1:]):
        #init value
        node_idx = idx
        node_neighbor_list = []
        #compare other article embedding
        for next_idx, next_embedding in iterable_matrix[idx+1:]:
            if cos_dist(embedding, next_embedding) >= FLAGS.similarity_threshold:
                node_neighbor_list.append(next_idx)
        
        dict_node_neighbor[node_idx] = node_neighbor_list
    serialize_in_msgpack(filename=FLAGS.output_dict_resources_path, obj=dict_node_neighbor)
    log_elapsed_time(start_find_similarity, 'Finalized find_similarity Loop')

    return dict_node_neighbor

def convert_dict_to_sparse_matrix(desired_convert_dict):

    start_converting = time()

    # convert dcit -> sparse matrix
    mat = sp.dok_matrix((73309,73309), dtype=np.int8) # 73309 means number of articles
    for article_id, neighbor_ids in tqdm(desired_convert_dict.items() ,desc="dict -> sparse matrix"):
        mat[article_id, neighbor_ids] = 1
    mat = mat.transpose().tocsr()

    # write sparse matrix
    # s_m_path = "/home/hengshiou/Documents/chameleon_recsys/adressa" + "/graph/sparse_matrix.npz"
    s_m_path = FLAGS.output_sparse_matrix_resources_path
    sp.save_npz(s_m_path, mat)

    log_elapsed_time(start_converting, 'Finalized save sparse_matrix.npz')

def get_adj_matrix(res_matrix):
    """
    memory issue
    """
    start_converting = time()

    # 取得 similarity_matrix, res_matrix[0] 為 padding 我們不要
    # v1
    # similarity_matrix = cosine_similarity(res_matrix[1:], dense_output=True)
    # v2
    similarity_matrix = 1-pairwise_distances(res_matrix[1:], metric='cosine')

    # 去除 self-loop
    np.fill_diagonal(similarity_matrix, 0)

    # 去除小於 threshold, 將留下的轉成 1,0 等易於 condition ? 1 : 0
    pass_threshold_matrix = np.where(similarity_matrix >= FLAGS.similarity_threshold, 1, 0)

    # 轉化成 csr_matrix
    mat = sp.csr_matrix(pass_threshold_matrix)

    #write sparse_matrix
    s_m_path = FLAGS.output_sparse_matrix_resources_path
    sp.save_npz(s_m_path, mat)

    log_elapsed_time(start_converting, 'Finalized save sparse_matrix.npz')
    return mat

def wrtie_csr_matrix_to(mat):
    start_converting = time()
    # s_m_path = "/home/hengshiou/Documents/chameleon_recsys/adressa" + "/graph/sparse_matrix.npz"
    s_m_path = FLAGS.output_sparse_matrix_resources_path
    sp.save_npz(s_m_path, mat)
    log_elapsed_time(start_converting, 'Finalized wrtie_csr_matrix_to')

def coculate_similarity(idx, tmp):
    node_neighbor_list = []
    arch_tuple = tmp[idx]
    arch_idx = arch_tuple[0]
    arch_val = arch_tuple[1]
    
    for next_idx, next_val in tmp[idx+1:]:
        if cos_dist(arch_val, next_val) >= FLAGS.similarity_threshold:
            node_neighbor_list.append(next_idx-1)
    return {arch_idx-1: node_neighbor_list}

def multiprocess_threads(rows, cols, vec_list):
    # rows = res_matrix[1:].shape[0]
    # cols = res_matrix[1:].shape[0]
    mat = sp.lil_matrix((rows, rows), dtype=np.int8) # 73309 means number of articles
    dict_node_neighbor = dict()
    pool = Pool()
    results = []
    vec_list_keys = [idx for idx, _ in vec_list[:]]
    #嘗試減少數量
    vec_list_keys = sorted(random.sample(vec_list_keys, 700))
    #assign task
    for i in tqdm(vec_list_keys[1:], desc="coculate_similarity"):
        results.append(pool.apply_async(coculate_similarity, args=(i, vec_list[:])))
    
    #all thread finished
    tf.logging.info('Wait for all tasks finished')
    pool.close()
    tf.logging.info('Pool closed!')
    # pool.join()
    tf.logging.info('all tasks finished!')
    #assign result
    for result in tqdm(results, desc="assign result -> dict"):
        dict_node_neighbor.update(result.get())
    #convert_to_lil_matrix
    for user_id, movie_ids in tqdm(dict_node_neighbor.items(), desc="dict -> sparse matrix"):
        mat[user_id, movie_ids] = 1
    # remove 1st rows and 1st columns because of padding
    
    return mat.tocsr()

def main(unused_argv):
    try:
        _, _, content_article_embeddings_matrix = load_acr_module_resources(FLAGS.acr_module_resources_path)
        print(content_article_embeddings_matrix.shape)
        row = content_article_embeddings_matrix[1:].shape[0]
        col = content_article_embeddings_matrix[1:].shape[1]

        # v2 ((stable))
        # dictnodes = get_dict_node_neighbor(content_article_embeddings_matrix)
        # convert_dict_to_sparse_matrix(dictnodes)

        # v6
        vec_list = list(enumerate(content_article_embeddings_matrix))

        csr_mat = multiprocess_threads(row, col, vec_list)

        wrtie_csr_matrix_to(csr_mat)

    except Exception as ex:
        tf.compat.v1.logging.error('ERROR: {}'.format(ex))
        raise

if __name__ == '__main__':  
    tf.app.run()    