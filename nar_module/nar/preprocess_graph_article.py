import logging
import tensorflow as tf
import numpy as np
import pickle
import scipy.sparse as sp
from .utils import log_elapsed_time
from time import time
from tqdm import tqdm
import gc
import msgpack

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

tf.flags.DEFINE_string('input_gnn_resources_path',
                    default='/graph', help='graph resources path')
tf.flags.DEFINE_string('output_graph_resources_path',
                    default='/graph', help='graph resources path')

FLAGS = tf.flags.FLAGS

def deserialize(filename):
    with tf.io.gfile.GFile(filename, 'rb') as handle:
        tf.compat.v1.logging.info('Deserialize {}'.format(filename))
        return pickle.load(handle)

def deserialize_in_msgpack(filename):
    # Read msgpack file
    with tf.io.gfile.GFile(filename, 'rb') as handle:
        tf.compat.v1.logging.info('Deserialize {}'.format(filename))
        return msgpack.unpack(handle, raw=False)

def main(unused_argv):
    try:
        start_time = time()
        # dict_path = "/home/hengshiou/Documents/chameleon_recsys/adressa" + "/graph/dict_node_neighbor.pickle"
        dict_path = FLAGS.input_gnn_resources_path
       
        # 讀取 dict_node_neighbor
        gc.disable()
        dict_node_neighbor = deserialize_in_msgpack(dict_path)
        gc.enable()

        # convert dcit -> sparse matrix
        mat = sp.dok_matrix((73309,73309), dtype=np.int8)
        for article_id, neighbor_ids in tqdm(dict_node_neighbor.items() ,desc="dcit -> sparse matrix"):
            mat[article_id, neighbor_ids] = 1
        mat = mat.transpose().tocsr()

        # write sparse matrix
        s_m_path = "/home/hengshiou/Documents/chameleon_recsys/adressa" + "/graph/sparse_matrix.npz"
        sp.save_npz(s_m_path, mat)

        log_elapsed_time(start_time, 'Finalized save sparse_matrix.npz')

    except Exception as ex:
        tf.compat.v1.logging.error('ERROR: {}'.format(ex))
        raise

if __name__ == '__main__':  
    tf.app.run()    