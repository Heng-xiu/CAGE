
import logging
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy import sparse
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse import vstack, kron
import pickle

## log
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

tf.flags.DEFINE_string('acr_module_resources_path',
                    default='/pickles', help='ACR module resources path')
tf.flags.DEFINE_string('graph_adj_matrix_path',
                default='/pickles', help='ACR module resources path')
tf.flags.DEFINE_string('output_content_embedding_path',
                    default='/pickles', help='graph resources path')

FLAGS = tf.flags.FLAGS

def serialize(filename, obj):
    #with open(filename, 'wb') as handle:
    # with tf.gfile.Open(filename, 'wb') as handle:
    with tf.io.gfile.Open(filename, 'wb') as handle:
        pickle.dump(obj, handle)#, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize(filename):
    #with open(filename, 'rb') as handle:
    # with tf.gfile.Open(filename, 'rb') as handle: # ml warning 不行
    with tf.io.gfile.GFile (filename, 'rb') as handle:
        # 一直出錯試試看
        return pickle.load(handle)

def deserialize_npz(filename):
    #with open(filename, 'rb') as handle:
    # with tf.gfile.Open(filename, 'rb') as handle: # ml warning 不行
    with tf.io.gfile.GFile (filename, 'rb') as handle:
        # 一直出錯試試看
        return sp.load_npz(handle)

def load_acr_module_resources(acr_module_resources_path):
    (acr_label_encoders, articles_metadata_df, content_article_embeddings) = \
              deserialize(acr_module_resources_path)

    tf.logging.info("Read ACR label encoders for: {}".format(acr_label_encoders.keys()))
    tf.logging.info("Read ACR articles metadata: {}".format(len(articles_metadata_df)))
    tf.logging.info("Read ACR article content embeddings: {}".format(content_article_embeddings.shape))

    return acr_label_encoders, articles_metadata_df, content_article_embeddings

def matmul(x, y, sparse=False):
    """Wrapper for sparse matrix multiplication."""
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    return tf.matmul(x, y)


class GraphConvLayer:
    def __init__(
            self,
            input_dim,
            output_dim,
            activation=None,
            use_bias=False,
            name="graph_conv"):
        """Initialise a Graph Convolution layer.
        Args:
            input_dim (int): The input dimensionality.
            output_dim (int): The output dimensionality, i.e. the number of
                units.
            activation (callable): The activation function to use. Defaults to
                no activation function.
            use_bias (bool): Whether to use bias or not. Defaults to `False`.
            name (str): The name of the layer. Defaults to `graph_conv`.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.name = name

        with tf.variable_scope(self.name):
            self.w = tf.get_variable(
                name='w',
                shape=(self.input_dim, self.output_dim),
                initializer=tf.initializers.glorot_uniform())

            if self.use_bias:
                self.b = tf.get_variable(
                    name='b',
                    initializer=tf.constant(0.1, shape=(self.output_dim,)))

    def call(self, adj_norm, x, sparse=False):
        x = matmul(x=x, y=self.w, sparse=sparse)  # XW
        x = matmul(x=adj_norm, y=x, sparse=True)  # AXW

        if self.use_bias:
            x = tf.add(x, self.use_bias)          # AXW + B

        if self.activation is not None:
            x = self.activation(x)                # activation(AXW + B)

        return x

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    # The zeroth element of the tuple contains the cell location of each
    # non-zero value in the sparse matrix
    # The first element of the tuple contains the value at each cell location
    # in the sparse matrix
    # The second element of the tuple contains the full shape of the sparse
    # matrix
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def main(unused_argv):
    try:
        ## Read input
        
        # path = "/home/hengshiou/Documents/chameleon_recsys/adressa" + "/graph/sparse_matrix_threadspool.npz"
        path = FLAGS.graph_adj_matrix_path
        # with sp.load_npz(path) as X:
        #     sparse_matrix = X
        tf.logging.info("Read load_npz for: {}".format(path))
        sparse_matrix = deserialize_npz(path)
        tf.logging.info("Loaded sparse_matrix for: {}".format(sparse_matrix.shape))
        # acr_module_resources_path = "/home/hengshiou/Documents/chameleon_recsys/adressa" + "/data_transformed/pickles/acr_articles_metadata_embeddings.pickle"
        acr_module_resources_path = FLAGS.acr_module_resources_path
        acr_labels_encoder, articles_metadata_df, content_article_embeddings_matrix = load_acr_module_resources(acr_module_resources_path)

        adj = vstack(sparse_matrix[:], format='csr')
        # e.g. features = sp.vstack((allx, tx)).tolil()
        feat_x = content_article_embeddings_matrix

        # TensorFlow placeholders
        ph = {
            'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_mat"),
            'x': tf.sparse_placeholder(tf.float32, name="x")
        }

        # out_dims_in_each_layer
        l_sizes = [200, 100, 50]

        o_fc1 = GraphConvLayer(
            input_dim=feat_x.shape[-1],
            output_dim=l_sizes[0],
            name='fc1',
            activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=ph['x'], sparse=True)

        o_fc2 = GraphConvLayer(
            input_dim=l_sizes[0],
            output_dim=l_sizes[1],
            name='fc2',
            activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc1)

        o_fc3 = GraphConvLayer(
            input_dim=l_sizes[1],
            output_dim=l_sizes[2],
            name='fc3',
            activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc2)

        config = tf.ConfigProto(
        #     device_count = {'GPU': 0}
        )
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        adj_tilde = adj + np.identity(n=adj.shape[0])
        d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
        d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2)
        d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
        adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
        adj_norm_tuple = sparse_to_tuple(sp.coo_matrix(adj_norm))
        feat_x_tuple = sparse_to_tuple(sp.coo_matrix((feat_x)))

        feed_dict = {ph['adj_norm']: adj_norm_tuple,
                    ph['x']: feat_x_tuple
                    }

        outputs = sess.run(o_fc3, feed_dict=feed_dict)
        serialize(FLAGS.output_content_embedding_path, outputs)

    except Exception as ex:
        tf.logging.error('ERROR: {}'.format(ex))
        raise

if __name__ == '__main__':  
    tf.app.run()   