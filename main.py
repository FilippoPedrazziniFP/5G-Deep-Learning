import os
import sys
import tensorflow as tf
import numpy as np
from preprocessing import Preprocessing
from model import SoftMaxRegression, SparseAutoEncoder
from plots import Plot
import argparse
import progressbar

bar = progressbar.ProgressBar()

parser = argparse.ArgumentParser()

""" Network Parameters """
parser.add_argument('--batch_size', type=int, default=100, help='batch size for the training.')
parser.add_argument('--dropout', type=float, default=0.2, help='keep probability of neurons during the training.')
parser.add_argument('--epochs', type=int, default=1, help='number of batch iterations.')
parser.add_argument('--validation', type=int, default=10, help='number of batch iterations.')
parser.add_argument('--weight_decay', type=float, default=0.1, help='scale for l2 regularization.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate.')
parser.add_argument('--model_path', type=str, default='./model/model', help='model checkpoints directory.')
parser.add_argument('--restore', type=bool, default=False, help='if True restore the model from --model_path.')
parser.add_argument('--save_scores', type=bool, default=True, help='if True save scores with parameters in a txt file.')

""" Sparse Autoecnoder Parameters """
parser.add_argument('--alpha', type=float, default=0.00001, help='regularization parameter for sparse autoencoder.')
parser.add_argument('--rho', type=float, default=0.05, help='sparsity value.')
parser.add_argument('--beta', type=float, default=3, help='regularization parameter for sparse autoencoder.')

args = parser.parse_args()

# Model constants
_TRAIN_PATH = './dataset_kdd/train.csv'
_TEST_PATH = './dataset_kdd/test.csv'
_PROCESSORS = 8

def next_batch(num, data, labels=None):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    if labels is not None:
        labels_shuffle = [labels[ i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    else:
        return np.asarray(data_shuffle)

def training(X_train, X_test, y_train, y_test):

    train_a = []
    train_c = []
    test_a = []
    test_c = []

    model = SoftMaxRegression()
    """ Tensorflow needs to see the graph before initilize 
        the variables for the computation """
    model.build_graph()
    config = tf.ConfigProto(intra_op_parallelism_threads=_PROCESSORS, inter_op_parallelism_threads=_PROCESSORS)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    """ To restore the saved model """
    if FLAGS.restore == True:
        saver = tf.train.import_meta_graph(FLAGS.model_path + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        print("Model restored from checkpoint")
    
    sess.run(tf.global_variables_initializer())

    print("Training classifier...")
    for epoch in bar(range(FLAGS.epochs*FLAGS.batch_size)):
        batch_x, batch_y = next_batch(FLAGS.batch_size, X_train, y_train)
        train_dict = {model.x: batch_x, model.y_: batch_y, model.learning_rate: FLAGS.learning_rate, model.dropout: FLAGS.dropout, model.weight_decay: FLAGS.weight_decay, model.is_training: True}
        sess.run(model.optimizer, feed_dict=train_dict)
        if epoch % 10 == 0:
            a, c = sess.run([model.acc, model.loss], feed_dict=train_dict)
            train_a.append(a)
            train_c.append(c)

            test_dict = {model.x: X_test, model.y_: y_test, model.learning_rate: FLAGS.learning_rate, model.dropout: FLAGS.dropout, model.weight_decay: FLAGS.weight_decay, model.is_training: False}
            a, c = sess.run([model.acc, model.loss], feed_dict=test_dict)
            test_a.append(a)
            test_c.append(c)

    """ Save model parameters """
    model_path = saver.save(sess, FLAGS.model_path)
    print("Model saved in: ", model_path)
    """ Computing predictions for further analysis """
    y_pred = sess.run([model.predictions], feed_dict=test_dict)
    return (train_a, train_c, test_a, test_c, y_pred)

def classifier(X_train, X_test, y_train, y_test):

    train_a, train_c, test_a, test_c, y_pred = training(X_train, X_test, y_train, y_test)
    Plot.visualizing_learning(train_a, train_c, test_a, test_c, y_pred, y_test)
    if FLAGS.save_scores == True:
        Plot.saving_scores(FLAGS, test_a[-1])
    return

def autoencoder(X_train, X_test):

    model = SparseAutoEncoder()
    """ Tensorflow needs to see the graph before initilize 
        the variables for the computation """
    model.build_graph()
    config = tf.ConfigProto(intra_op_parallelism_threads=_PROCESSORS, inter_op_parallelism_threads=_PROCESSORS)
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    print("Training Autoencoder...")
    for epoch in range(FLAGS.epochs*FLAGS.batch_size):
        batch_x = next_batch(FLAGS.batch_size, X_train)
        train_dict = {model.x: batch_x, model.learning_rate: FLAGS.learning_rate, model.alpha: FLAGS.alpha, model.beta: FLAGS.beta, model.rho: FLAGS.rho }
        sess.run(model.optimizer, feed_dict=train_dict)
        if epoch % 10 == 0:
            c = sess.run([model.loss], feed_dict=train_dict)
            print("Train Loss: ", c)

    X_train_dict = {model.x: X_train, model.learning_rate: FLAGS.learning_rate, model.alpha: FLAGS.alpha, model.beta: FLAGS.beta, model.rho: FLAGS.rho }
    X_test_dict = {model.x: X_test, model.learning_rate: FLAGS.learning_rate, model.alpha: FLAGS.alpha, model.beta: FLAGS.beta, model.rho: FLAGS.rho }
    """ Computing the encoded version of X_train and X_test """
    X_train = np.asarray(sess.run([model.x_encoded], feed_dict=X_train_dict))
    X_test = np.asarray(sess.run([model.x_encoded], feed_dict=X_test_dict))
    X_train = X_train.reshape(-1, 30)
    X_test = X_test.reshape(-1, 30)
    return X_train, X_test

def main(argv):

    X_train, X_test, y_train, y_test = Preprocessing.train_preprocessing(_TRAIN_PATH)
    
    """ Running the autoencoder """
    X_train, X_test = autoencoder(X_train, X_test)

    """ Running the classifier """
    classifier(X_train, X_test, y_train, y_test)
    return

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

"""
# 1. Declare summaries that you'd like to collect.
tf.scalar_summary("summary_name", tensor, name = "summary_op_name")

# 2. Construct a summary writer object for the computation graph, once all summaries are defined.
summary_writer = tf.train.SummaryWriter(summary_dir_name, sess.graph)

# 3. Group all previously declared summaries for serialization. Usually we want all summaries defined
# in the computation graph. To pick a subset, use tf.merge_summary([summaries]).
summaries_tensor = tf.merge_all_summaries()

# 4. At runtime, in appropriate places, evaluate the summaries_tensor, to assign value.
summary_value, ... = sess.run([summaries_tensor, ...], feed_dict={...})

# 5. Write the summary value to disk, using summary writer.
summary_writer.add_summary(summary_value, global_step=step)

"""

"""
other method without parsing parameters
tf.app.flags.DEFINE_boolean("some_flag", False, "Documentation")

FLAGS = tf.app.flags.FLAGS

def main(_):
  # your code goes here...
  # use FLAGS.some_flag in the code.

if __name__ == '__main__':
    tf.app.run()
"""

    