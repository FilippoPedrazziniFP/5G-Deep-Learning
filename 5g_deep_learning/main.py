import os
import sys
import tensorflow as tf
import numpy as np
from preprocessing import Preprocessing
from model import SoftMaxRegression, SparseAutoEncoder, StackedAutoEncoder
from plots import Plot
import argparse
import progressbar

bar = progressbar.ProgressBar()

parser = argparse.ArgumentParser()

""" General Parameters """
parser.add_argument('--model_path', type=str, default='./../model/model', help='model checkpoints directory.')
parser.add_argument('--restore', type=bool, default=False, help='if True restore the model from --model_path.')
parser.add_argument('--save_scores', type=bool, default=True, help='if True save scores with parameters in a txt file.')
parser.add_argument('--test', type=bool, default=False, help='if True compute the score on the test set.')
parser.add_argument('--plot', type=bool, default=False, help='if True plots train and test accuracy/loss.')
parser.add_argument('--report', type=bool, default=True, help='if True plots classification report.')
parser.add_argument('--autoencoder', type=str, choices= ["stacked" , "sparse", None], default="stacked", help='which autoencoder to use')
parser.add_argument('--log_dir', type=str, default='./../tensorbaord', help='directory where to store tensorbaord values.')

""" Softmax Regression Parameters """
parser.add_argument('--batch_size', type=int, default=100, help='batch size for the training.')
parser.add_argument('--dropout', type=float, default=0.2, help='keep probability of neurons during the training.')
parser.add_argument('--epochs', type=int, default=5, help='number of batch iterations.')
parser.add_argument('--validation', type=int, default=10, help='number of batch iterations.')
parser.add_argument('--weight_decay', type=float, default=0.1, help='scale for l2 regularization.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate.')

""" Sparse Autoencoder Parameters """
parser.add_argument('--reg', type=float, default=0.00001, help='regularization parameter for sparse autoencoder.')
parser.add_argument('--rho', type=float, default=0.05, help='sparsity value.')
parser.add_argument('--beta', type=float, default=3, help='regularization parameter for sparse autoencoder.')
parser.add_argument('--learning_rate_sparse', type=float, default=0.1, help='initial learning rate.')

""" Stacked Denoising Autoencoder Parameters """
"""(REG, NOISE, FRACTION, LEARNING RATE):  [0.5, 'salt_and_pepper', 0.8, 0.0001]"""
parser.add_argument('--noise', type=str, choices= ["masking" , "salt_and_pepper", None], default="salt_and_pepper", help='noising method for corrupting the input.')
parser.add_argument('--fraction', type=float, default=0.1, help='fraction of the input to corrupt.')
parser.add_argument('--learning_rate_stacked', type=float, default=0.01, help='initial learning rate.')
parser.add_argument('--reg_stacked', type=float, default=0.01, help='regularization parameter for stacked autoencoder.')

args = parser.parse_args()

# Model constants
_TRAIN_PATH = './../data/train.csv'
_TEST_PATH = './../data/test.csv'
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

def training_classifier(X_train, X_test, y_train, y_test):

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
        saver.restore(sess, tf.train.latest_checkpoint('./../model/'))
        print("Model restored from checkpoint")
    
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./../' + FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter('./../' + FLAGS.log_dir + '/test', sess.graph)

    print("Training classifier...")
    for epoch in range(FLAGS.epochs*FLAGS.batch_size):
        batch_x, batch_y = next_batch(FLAGS.batch_size, X_train, y_train)
        train_dict = {model.x: batch_x, model.y_: batch_y, model.learning_rate: FLAGS.learning_rate, model.dropout: FLAGS.dropout, model.weight_decay: FLAGS.weight_decay, model.is_training: True}
        sess.run(model.optimizer, feed_dict=train_dict)
        if epoch % 10 == 0:
            summary_train, a, c = sess.run([model.summaries_tensor, model.acc, model.loss], feed_dict=train_dict)
            train_writer.add_summary(summary_train, global_step=epoch)
            train_a.append(a)
            train_c.append(c)
            print("Train Accuracy: ", a)

            test_dict = {model.x: X_test, model.y_: y_test, model.learning_rate: FLAGS.learning_rate, model.dropout: FLAGS.dropout, model.weight_decay: FLAGS.weight_decay, model.is_training: False}
            summary_test, a, c = sess.run([model.summaries_tensor, model.acc, model.loss], feed_dict=test_dict)
            test_writer.add_summary(summary_test, global_step=epoch)
            test_a.append(a)
            test_c.append(c)
            print("Test Accuracy: ", a)

    """ Save model parameters """
    model_path = saver.save(sess, FLAGS.model_path)
    print("Model saved in: ", model_path)
    """ Computing predictions for further analysis """
    y_pred = sess.run([model.predictions], feed_dict=test_dict)
    return (train_a, train_c, test_a, test_c, y_pred)

def classifier(X_train, X_test, y_train, y_test):

    train_a, train_c, test_a, test_c, y_pred = training_classifier(X_train, X_test, y_train, y_test)
    Plot.visualizing_learning(train_a, train_c, test_a, test_c, FLAGS.plot)
    f1_score = Plot.report(y_pred, y_test, FLAGS.report)
    if FLAGS.save_scores == True:
        Plot.saving_scores(FLAGS, test_a[-1], f1_score)
    return

def train_stacked_autoencoder(X_train, X_test):
    model = StackedAutoEncoder()
    """ Tensorflow needs to see the graph before initilize 
        the variables for the computation """
    model.build_graph()
    config = tf.ConfigProto(intra_op_parallelism_threads=_PROCESSORS, inter_op_parallelism_threads=_PROCESSORS)
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    print("Training Stacked Autoencoder...")
    for epoch in range(FLAGS.epochs*FLAGS.batch_size):
        batch_x = next_batch(FLAGS.batch_size, X_train)
        train_dict = {model.x: batch_x, model.learning_rate_stacked: FLAGS.learning_rate_stacked, model.reg_stacked: FLAGS.reg_stacked, model.noise: FLAGS.noise, model.fraction: FLAGS.fraction}
        sess.run(model.optimizer, feed_dict=train_dict)
        if epoch % 10 == 0:
            c = sess.run([model.loss], feed_dict=train_dict)
            print("Train Loss: ", c)
    """ Computing the encoded version of X_train and X_test """
    X_train_dict = {model.x: X_train, model.learning_rate_stacked: FLAGS.learning_rate_stacked, model.reg_stacked: FLAGS.reg_stacked, model.noise: FLAGS.noise, model.fraction: FLAGS.fraction}
    X_test_dict = {model.x: X_test, model.learning_rate_stacked: FLAGS.learning_rate_stacked, model.reg_stacked: FLAGS.reg_stacked, model.noise: FLAGS.noise, model.fraction: FLAGS.fraction}

    X_train = np.asarray(sess.run([model.x_encoded], feed_dict=X_train_dict))
    X_test = np.asarray(sess.run([model.x_encoded], feed_dict=X_test_dict))
    X_train = X_train.reshape(-1, 30)
    X_test = X_test.reshape(-1, 30)
    """ Closing the session to avoid cnflicts with the test """
    return X_train, X_test

def train_sparse_autoencoder(X_train, X_test):
    model = SparseAutoEncoder()
    """ Tensorflow needs to see the graph before initilize 
        the variables for the computation """
    model.build_graph()
    config = tf.ConfigProto(intra_op_parallelism_threads=_PROCESSORS, inter_op_parallelism_threads=_PROCESSORS)
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    print("Training Sparse Autoencoder...")
    for epoch in range(FLAGS.epochs*FLAGS.batch_size):
        batch_x = next_batch(FLAGS.batch_size, X_train)
        train_dict = {model.x: batch_x, model.learning_rate_sparse: FLAGS.learning_rate_sparse, model.reg: FLAGS.reg, model.beta: FLAGS.beta, model.rho: FLAGS.rho }
        sess.run(model.optimizer, feed_dict=train_dict)
        if epoch % 10 == 0:
            c = sess.run([model.loss], feed_dict=train_dict)
            print("Train Loss: ", c)
    
    """ Computing the encoded version of X_train and X_test """    
    X_train_dict = {model.x: X_train, model.learning_rate_sparse: FLAGS.learning_rate_sparse, model.reg: FLAGS.reg, model.beta: FLAGS.beta, model.rho: FLAGS.rho }
    X_test_dict = {model.x: X_test, model.learning_rate_sparse: FLAGS.learning_rate_sparse, model.reg: FLAGS.reg, model.beta: FLAGS.beta, model.rho: FLAGS.rho }

    X_train = np.asarray(sess.run([model.x_encoded], feed_dict=X_train_dict))
    X_test = np.asarray(sess.run([model.x_encoded], feed_dict=X_test_dict))
    X_train = X_train.reshape(-1, 30)
    X_test = X_test.reshape(-1, 30)
    """ Closing the session to avoid cnflicts with the test """
    return X_train, X_test

def test():
    X_train, X_test, y_train, y_test = Preprocessing.test_preprocessing(_TRAIN_PATH, _TEST_PATH)

    """ Running the Autoencoder """
    if FLAGS.autoencoder == "stacked":
        X_train, X_test = train_stacked_autoencoder(X_train, X_test)
    elif FLAGS.autoencoder == "sparse":
        X_train, X_test = train_sparse_autoencoder(X_train, X_test)
    """ Running the classifier """
    classifier(X_train, X_test, y_train, y_test)
    tf.reset_default_graph()

def main(argv):  
    if FLAGS.test == True:
        test()
    else:
        X_train, X_test, y_train, y_test = Preprocessing.train_preprocessing(_TRAIN_PATH)
        """ Running the Autoencoder """
        if FLAGS.autoencoder == "stacked":
            X_train, X_test = train_stacked_autoencoder(X_train, X_test)
        elif FLAGS.autoencoder == "sparse":
            X_train, X_test = train_sparse_autoencoder(X_train, X_test)
        """ Running the classifier """

        """ Running the classifier """
        classifier(X_train, X_test, y_train, y_test)
        tf.reset_default_graph()
    return

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    