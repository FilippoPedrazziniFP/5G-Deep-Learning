import os
import sys
import tensorflow as tf
import numpy as np
from preprocessing import Preprocessing
from model import SoftMaxRegression, SparseAutoEncoder, StackedAutoEncoder
from plots import Plot
import argparse
import progressbar
from itertools import product
from sklearn.model_selection import StratifiedKFold

bar = progressbar.ProgressBar()
parser = argparse.ArgumentParser()

""" General parameters """
parser.add_argument('--model_path', type=str, default='./model/model', help='model checkpoints directory.')
parser.add_argument('--restore', type=bool, default=False, help='if True restore the model from --model_path.')
parser.add_argument('--save_scores', type=bool, default=True, help='if True save scores with parameters in a txt file.')
parser.add_argument('--test', type=bool, default=True, help='if True compute the score on the test set.')
parser.add_argument('--plot', type=bool, default=False, help='if True plots train and test accuracy/loss.')
parser.add_argument('--report', type=bool, default=False, help='if True plots classification report.')
parser.add_argument('--k', type=int, default=5, help='k fold cross validation.')

""" Softmax parameters """
parser.add_argument('--batch_size', type=int, default=100, help='batch size for the training.')
parser.add_argument('--dropout', type=float, default=0.2, help='keep probability of neurons during the training.')
parser.add_argument('--epochs', type=int, default=5, help='number of batch iterations.')
parser.add_argument('--validation', type=int, default=10, help='number of batch iterations.')
parser.add_argument('--weight_decay', type=float, default=0.1, help='scale for l2 regularization.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate.')

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
    sess.run(tf.global_variables_initializer())
    for epoch in range(FLAGS.epochs*FLAGS.batch_size):
        batch_x, batch_y = next_batch(FLAGS.batch_size, X_train, y_train)
        train_dict = {model.x: batch_x, model.y_: batch_y, model.learning_rate: FLAGS.learning_rate, model.dropout: FLAGS.dropout, model.weight_decay: FLAGS.weight_decay, model.is_training: True}
        sess.run(model.optimizer, feed_dict=train_dict)
        if epoch % 100 == 0:
            a, c = sess.run([model.acc, model.loss], feed_dict=train_dict)
            train_a.append(a)
            train_c.append(c)

            test_dict = {model.x: X_test, model.y_: y_test, model.learning_rate: FLAGS.learning_rate, model.dropout: FLAGS.dropout, model.weight_decay: FLAGS.weight_decay, model.is_training: False}
            a, c = sess.run([model.acc, model.loss], feed_dict=test_dict)
            test_a.append(a)
            test_c.append(c)
    y_pred = sess.run([model.predictions], feed_dict=test_dict)
    return (train_a, train_c, test_a, test_c, y_pred)

def classifier(X_train, X_test, y_train, y_test):

    train_a, train_c, test_a, test_c, y_pred = training_classifier(X_train, X_test, y_train, y_test)
    f1_score = Plot.report(y_pred, y_test, FLAGS.report)
    if FLAGS.save_scores == True:
        Plot.saving_scores(FLAGS, test_a[-1], f1_score)
    return f1_score

def train_stacked_autoencoder(X_train, X_test, reg, nois, frac, lr):
    model = StackedAutoEncoder()
    """ Tensorflow needs to see the graph before initilize 
        the variables for the computation """
    model.build_graph()
    config = tf.ConfigProto(intra_op_parallelism_threads=_PROCESSORS, inter_op_parallelism_threads=_PROCESSORS)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for epoch in range(FLAGS.epochs*FLAGS.batch_size):
        batch_x = next_batch(FLAGS.batch_size, X_train)
        train_dict = {model.x: batch_x, model.learning_rate_stacked: lr, model.reg_stacked: reg, model.noise: nois, model.fraction: frac }
        sess.run(model.optimizer, feed_dict=train_dict)
        if epoch % 100 == 0:
            c = sess.run([model.loss], feed_dict=train_dict)
    """ Computing the encoded version of X_train and X_test """
    X_train_dict = {model.x: X_train, model.learning_rate_stacked: lr, model.reg_stacked: reg, model.noise: nois, model.fraction: frac }
    X_test_dict = {model.x: X_test, model.learning_rate_stacked: lr, model.reg_stacked: reg, model.noise: nois, model.fraction: frac }

    X_train = np.asarray(sess.run([model.x_encoded], feed_dict=X_train_dict))
    X_test = np.asarray(sess.run([model.x_encoded], feed_dict=X_test_dict))
    X_train = X_train.reshape(-1, 30)
    X_test = X_test.reshape(-1, 30)
    """ Closing the session to avoid cnflicts with the test """
    return X_train, X_test

def grid_search():
    reg_stacked = [0.1, 0.01, 0.001, 0.5]   
    noise = ["masking", "salt_and_pepper"]
    fraction = [0.1, 0.0, 0.5, 0.8]
    learning_rate = [0.1, 0.001, 0.5, 0.0001]
    best_f1 = 0
    model_dict = {}
    for reg, nois, frac, lr in product(reg_stacked, noise, fraction, learning_rate):
        f1_score = np.average(model(reg, nois, frac, lr))
        if f1_score > best_f1:
            model_dict[f1_score] = [reg, nois, frac, lr]
            best_f1 = f1_score
    print("Best f1: %s with parameters: %s" %(best_f1, model_dict[best_f1]))
    return model_dict[best_f1]

def model(reg, nois, frac, lr):
    k_fold = StratifiedKFold(n_splits=FLAGS.k, random_state=0)
    """ just 20 percent of the dataset to speed up the computation """
    X_train, X_test, y_train, y_test = Preprocessing.train_preprocessing(_TRAIN_PATH)
    avg_f1_score = 0
    y_copy = np.copy(y_train) 
    y_copy_first_column = y_copy[:,0]
    for train, test in k_fold.split(X_train, y_copy_first_column):
        X_train_enc, X_test_enc = train_stacked_autoencoder(X_train[train], X_train[test], reg, nois, frac, lr)
        avg_f1_score = np.average(avg_f1_score + classifier(X_train_enc, X_test_enc, y_train[train], y_train[test]))
        tf.reset_default_graph()
        print(avg_f1_score)
    return avg_f1_score/FLAGS.k

def main(argv):
    best_parameters = grid_search()
    print("Best Parameters found (REG, NOISE, FRACTION, LEARNING RATE): ", best_parameters)
    return

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


    