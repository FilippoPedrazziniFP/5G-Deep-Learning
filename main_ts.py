import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# importing the dataset
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"]

dataset = pd.read_csv('./dataset_kdd/train.csv', delimiter=',',header=None, names=col_names, index_col=False)

# preprocessing
target = dataset['labels'].copy()
target[target != 'normal'] = 'attack'
X = dataset.iloc[:, :-1]
X = pd.get_dummies(X)
X = X.astype(float).values

# y_to_binary
le = preprocessing.LabelEncoder()
le.fit(target)
binary_target = le.transform(target)
y = binary_target


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# get dimensions of input and output
dimof_input = X_train.shape[1]
dimof_output = np.max(y_train) + 1
print('dimof_input: ', dimof_input)
print('dimof_output: ', dimof_output)

# Set y categorical
y_train = np_utils.to_categorical(y_train, dimof_output)
y_test = np_utils.to_categorical(y_test, dimof_output)

print("X shape: ", X_train.shape)
print("y shape: ", y_train.shape)

# some constants
batch_size = 128
dimof_middle = 100
dropout = 0.2
count_of_epoch = 10
verbose = 0
encoding_dim = 20

print('batch_size: ', batch_size)
print('dimof_middle: ', dimof_middle)
print('dropout: ', dropout)
print('countof_epoch: ', count_of_epoch)
print('verbose: ', verbose)
print('encoding_dim: ', encoding_dim)

# placeholders
x = tf.placeholder(np.float32, shape=[None, dimof_input])
y_ = tf.placeholder(np.float32, shape=[None, dimof_output])


# Definition of the variables
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def model(X, dimof_input, dimof_middle, dimof_output):

    W1 = weight_variable([dimof_input, dimof_middle])
    b1 = bias_variable([dimof_middle])
    W2 = weight_variable([dimof_middle, dimof_middle])
    b2 = bias_variable([dimof_middle])
    W3 = weight_variable([dimof_middle, dimof_output])
    b3 = bias_variable([dimof_output])

    # Small epsilon value for the BN transform
    epsilon = 1e-3
    prob = 0.8

    # layer 1
    Z1_BN = tf.nn.sigmoid(tf.matmul(X, W1) + b1)

    # Calculate batch mean and variance
    batch_mean1, batch_var1 = tf.nn.moments(Z1_BN, [0])

    # Apply the initial batch normalizing transform
    Z1_hat = (Z1_BN - batch_mean1) / tf.sqrt(batch_var1 + epsilon)

    # Create two new parameters, scale and beta (shift)
    scale1 = tf.Variable(tf.ones([100]))
    beta1 = tf.Variable(tf.zeros([100]))

    # Scale and shift to obtain the final output of the batch normalization
    # this value is fed into the activation function (here a sigmoid)
    BN1 = scale1 * Z1_hat + beta1

    Z1 = tf.nn.sigmoid(BN1)

    # Dropout
    Z1_dropout = tf.nn.dropout(Z1, prob)

    # layer 1

    Z2_BN = tf.nn.sigmoid(tf.matmul(Z1_dropout, W2) + b2)

    # Calculate batch mean and variance
    batch_mean2, batch_var2 = tf.nn.moments(Z2_BN, [0])

    # Apply the initial batch normalizing transform
    Z2_hat = (Z1_BN - batch_mean1) / tf.sqrt(batch_var1 + epsilon)

    # Create two new parameters, scale and beta (shift)
    scale1 = tf.Variable(tf.ones([100]))
    beta1 = tf.Variable(tf.zeros([100]))

    # Scale and shift to obtain the final output of the batch normalization
    # this value is fed into the activation function (here a sigmoid)
    BN2 = scale1 * Z2_hat + beta1

    Z2 = tf.nn.sigmoid(BN2)

    # Dropout
    Z2_dropout = tf.nn.dropout(Z2, prob)

    # softmax is applied in the training when compuitng the loss
    return tf.matmul(Z2_dropout, W3) + b3

# Output
logits = model(x, dimof_input=dimof_input, dimof_output=dimof_output, dimof_middle=dimof_middle)

# Loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

# Optimizer
train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Prediction
prediction = tf.argmax(logits, 1)

# Correct prediction
correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))

# Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def custom_confusion_matrix(y_true, y_pred, labels=["False", "True"]):
    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ["Predicted " + l for l in labels]
    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
    return df

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

num_samples = X_train.shape[0]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(count_of_epoch):
        total_batch = int(num_samples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = next_batch(batch_size, X_train, y_train)
            sess.run(train_op, feed_dict={x: batch_x, y_: batch_y})
        if epoch % 5 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: X_train, y_: y_train})
            print('step %d, training accuracy %g' % (i, train_accuracy))

    # Validation
    # print('test accuracy %g' % accuracy.eval(feed_dict={x: X_test, y_: y_test}))
    ############
    y_pred = prediction.eval(feed_dict={x: X_test})
    y_test = np.argmax(y_test, axis=1)
    print("Confusion Matrix")
    print(custom_confusion_matrix(y_test, y_pred, ['Attack', 'Normal']))
    print("Classification Report")
    print(classification_report(y_test, y_pred))







