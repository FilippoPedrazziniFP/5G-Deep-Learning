import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Get dimensions of input and output
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
count_of_epoch = 100
verbose = 0
encoding_dim = 30

print('batch_size: ', batch_size)
print('dimof_middle: ', dimof_middle)
print('dropout: ', dropout)
print('countof_epoch: ', count_of_epoch)
print('verbose: ', verbose)
print('encoding_dim: ', encoding_dim)

# callbacks to improve the model
checkpointer = ModelCheckpoint(filepath='./models/weights.hdf5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='loss', min_delta=0, patience=1, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir='./models/tensorboard/') # command for the bash: tensorboard --logdir='./models/tensorboard/'


def build_model(dimof_input, dimof_output, dimof_middle, dropout):

    model = Sequential()

    model.add(Dense(dimof_middle, input_dim=dimof_input, kernel_initializer='uniform', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(dimof_middle, kernel_initializer='uniform', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(dimof_output, kernel_initializer='uniform', activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['binary_accuracy'])

    return model

# Learning curve
"""

Before starting the tuning of the model and change it, it is better to see how many samples we need
to perform better. After understanding the number of samples, we can start to design our model.

"""

def learningCurve(X_train, y_train, X_test, y_test, model, earlystopper):

    initial_weights = model.get_weights()
    train_sizes = (len(X_train) * np.linspace(0.1, 0.99, 4)).astype(int) #### 4 different lenghts --> from 10% to 99%

    train_scores = []
    test_scores = []

    for train_size in train_sizes:

        x_train_frac, _, y_train_frac, _ = train_test_split(X_train, y_train, train_size=train_size)
        model.set_weights(initial_weights)

        # fitting the model
        h = model.fit(x_train_frac, y_train_frac, verbose=1, epochs=10, callbacks=[earlystopper])

        # compute the train score
        r = model.evaluate(x_train_frac, y_train_frac, verbose=0)
        train_scores.append(r[-1])

        # compute the test score
        e = model.evaluate(X_test, y_test, verbose=0)
        test_scores.append(e[-1])

        print("Done size: ", train_size)

    plt.plot(train_sizes, train_scores, 'o-', label="Training Score")
    plt.plot(train_sizes, test_scores, 'o-', label="Test Score")
    plt.legend(loc='best')
    plt.show()

# learningCurve(X_train, y_train, X_test, y_test, model, earlystopper)
# in this case the difference between the number of samples is not significant

# model = build_model()
# model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=1, verbose=verbose, callbacks=[earlystopper])
# model = KerasClassifier(build_fn=build_model, epochs=count_of_epoch, verbose=verbose, batch_size=batch_size)
# cv = KFold(10, shuffle=True, random_state=0)
# scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
# print("mean: ", scores.mean())
# print("std: ", scores.std())
# generally speaking now the model performs

# Cross validation
def cross_validation(model, X_train, y_train, splits, epochs, verbose):
    scores = []
    for repeat in range(splits):
        model = build_model()
        h = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, verbose=verbose)
        scores.append(h.history['acc'])
        print(repeat, end=' ')
    scores = np.array(scores)
    mean_acc = scores.mean()
    std_acc = scores.std()
    return mean_acc, std_acc

model = build_model(dimof_input=dimof_input, dimof_output=dimof_output, dropout=0.2, dimof_middle=dimof_middle)

print("Model Summary")
model.summary()

"""
print("Cross Validation score")
mean_acc, std_acc = cross_validation(model=model, X_train=X_train, y_train=y_train, splits=10, epochs=count_of_epoch, verbose=verbose)
print("mean: ", mean_acc)
print("std: ", std_acc)
"""

# The problem with this method of evaluation is that Keras doesnt't have other metrics a part from accuracy
"""print("Hold Out Score")
model.fit(X_train, y_train, validation_split=0.2, epochs=count_of_epoch, verbose=0, batch_size=batch_size)
result = model.evaluate(X_test, y_test, batch_size=batch_size)
print(result)"""

model.fit(X_train, y_train, validation_split=0.2, epochs=count_of_epoch, verbose=0, batch_size=batch_size)
result = model.evaluate(X_test, y_test, batch_size=batch_size)
print("accuarcy: ", result)
y_pred = model.predict(X_test)

# transforming pred and test into an array to get the classification report
y_pred_clas = np.argmax(y_pred, axis=1)
y_test_clas = np.argmax(y_test, axis=1)

# validation
def custom_confusion_matrix(y_true, y_pred, labels=["False", "True"]):

    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ["Predicted " + l for l in labels]
    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
    return df
print("Confusion Matrix")
print(custom_confusion_matrix(y_test_clas, y_pred_clas, ['Attack', 'Normal']))
print("Classification Report")
print(classification_report(y_test_clas, y_pred_clas))