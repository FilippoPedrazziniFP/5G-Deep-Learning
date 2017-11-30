import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout
from sklearn.metrics import confusion_matrix

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
dataset_test = pd.read_csv('./dataset_kdd/test.csv', delimiter=',',header=None, names=col_names, index_col=False)

# preprocessing train set
target = dataset['labels'].copy()
target[target != 'normal'] = 'attack'
X_train = dataset.iloc[:, :-1]

# y_to_binary
le = preprocessing.LabelEncoder()
le.fit(target)
binary_target = le.transform(target)
y_train = binary_target

target = dataset_test['labels'].copy()
target[target != 'normal'] = 'attack'
X_test = dataset_test.iloc[:, :-1]

le = preprocessing.LabelEncoder()
le.fit(target)
binary_target = le.transform(target)
y_test = binary_target

all_data = pd.concat((X_train,X_test))
for column in all_data.select_dtypes(include=[np.object]).columns:
    X_train[column] = X_train[column].astype('category', categories=all_data[column].unique())
    X_test[column] = X_test[column].astype('category', categories=all_data[column].unique())
    print(column, all_data[column].unique())

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train = X_train.astype(float).values
X_test = X_test.astype(float).values

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)

print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

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

# some constants
batch_size = 128
dimof_middle = 100
dropout = 0.05
count_of_epoch = 20
verbose = 0
encoding_dim = 30

print('batch_size: ', batch_size)
print('dimof_middle: ', dimof_middle)
print('dropout: ', dropout)
print('countof_epoch: ', count_of_epoch)
print('verbose: ', verbose)
print('encoding_dim: ', encoding_dim)

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

model = build_model(dimof_input=dimof_input, dimof_output=dimof_output, dropout=0.2, dimof_middle=dimof_middle)

print("Model Summary")
model.summary()

model.fit(X_train, y_train, epochs=count_of_epoch, verbose=0, batch_size=batch_size)
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