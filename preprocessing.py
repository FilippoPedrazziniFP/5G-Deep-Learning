import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler

_CLASSES = 2

class Preprocessing(object):
	"""docstring for Preprocessing"""
	def __init__(self, arg):
		super(Preprocessing, self).__init__()
		self.arg = arg
	
	@staticmethod
	def importing_the_dataset(data_path):
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

		df = pd.read_csv(data_path, delimiter=',',header=None, names=col_names, index_col=False)
		
		return df

	@staticmethod
	def hold_out_split(X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
		return X_train, X_test, y_train, y_test

	@staticmethod
	def getting_x_y(df):

		target = df['labels'].copy()
		target[target != 'normal'] = 'attack'
		X = df.iloc[:, :-1]
		X = pd.get_dummies(X)
		X = X.astype(float).values
		le = preprocessing.LabelEncoder()
		le.fit(target)
		binary_target = le.transform(target)
		y = binary_target
		
		return X, y

	@staticmethod
	def feature_scaling(X_train, X_test):
		
		sc = StandardScaler()
		sc.fit(X_train)
		X_train = sc.transform(X_train)
		X_test = sc.transform(X_test)
		
		return X_train, X_test

	@staticmethod
	def y_to_categorical(y, classes):
		return np_utils.to_categorical(y, classes)

	@staticmethod
	def train_preprocessing(train_path):
		
		df = Preprocessing.importing_the_dataset(train_path)
		X, y = Preprocessing.getting_x_y(df)
		X_train, X_test, y_train, y_test = Preprocessing.hold_out_split(X, y)
		X_train, X_test = Preprocessing.feature_scaling(X_train, X_test)
		y_train = Preprocessing.y_to_categorical(y_train, _CLASSES)
		y_test = Preprocessing.y_to_categorical(y_test, _CLASSES)

		return X_train, X_test, y_train, y_test

	@staticmethod
	def test_preprocessing(train_path, test_path):

		df_train = Preprocessing.importing_the_dataset(train_path)
		df_test = Preprocessing.importing_the_dataset(test_path)
		
		""" X_train, y_train split """
		target = df_train['labels'].copy()
		target[target != 'normal'] = 'attack'
		X_train = df_train.iloc[:, :-1]
		le = preprocessing.LabelEncoder()
		le.fit(target)
		binary_target = le.transform(target)
		y_train = binary_target

		""" X_test, y_test split """
		target = df_test['labels'].copy()
		target[target != 'normal'] = 'attack'
		X_test = df_test.iloc[:, :-1]
		le = preprocessing.LabelEncoder()
		le.fit(target)
		binary_target = le.transform(target)
		y_test = binary_target

		""" Test and Train have different cat variables. The simple OHE doensn't work """
		all_data = pd.concat((X_train,X_test))
		for column in all_data.select_dtypes(include=[np.object]).columns:
		    X_train[column] = X_train[column].astype('category', categories=all_data[column].unique())
		    X_test[column] = X_test[column].astype('category', categories=all_data[column].unique())

		X_train = pd.get_dummies(X_train)
		X_test = pd.get_dummies(X_test)

		X_train = X_train.astype(float).values
		X_test = X_test.astype(float).values

		X_train, X_test = Preprocessing.feature_scaling(X_train, X_test)
		y_train = Preprocessing.y_to_categorical(y_train, _CLASSES)
		y_test = Preprocessing.y_to_categorical(y_test, _CLASSES)

		return X_train, X_test, y_train, y_test


		




