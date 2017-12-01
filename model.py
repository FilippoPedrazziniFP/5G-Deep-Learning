import tensorflow as tf
import numpy as np
import functools
import math

_CLASSES = 2
_INPUT_DIMENSION = 122
_BN_EPSILON = 0.001
_ENCODING_DIMESION = 30
_K_LEARNING_DECAY = 0.5
_DECAY_RATE_LD = 0.96

class SoftMaxRegression(object):
	"""docstring for DeepModel"""
	def __init__(self):
		super(SoftMaxRegression, self).__init__()
		self.placeholders()

	def placeholders(self):
		self.x = tf.placeholder(np.float32, shape=[None, _ENCODING_DIMESION])
		self.y_ = tf.placeholder(np.float32, shape=[None, _CLASSES])
		self.learning_rate = tf.placeholder(tf.float32)
		self.dropout = tf.placeholder(tf.float32)
		self.weight_decay = tf.placeholder(tf.float32)
		self.is_training = tf.placeholder(tf.bool)
	
	def inference(self, x, reuse):
		with tf.variable_scope('output_layer', reuse=reuse):
			out = Layers.fc_layer(x, _CLASSES, self.weight_decay)
		return out
	
	def error(self, y, y_):
		return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

	def optimize(self, loss):
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.inverse_time_decay(self.learning_rate, global_step, _K_LEARNING_DECAY, _DECAY_RATE_LD)
		return tf.train.AdamOptimizer(lr).minimize(loss)

	def accuracy(self, logits, y_):
		prediction = tf.nn.softmax(logits)
		correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
		return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def build_graph(self):
		logits = self.inference(self.x, reuse=False)
		vali_logits = self.inference(self.x, reuse=True)
		self.predictions = tf.nn.softmax(vali_logits)
		self.loss = self.error(logits, self.y_) 
		self.optimizer = self.optimize(self.loss)
		self.acc = self.accuracy(vali_logits, self.y_)

class SparseAutoEncoder(object):
	"""docstring for AutoEncoder"""
	def __init__(self):
		super(SparseAutoEncoder, self).__init__()
		self.placeholders()

	def placeholders(self):
		self.x = tf.placeholder(np.float32, shape=[None, _INPUT_DIMENSION])
		self.learning_rate_sparse = tf.placeholder(tf.float32)
		self.reg = tf.placeholder(tf.float32)
		self.beta = tf.placeholder(tf.float32)
		self.rho = tf.placeholder(tf.float32)

	def kl_divergence(self, rho, rho_hat):
		return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

	def regularization(self, weights):
		return tf.nn.l2_loss(weights)

	def encode(self, x, reuse):
		with tf.variable_scope('enc_layer', reuse=reuse):
			fc = Layers.fc_layer(x, _ENCODING_DIMESION)
		return tf.nn.sigmoid(fc)

	def decode(self, h, reuse):
		with tf.variable_scope('dec_layer', reuse=reuse):
			fc = Layers.fc_layer(h, _INPUT_DIMENSION)
		return tf.nn.sigmoid(fc)
	
	def error(self, x):
		x_encoded = self.encode(x, reuse=False)
		rho_hat = tf.reduce_mean(x_encoded, axis=0)
		kl = self.kl_divergence(self.rho, rho_hat)
		x_decoded = self.decode(x_encoded, reuse=False)
		with tf.variable_scope('enc_layer', reuse=True):
			w1 = tf.get_variable("fc_weights")
		with tf.variable_scope('dec_layer', reuse=True):
			w2 = tf.get_variable("fc_weights")
		diff = x - x_decoded
		cost = 0.5*tf.reduce_mean(tf.reduce_sum(diff**2,axis=1), name="loss") + 0.5*self.reg*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)) + self.beta*tf.reduce_sum(kl)
		return cost

	def optimize(self, loss):
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.inverse_time_decay(self.learning_rate_sparse, global_step, _K_LEARNING_DECAY, _DECAY_RATE_LD)
		return tf.train.AdamOptimizer(lr).minimize(loss, name="optimizer")

	def build_graph(self):
		self.loss = self.error(self.x) 
		self.optimizer = self.optimize(self.loss)
		self.x_encoded = self.encode(self.x, reuse=True)

class StackedAutoEncoder(object):
	"""docstring for AutoEncoder"""
	def __init__(self):
		super(StackedAutoEncoder, self).__init__()
		self.placeholders()

	def placeholders(self):
		self.x = tf.placeholder(np.float32, shape=[None, _INPUT_DIMENSION])
		self.learning_rate_stacked = tf.placeholder(tf.float32)
		self.reg_stacked = tf.placeholder(tf.float32)
		self.noise = tf.placeholder(tf.string)
		self.fraction = tf.placeholder(tf.float32)

	def add_noise(self, x):
		if self.noise == "masking":
			x_corrupted = self.masking_noise(x, self.fraction)
		elif self.noise == "salt_and_pepper":
			x_corrupted = self.salt_and_pepper_noise(x, self.fraction)
		else:
			x_corrupted = x
		return x_corrupted

	def masking_noise(self, x, fraction):
		x_noise = x.copy()
		n_samples = x.shape[0]
		n_features = x.shape[1]
		for i in range(n_samples):
			mask = np.random.randint(0, n_features, fraction)
			for m in mask:
				x_noise[i][m] = 0.
		return x_noise
	
	def salt_and_pepper(self, x, fraction):
		x_noise = x.copy()
		n_features = x.shape[1]
		mn = x.min()
		mx = x.max()
		for i, sample in enumerate(x):
			mask = np.random.randint(0, n_features, fraction)
			for m in mask:
				if np.random.random() < 0.5:
					x_noise[i][m] = mn
				else:
					x_noise[i][m] = mx
		return x_noise

	def encode(self, x, reuse):
		with tf.variable_scope('enc_layer1', reuse=reuse):
			fc1 = Layers.fc_layer(x, _ENCODING_DIMESION*2)
		fc1 = tf.nn.sigmoid(fc1)
		with tf.variable_scope('enc_layer2', reuse=reuse):
			fc2 = Layers.fc_layer(fc1, _ENCODING_DIMESION)
		return tf.nn.sigmoid(fc2)

	def decode(self, h, reuse):
		with tf.variable_scope('dec_layer1', reuse=reuse):
			fc1 = Layers.fc_layer(h, _ENCODING_DIMESION*2, weight_decay=self.reg_stacked)
		fc1 = tf.nn.sigmoid(fc1)
		with tf.variable_scope('dec_layer2', reuse=reuse):
			fc2 = Layers.fc_layer(fc1, _INPUT_DIMENSION, weight_decay=self.reg_stacked)
		return tf.nn.sigmoid(fc2)

	def error(self, x):
		x_corrupted = self.add_noise(x)
		x_encoded = self.encode(x_corrupted, reuse=False)
		x_decoded = self.decode(x_encoded, reuse=False)
		diff = x - x_decoded
		cost = tf.sqrt(tf.reduce_mean(tf.square(diff)), name="loss")
		return cost

	def optimize(self, loss):
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.inverse_time_decay(self.learning_rate_stacked, global_step, _K_LEARNING_DECAY, _DECAY_RATE_LD)
		""" AdagradOptimizer, AdamOptimizer, GradientDescentOptimizer, """
		return tf.train.AdamOptimizer(lr).minimize(loss, name="optimizer")

	def build_graph(self):
		self.loss = self.error(self.x) 
		self.optimizer = self.optimize(self.loss)
		self.x_encoded = self.encode(self.x, reuse=True)

class Layers(object):
	"""docstring for Layers"""
	def __init__(self, arg):
		super(Layers, self).__init__()
		self.arg = arg

	@staticmethod
	def get_variables(name, shape, weight_decay=None, initializer=tf.contrib.layers.xavier_initializer()):
		if weight_decay is not None:
			regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
			new_variables = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)
		else:
			new_variables = tf.get_variable(name, shape=shape, initializer=initializer)
		return new_variables

	@staticmethod
	def fc_layer(input_layer, hidden_units, weight_decay=None):
	    input_dim = input_layer.get_shape()[1]
	    
	    fc_w = Layers.get_variables(name='fc_weights', 
	    	shape=[input_dim, hidden_units], 
	    	weight_decay=weight_decay)
	    fc_b = Layers.get_variables(name='fc_bias', 
	    	shape=[hidden_units],
	    	weight_decay=weight_decay)
	    fc_h = tf.matmul(input_layer, fc_w) + fc_b

	    return fc_h
	
	@staticmethod
	def bn_layer(input_layer):
		dimension = input_layer.get_shape()[1]

		mean, variance = tf.nn.moments(input_layer, axes=[0])
		beta = tf.get_variable('beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
		gamma = tf.get_variable('gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
		bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, _BN_EPSILON)

		return bn_layer
		