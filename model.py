import tensorflow as tf
import numpy as np
import functools
import math

_CLASSES = 2
_INPUT_DIMENSION = 122
_BN_EPSILON = 0.001
_HIDDEN_UNITS = 20
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
		with tf.variable_scope('first_layer', reuse=reuse):
			fc1 = tf.nn.relu(Layers.fc_layer(x, _HIDDEN_UNITS, self.weight_decay))
			fc1_dp = tf.layers.dropout(fc1, self.dropout, training=self.is_training)
		with tf.variable_scope('output_layer', reuse=reuse):
			out = Layers.fc_layer(fc1_dp, _CLASSES, self.weight_decay)
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
		self.learning_rate = tf.placeholder(tf.float32)
		self.alpha = tf.placeholder(tf.float32)
		self.beta = tf.placeholder(tf.float32)
		self.rho = tf.placeholder(tf.float32)

	def init_weights(self, shape):
		r = math.sqrt(6) / math.sqrt(_INPUT_DIMENSION + _ENCODING_DIMESION + 1)
		weights = tf.random_normal(shape, stddev=r)
		return tf.Variable(weights)

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
		cost = 0.5*tf.reduce_mean(tf.reduce_sum(diff**2,axis=1)) + 0.5*self.alpha*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)) + self.beta*tf.reduce_sum(kl)
		return cost

	def optimize(self, loss):
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.inverse_time_decay(self.learning_rate, global_step, _K_LEARNING_DECAY, _DECAY_RATE_LD)
		return tf.train.AdamOptimizer(lr).minimize(loss)

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
		