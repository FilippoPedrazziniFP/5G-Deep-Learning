import tensorflow as tf
import numpy as np
import functools

_CLASSES = 2
_INPUT_DIMENSION = 122
_BN_EPSILON = 0.001
_HIDDEN_UNITS = 15
_ENCODING_DIMESION = 30
_K_LEARNING_DECAY = 0.5
_DECAY_RATE_LD = 0.96

class DeepModel(object):
	"""docstring for DeepModel"""
	def __init__(self):
		super(DeepModel, self).__init__()
		self.placeholders()

	def placeholders(self):
		self.x = tf.placeholder(np.float32, shape=[None, _INPUT_DIMENSION])
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

class Layers(object):
	"""docstring for Layers"""
	def __init__(self, arg):
		super(Layers, self).__init__()
		self.arg = arg

	@staticmethod
	def get_variables(name, shape, weight_decay, initializer=tf.contrib.layers.xavier_initializer()):
		regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
		new_variables = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)
		return new_variables

	@staticmethod
	def fc_layer(input_layer, hidden_units, weight_decay):
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
		