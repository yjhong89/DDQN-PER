import tensorflow as tf
import numpy as np 
import os, time
import utils

class Q_network():
	def __init__(self, args, name='Q_network'):
		print('Initializing networks')
		self.args = args
		self.name = name
		
		with tf.variable_scope(self.name):
			# Placeholders
			# State placeholder : [batch size, 84, 84, 4]
			self.states = tf.placeholder(tf.float32, [None, 84, 84, 4], name='State')
			# Action placeholder : [batch size, one hot encoded action]
			self.actions = tf.placeholder(tf.float32, [None, self.args.num_actions], name='Actions')
			# Reward placeholder
			self.rewards = tf.placeholder(tf.float32, [None], name='Rewards')
			# Q max placeholder
			self.q_max = tf.placeholder(tf.float32, [None])
			# Terminal placeholder
			self.terminals = tf.placeholder(tf.float32, [None], name='Terminals')  

			conv1, conv1_shape = utils.conv2d(self.states, output_dim=16, filter_height=8, filter_width=8, stride=4, name='Conv1')
			print('1st convolution layer shape : %s' % conv1_shape) 
			conv1_nl = tf.nn.relu(conv1)

			conv2, conv2_shape = utils.conv2d(conv1_nl, output_dim=32, filter_height=4, filter_width=4, stride=2, name='Conv2')
			print('2nd convolution layer shape : %s' % conv2_shape)
			conv2_nl = tf.nn.relu(conv2)

			conv_shape = conv2_nl.get_shape().as_list()
		
			# FC layer
			fc_flat = tf.reshape(conv2_nl, [-1, conv_shape[-1]*conv_shape[-2]*conv_shape[-3]])
			fc = utils.linear(fc_flat, self.args.final_fc, name='FC1')
			fc_nl = tf.nn.relu(fc)

			# [batch size, num_actions]
			self.q_value = utils.linear(fc_nl, self.args.num_actions, name='FC2')
	
			# q_target, q_pred : [batch,]
			# If terminal state, next state q : 0
			self.q_target = self.rewards + tf.mul(1-self.terminals, tf.mul(self.args.discount_factor,self.q_max))
			# Only get q value for corresponding action
			self.q_pred = tf.reduce_sum(tf.mul(self.q_value, self.actions), reduction_indices=1)

			self.td_error = self.q_target - self.q_pred
			# Accumulate weight change
			self.loss = tf.reduce_sum(0.5 * tf.pow(self.td_error, 2))


		self.tr_vrbs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
		for i in xrange(len(self.tr_vrbs)):
			print(self.tr_vrbs[i])
		
		if self.name == 'Q_net':
			self.optimizer = tf.train.RMSPropOptimizer(self.args.learning_rate, 0.99, 0, 1e-6)
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			# Get gradient of each trainable variables as g/v list
			self.grads = self.optimizer.compute_gradients(self.loss, self.tr_vrbs)
			# Convert gradient to tf.Variables to accumulate
			self.accum_vars = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.tr_vrbs]
#			self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.grads)]
			self.train_op = self.optimizer.apply_gradients([(self.accum_vars[i], gv[1]) for i, gv in enumerate(self.grads)], global_step=self.global_step)
