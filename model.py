import tensorflow as tf
import numpy as np 
import os, time
import utils

class Q_network():
	def __init__(self, args, name='Q_network'):
		self.args = args
		self.name = name
		print('Initializing %s networks' % self.name)
		
		with tf.variable_scope(self.name):
			# Placeholders
			# State placeholder : [batch size, 84, 84, 4]
			self.states = tf.placeholder(tf.float32, [None, 84, 84, 4], name='State')
			# Action placeholder : [batch size, one hot encoded action]
			self.actions = tf.placeholder(tf.float32, [None, self.args.num_actions], name='Actions')
			# Reward placeholder
			self.rewards = tf.placeholder(tf.float32, [None], name='Rewards')
			# Q max placeholder
			self.q_max = tf.placeholder(tf.float32, [None], name='Q_max')
			# Terminal placeholder
			self.terminals = tf.placeholder(tf.float32, [None], name='Terminals')  

			self.conv1 = utils.conv2d(self.states, output_dim=16, filter_height=8, filter_width=8, stride=4, name='Conv1')
			self.conv1_nl = tf.nn.relu(self.conv1)

			self.conv2 = utils.conv2d(self.conv1_nl, output_dim=32, filter_height=4, filter_width=4, stride=2, name='Conv2')
			self.conv2_nl = tf.nn.relu(self.conv2)

			conv_shape = self.conv2_nl.get_shape().as_list()
		
			# FC layer
			self.fc_flat = tf.reshape(self.conv2_nl, [-1, conv_shape[-1]*conv_shape[-2]*conv_shape[-3]])
			self.fc = utils.linear(self.fc_flat, self.args.final_fc, name='FC1')
			self.fc_nl = tf.nn.relu(self.fc)

			# [batch size, num_actions]
			self.q_value = utils.linear(self.fc_nl, self.args.num_actions, name='FC2')
	
			# q_target, q_pred : [batch,]
			# If terminal state, next state q : 0
			self.q_target = self.rewards + tf.multiply(1-self.terminals, tf.multiply(self.args.discount_factor,self.q_max))
			# Only get q value for corresponding action
			self.q_pred = tf.reduce_sum(tf.multiply(self.q_value, self.actions), reduction_indices=1)
			self.td_error = self.q_target - self.q_pred

			self.tr_vrbs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
			for i in xrange(len(self.tr_vrbs)):
				print(self.tr_vrbs[i])

		if self.name == 'Q_net':
			# Need to bias annealing with importance sampling weight
			self.is_weight = tf.placeholder(tf.float32, [None], name='IS_Weight')
			# w*(td_error)^2
			self.loss = tf.reduce_sum(0.5 * self.is_weight * tf.pow(self.td_error, 2))
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.train_op = tf.train.RMSPropOptimizer(self.args.learning_rate, 0.99, 0, 1e-6).minimize(self.loss, global_step=self.global_step)
			# Get gradient of each trainable variables as g/v list
			#grads, vrbs = zip(*self.optimizer.compute_gradients(self.loss))
			# Convert gradient to tf.Variables to accumulate
			#self.accum_vars = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.tr_vrbs]
			#self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.grads)]
			#self.train_op = self.optimizer.apply_gradients([(self.accum_vars[i], gv[1]) for i, gv in enumerate(self.grads)], global_step=self.global_step)
			#self.train_op = self.optimizer.apply_gradients(zip(grads, vrbs))
