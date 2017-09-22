import numpy as np
import tensorflow as tf
import os, time

LOG_DIR = './logs'
TRAIN = 'train.csv'
EVAL = 'eval.csv'

def initialize_log():
	train_path = os.path.join(LOG_DIR, TRAIN)
	eval_path = os.path.join(LOG_DIR, EVAL)
	if os.path.exists(train_path):
		train_log = open(train_path, 'a')
	else:
		print('Initialize log..')
		train_log = open(train_path, 'w')
		train_log.write('Step\t'+',episode.rwd\t'+',episode.q\t'+',epsilon\t'+',time\n')
	if os.path.exists(eval_path):
		eval_log = open(eval_path, 'a')
	else:
		eval_log = open(eval_path, 'w')
		eval_log.write('Step\t'+',episode.rwd\t+'+',episode.q\t'+',epsilon\t'+',time\n')

	return train_log, eval_log


def write_log(steps, total_rwd, total_q, num_episode, epsilon, mode, total_loss = 0):
	train_log, eval_log = initialize_log()

	if mode == 'train':
		print('At Training step %d, %d-th episode => total.Q : %3.4f, total.rwd : %3.4f' % \
		(steps, num_episode, total_q, total_rwd, total_loss))
		train_log.write('%d\t,%3.4f\t,%3.4f\t,%3.6f\t\n' % (steps, total_rwd, total_q, epsilon))
		train_log.flush()
	elif mode == 'eval':
		print('At Evaluation step %d, %d-th episode => total.Q : %3.4f, total.rwd : %3.4f' % \
		(steps, num_episode, total_q, total_rwd))
		eval_log.write('%d\t,%3.4f\t,%3.4f\t,%3.6f\t\n' % (steps, total_rwd, total_q, epsilon))
		eval_log.flush()


# Define convolutional layer
# 'inp' : [batch, in_height, in_widht, in_channels]
# truncated_normal : bound with 2*stddev
def conv2d(inp, output_dim, filter_height, filter_width, stride,stddev=0.02, name=None):
	with tf.variable_scope(name or 'conv2d'):
		weight = tf.get_variable('weight', [filter_height, filter_width, inp.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
		weight.initializer.run()
		# padding=SAME : ceil(float(in_height))/float(stride)
		# padding=VALID: cell(float(in_height-filter_height)+1)/float(stride)
		conv = tf.nn.conv2d(inp, weight, strides=[1,stride,stride,1], padding='VALID')
		bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0))
		bias.initializer.run()
		conv_wb = tf.add(conv,bias)
	return conv_wb
 
def linear(inp, output_size, name=None, stddev=0.02):
	with tf.variable_scope(name or 'linear'):
		weight = tf.get_variable('weight', [inp.get_shape()[-1], output_size], initializer=tf.truncated_normal_initializer(stddev=stddev))
		weight.initializer.run()
		bias = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0))
		bias.initializer.run()
		weighted_sum = tf.matmul(inp, weight) + bias
   	return weighted_sum



	
	
