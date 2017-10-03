import numpy as np
import tensorflow as tf
import binary_tree
import os

'''
Prirotized experience memory
'''

class PER:
	def __init__(self, args):
		print('Initializing Priortized Experience Replay')
		self.args = args
		# Store (state, action, reward, next_state)
		self.size = self.args.replay_size

		# Batches
		self.batch_states = np.empty([self.args.batch_size, 84, 84, 4])
		self.batch_actions = np.empty([self.args.batch_size])
		self.batch_rewards = np.empty([self.args.batch_size])
		self.batch_terminals = np.empty([self.args.batch_size])
		self.batch_next_states = np.empty([self.args.batch_size, 84, 84, 4])

		# Count number of samples in binary tree
		self.counter = 0
		self.full_flag = False
		
		# Create binary tree
		self.bt = binary_tree.BinaryTree(self.size)
		# To normalize importance weight
		self.max_priority = 0


	def get_batches(self):
		print('Priority sum is %3.4f' % self.bt.root.value)
		divide_index = self.bt.root.value / self.args.batch_size
		self.batch_history = list()
		self.batch_priority = list()
		random_index = list()
		# To sample mini batch of size k, the range[0, sum] is divided eequally into k ranges
		for i in xrange(self.args.batch_size):
			random_number = (i * divide_index) + (np.random.random(1) * divide_index)
			#print('%d-th random number %3.3f between %3.3f : %3.3f' %(i+1, random_number, divide_index*i, divide_index*(i+1)))
			# 'transition' will be list of experience
			transition, priority = self.bt.retrieve(random_number)			
			# Store each batch track path to update transition priority
			self.batch_history.append(self.bt.track)
			self.batch_priority.append(priority)
			self.batch_states[i] = transition[0]
			self.batch_actions[i] = transition[1]
			self.batch_rewards[i] = transition[2]
			self.batch_terminals[i] = transition[3]
			self.batch_next_states[i] = transition[4]
			self.bt.renew_track()
		return self.batch_states, self.batch_actions, self.batch_rewards, self.batch_terminals, self.batch_next_states, self.bt.root.value, self.batch_history, self.batch_priority


	def insert(self, state, act, rwd, ter, next_state, priority):
		print('Insert priority : %3.4f' % priority)
		# Update maximum value
		if priority > self.max_priority:
			self.max_priority = priority
		# Stochastic prioritization
		weighted_priority = priority ** self.args.alpha
		# Make list of [states, actions, rewards, terminals, next_states]
		experience_list = list()
		experience_list.append(state)
		experience_list.append(act)
		experience_list.append(rwd)
		experience_list.append(ter)
		experience_list.append(next_state)

		if self.full_flag:
			# Replace old samples
			replace = self.bt.full_size(experience_id=experience_list, td_error=weighted_priority) 
			replace(node=None)
		else:
			# Insert to binary tree
			self.bt.insert(experience_id=experience_list, td_error=weighted_priority, node=None)

		# Update counter
		self.counter += 1
		
		# Check full flag
		if self.counter == self.size:
			self.full_flag = True
			self.counter = 0	

	@property
	def get_size(self):
		if self.full_flag:
			return self.size
		else:
			return self.counter
		


