import numpy as np
import tensorflow as tf
import math

class Leaf():
	def __init__(self, experience_id, td_error):
		self.parent = None
		self.left = None
		self.num_leftchild = 0
		self.num_rightchild = 0
		self.right = None
		self.experience = experience_id
		self.value = td_error


# Unsorted binary sum tree
class BinaryTree():
	def __init__(self, size):
		self.root = None
		self.size = size
		self.counter = 0
		# Including root
		self.num_element = 1
		self.track = list()
		self.update_flag = False
	
	# For recursive insert
	def insert(self, experience_id, td_error, node=None):

		if node is None:
			node = self.root

		if self.root is None:
			self.root = Leaf(experience_id, td_error)
	
		# In case root is not None
		else:
			if node.left is None:
				# Move node`s vlaue to node`s left childe
				node.left = Leaf(node.experience, node.value)
				# Add new node to right child
				node.right = Leaf(experience_id, td_error)
				node.num_leftchild += 1
				node.num_rightchild += 1
				self.node_update(node=node)
				print('\n')
				# Count number of element
				self.num_element += 1
				return
			# New layer need to be added 
			else:
				#print('Number of left child : %d, right child : %d' %(node.num_leftchild, node.num_rightchild))
				left_full, left_height = self.is_full(node.num_leftchild)
				right_full, right_height = self.is_full(node.num_rightchild)
				#print('Left {},{} || Right {},{}'.format(left_full, left_height, right_full, right_height))
				if (left_height != right_height):
					if (left_full is not True):
						self.insert(experience_id, td_error, node=node.left)
					else:
						self.insert(experience_id, td_error, node=node.right)
				else:
					if right_full:
						self.insert(experience_id, td_error, node=node.left)
					else:
						self.insert(experience_id, td_error, node=node.right)
				
	def is_full(self, num_of_element):
		exp = 0
		pwr = 1
		while pwr < (num_of_element+1):
			exp += 1
			pwr *= 2
		# If num_of_element+1 is power of 2, binary tree is full
		# 'exp' will be height of binary tree
		if pwr == num_of_element+1:
			return True, exp
		else:
			return False, exp

	# When binary tree has saturated, change Leaf node to new node from oldest sample
	def full_size(self, experience_id, td_error):
		if self.counter > self.size:
			self.counter = 0
		
		# Get digit of binary tree size
		digits = int(math.floor(math.log(self.size)))
		# Returns string, convert it in reverse way
		binary_path = list(bin(self.counter)[::-1][:-2])
		# Fill binary path list until it has 'digits' element
		for _ in xrange(len(binary_path), digits+1, 1):
			binary_path.append(0)
		# 0 : Left, 1: Right
		binary_path.reverse()
		print(binary_path)
		def search(node=None):
			#print('Searching leaf from root')
			if node is None:
				node = self.root
	
			for path_index, path in enumerate(binary_path):
				if int(path):
					#print('Going right\n')
					node = node.right
				else:
					#print('Going left\n')
					node = node.left
	
				# Replace node`s experience and priority
				if node.left is None:
					#print('Arrive target leaf\n')
					node.experience = experience_id
					node.value = td_error
					# Update node`s value
					self.node_update(node=node.parent)
					return
		self.counter += 1
		return search

	def update_transition(self, updated_priority, track_path):
		node = self.root
		self.update_flag = True
		print(track_path)
		for _, track in enumerate(track_path):
			if track:
				#print('Going right\n')
				node = node.right
			else:
				#print('Going left\n')
				node = node.left

			if node.left is None:
				#print('Arrived target leaf\n')
				node.value = updated_priority
				# Update node`s value
				self.node_update(node=node.parent)
		self.update_flag = False
		return
		
		
	
	def node_update(self, node=None):
		node.left.parent = node
		node.right.parent = node
		node.value = node.left.value + node.right.value
		#print('Node left value : %3.3f, Node right value : %3.3f, Node parent value : %3.3f' %(node.left.value, node.right.value, node.value))
		if node.parent is None:
			#print('Here is Root')
			return
		elif node.parent.left == node:
			if (self.num_element == self.size) or self.update_flag:
				pass
			else:
				node.parent.num_leftchild += 2
		elif node.parent.right == node:
			if (self.num_element == self.size) or self.update_flag:
				pass
			else:
				node.parent.num_rightchild += 2
		else:
			raise Exception('No parents!')
		#print('Left child %d, right child %d' %(node.parent.num_leftchild, node.parent.num_rightchild))
		# Parent`s value update
		self.node_update(node=node.parent)

	def renew_track(self):
		self.track = list()

	def retrieve(self, s, node=None):
		if node is None:
			node = self.root
		
		# If node is leaf node, return experience and priority
		if node.left is None:
			return node.experience, node.value
		
		if node.left.value >= s:
			self.track.append(0)	
			return self.retrieve(s, node=node.left)
		else:
			self.track.append(1)
			return self.retrieve(s - node.left.value, node=node.right)
			
		
if __name__ == "__main__":
	bt = BinaryTree(8)		
	bt.insert(100,11)
	bt.insert(100,9)
	bt.insert(100,30)
	bt.insert(100,20)
	bt.insert(100,15)
	bt.insert(100,3)
	bt.insert(100,5)
	bt.insert(100,6)
	print(bt.num_element)
	_,v = bt.retrieve(45)
	print(bt.track, v)
	bt.update_transition(31)
#	a=bt.full_size(100,10)
#	a(node=None)
