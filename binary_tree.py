import numpy as np
import tensorflow as tf


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
	def __init__(self):
		self.root = None
	
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
				print('Node parent : %s' % node.parent)
				self.node_update(node=node)
				print('\n')
				return
			# New layer need to be added 
			else:
				print('Number of left child : %d, right child : %d' %(node.num_leftchild, node.num_rightchild))
				left_full, left_height = self.is_full(node.num_leftchild)
				right_full, right_height = self.is_full(node.num_rightchild)
				print('Left {},{} || Right {},{}'.format(left_full, left_height, right_full, right_height))
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

	
	def node_update(self, node=None):
		node.left.parent = node
		node.right.parent = node
		node.value = node.left.value + node.right.value
		print('Node left value : %3.3f, Node right value : %3.3f, Node parent value : %3.3f' %(node.left.value, node.right.value, node.value))
		if node.parent is None:
			print('Root')
			return
		elif node.parent.left == node:
			print('Parent`s left child')
			node.parent.num_leftchild += 2
		elif node.parent.right == node:
			print('Parent`s right child')
			node.parent.num_rightchild += 2
		else:
			raise Exception('No parents!')
		print('Left child %d, right child %d' %(node.parent.num_leftchild, node.parent.num_rightchild))
		# Parent`s value update
		self.node_update(node=node.parent)


	def retrieve(self, s, node=None):
		if node is None:
			node = self.root
		
		# If node is leaf node, return experience and priority
		if node.left is None:
			return node.exeperience_id, node.value
		
		if node.left.value >= s:
			return self.retrieve(s, node=node.left)
		else:
			return self.retrieve(s - node.left.value, node=node.right)
			
		
if __name__ == "__main__":
	bt = BinaryTree()		
	bt.insert(100,11)
	bt.insert(100,9)
	bt.insert(100,30)
	bt.insert(100,20)
	bt.insert(100,15)
	bt.insert(100,3)
	bt.insert(100,5)
	bt.insert(100,6)
