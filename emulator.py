import numpy as np
import cv2
import time
import sys
sys.path.append('/home/yjhong89/tensorflows/rl/atari_dqn/Arcade-Learning-Environment-master')
print(sys.path)
from ale_python_interface import ALEInterface

'''
Descrription of atari game environment
'''

class emulator:
	def __init__(self, rom_name, vis, windowname='preview'):
  		self.ale = ALEInterface()
		# When it starts
  		self.ale.setInt("random_seed", 123)
		# Skipping 4 frames 
  		self.ale.setInt("frame_skip", 4)
  		self.ale.loadROM('roms/' + rom_name)
  		self.legal_actions = self.ale.getMinimalActionSet()
  		print('Actions : %s' % self.legal_actions)
  		self.action_map = dict()
  		self.windowname = windowname
  		# Raw atari frames, 210 * 160 pixel images 
  		self.screen_width, self.screen_height = self.ale.getScreenDims()
  		print("widht/height: " + str(self.screen_width) + "/" + str(self.screen_height))
  		# Visualize
  		self.vis = vis
  		if vis:
   			cv2.startWindowThread()
   			cv2.namedWindow(self.windowname)

 	def get_image(self):
  		# Need to specify data type as uint8
  		numpy_surface = np.zeros([self.screen_width * self.screen_height * 3], dtype=np.uint8)
  		# get RGB values
  		self.ale.getScreenRGB(numpy_surface)
  		image = np.reshape(numpy_surface, [self.screen_height, self.screen_width, 3])
  		return image

 	def new_game(self):
  		self.ale.reset_game()
  		# Reset game and getting reset image value
  		return self.get_image()

 	def next(self, action_index):
  		# Get R(s,a)
  		reward = self.ale.act(action_index)
  		# Get image pixel value after taking an action
  		next_state = self.get_image()
  		if self.vis:
   			cv2.imshow(self.windowname, next_state)
  		# self.ale.game_over() returns True when game is over
  		return next_state, reward, self.ale.game_over()

if __name__ == "__main__":
 	engine = emulator('breakout.bin', False)
 	engine.next(0)
 	time.sleep(5)
