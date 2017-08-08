import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import numpy as np
import argparse, time, os, sys
from ddqn import *

def main():
 	parser = argparse.ArgumentParser()
 	parser.add_argument('--visualize', type=str2bool, default='false')
 	parser.add_argument('--num_iterations', type=int, default=1e8)
 	parser.add_argument('--save_interval', type=int, default=50000)
 	parser.add_argument('--copy_interval', type=int, default=10000)
 	parser.add_argument('--log_interval', type=int, default=10000)
 	parser.add_argument('--replay_size', type=int, default=1000000)
 	parser.add_argument('--batch_size', type=int, default=32)
 	parser.add_argument('--num_actions', type=int, default=None)
 	parser.add_argument('--initial_eps', type=float, default=1.0)
 	parser.add_argument('--eps_min', type=float, default=0.1)
 	parser.add_argument('--eps_step', type=int, default=1000000)
 	parser.add_argument('--discount_factor', type=float, default=0.95)
	parser.add_argument('--epsilon', type=float, default=0.01)
 	parser.add_argument('--learning_rate', type=float, default=0.00025)
 	parser.add_argument('--img_scale', type=float, default=255.0)
 	parser.add_argument('--train_start', type=int, default=1000)
	parser.add_argument('--final_fc', type=int, default=256)
	parser.add_argument('--alpha', type=float, default=0.6)
	parser.add_argument('--initial_beta', type=float, default=0.4)
	parser.add_argument('--beta_step', type=int, default=100000)
 	parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
 	parser.add_argument('--log_dir', type=str, default='./logs')
 	parser.add_argument('--train', type=str2bool, default='true')

	args = parser.parse_args()
	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)


	run_config = tf.ConfigProto()
	run_config.log_device_placement=False
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		ddqn_per = Atari(args, sess)
		if args.train:
			ddqn_per.train()
		else:
			ddqn_per.eval()

def str2bool(v):
	if v.lower() in ('yes', 'y', '1', 'true', 't'):
		return True
	elif v.lower() in ('no', 'n', '0', 'false', 'f'):
		return False

if __name__ == "__main__":
 	main()
  
 
