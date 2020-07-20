# thanks to @edersantana and @fchollet for suggestions & help.

import numpy as np
from ple import PLE  # our environment
#from ple.games.catcher import Catcher
from ple.games.waterworld import WaterWorld
import math
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.optimizers import SGD,Adam

from example_support import ExampleAgent, ReplayMemory, loop_play_forever


class Agent(ExampleAgent):
	"""
		Our agent takes 1D inputs which are flattened.
		We define a full connected model below.
	"""

	def __init__(self, *args, **kwargs):
		ExampleAgent.__init__(self, *args, **kwargs)

		self.state_dim = self.env.getGameStateDims()
		#self.state_shape = np.prod((num_frames,) + self.state_dim)
		# self.input_shape = (batch_size, self.state_shape)
		# self.input_shape = (self.num_frames,) + self.frame_dim

	def build_model(self):
		model = Sequential()
		#model.add(Dense(
		#	input_dim=self.state_shape, output_dim=256, activation="relu", init="he_uniform"
		#))
		model.add(Dense(kernel_initializer="he_uniform", activation="relu", input_dim=self.state_shape, units=256))
		#model.add(Dense(
		#	512, activation="relu", init="he_uniform"
		#))
		model.add(Dense(512, kernel_initializer="he_uniform", activation="relu"))
		#model.add(Dense(
		#	self.num_actions, activation="linear", init="he_uniform"
		#))
		model.add(Dense(self.num_actions, kernel_initializer="he_uniform", activation="linear"))
		model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.lr))
		#model.compile(optimizer=SGD(lr=self.lr))

		self.model = model

	def build_model2(self):
		import tensorflow as tf
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		session = tf.Session(config=config)
		
		model = Sequential()
		model.add(Conv2D(16, (8, 8), input_shape=self.frame_dim+(self.num_frames,), activation="relu", kernel_initializer="he_uniform"))
		model.add(Conv2D(16, (4, 4), strides=(2, 2), kernel_initializer="he_uniform", activation="relu"))
		# model.add(Conv2D(32, (2, 2), strides=(1, 1), kernel_initializer="he_uniform", activation="relu"))
		model.add(Flatten())
		model.add(Dense(256, kernel_initializer="he_uniform", activation="relu"))

		model.add(Dense(5, kernel_initializer="he_uniform", activation="linear"))

		model.compile(loss='mse', optimizer=self.optimizer)

		self.model = model

	def save_weights(self, filepath, overwrite=False):
		self.model.save_weights(filepath, overwrite=overwrite)
	
	def self_loss(self,y_true,y_pred):
		colors = ['G','R','Y']
		state = self.env.getGameState()
                print state
		distance = state['creep_dist']
		DIST = list()
		for color in colors:
                    for d in distance[color]:
                        tmp = [color,d]
                        DIST.append(tmp)
		def takeSecond(elem):
                    return elem[1]
		DIST.sort(key=takeSecond)
		DIST = np.array(DIST)[:3]
		print DIST
		step = state['step']
		# print DIST
		value = abs(y_true-y_pred)
		for i in range(len(DIST)):
                    scored = -1.0*score(colors.index(DIST[i][0]),step)
                    value += scored*float(DIST[i][1])

		return value
def score_function(color, time):
        score = 0 

	if color == 'G': # Green
            score = 5*math.cos(time)

	elif color == 'R': # Red
			sign = 1.0
			tmp = math.cos(time)+math.sin(time)
			if tmp < 0 :
					sign = -1.0
			score = 5*sign*(tmp**2)/2

	elif color == 'Y': # Yellow
			score = 5*math.sin(time)

	return score 

def nv_state_preprocessor(state):
	"""
		This preprocesses our state from PLE. We rescale the values to be between
		0,1 and -1,1.
	"""

	# taken by inspection of source code. Better way is on its way!
	# max_values = np.array([64.0, 20.0, 64.0, 64.0])
	colors = ['G','R','Y']
	val = state.values()
	x_pos = state['player_x']/256.0
	y_pos = state['player_y']/256.0
	x_vel = state['player_velocity_x']/64.0
	y_vel = state['player_velocity_y']/64.0
	step = state['step']

	creep_pos = state['creep_pos']
	all_pos = np.empty((0,2))

	for color in colors:
		for position in creep_pos[color]:
			pos = np.array(position)/256.0
			all_pos = np.append(all_pos, [[color, pos]], axis = 0)

	#green, red, yellow = score_function(time)

	creep_dist = state['creep_dist']

	all_pos = np.array(state['creep_pos'].values())/256.0	

	all_dist = np.empty((0,2))
	for color in colors:
		for distance in creep_dist[color]:
			dist = np.array(distance)/362.1
			all_dist = np.append(all_dist, [[color, dist]], axis = 0)
	all_dist = all_dist[ all_dist[:,1].argsort() ][:5]
		
        score = 0
        sum_dist = 0.0
        for i in all_dist:
                sum_dist += float(i[1])
        #print sum_dist
        #print type(sum_dist)
        for dist in all_dist :
                #print "dist:{}".format(dist)
                color_score = score_function(dist[0],step)
                #print "norm_dist:{}".format(all_dist[:,1])
                norm_dist = float(dist[1]) / sum_dist
                #print color_score
                #print dist[1]
                score = -1 * float(color_score) * float(norm_dist)


	#print "sorted all dist:{}".format(all_dist)

	all_state = np.append([x_pos,y_pos,x_vel,y_vel],score)
	#all_state = np.append([x_pos,y_pos,x_vel,y_vel,time,green,red,yellow],all_pos)

	return all_state.flatten()


def score(color,time):
	import math
	ret = 0.0
	if color > 2:
		return
	if color == 0:
		ret = 5*math.cos(time)
	elif color == 1:
		sign = 1.0
		tmp = math.cos(time)+math.sin(time)
		if tmp < 0 :
			sign = -1.0
		ret = 5*sign*(tmp**2)/2
	elif color == 2:
		ret = 5*math.sin(time)
	if ret-int(ret)>=0.5 :
		ret = int(ret)+1
	elif ret-int(ret)<0.5 :
		ret = int(ret)
	return ret

def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
	   return v
	return v / norm



def preprocess(state,screen):
	colors = ['G','R','Y']
	pos = state['creep_pos']
	pos_G = pos['G']
	pos_r = pos['R']
	pos_y = pos['Y']
	scores = np.array([])
	step = state['step']
	for i in range(3):
		scores = np.append(scores,[score(i,step/50.0)])
	biggest = np.argmax(scores)
	smallest = np.argmin(scores)
	medium = 3-biggest-smallest
	for i in pos[colors[biggest]]:
		tmp = [int(k) for k in i]
		screen[tmp] = 500
	for i in pos[colors[smallest]]:
		tmp = [int(k) for k in i]
		screen[tmp] = 0
	for i in pos[colors[medium]]:
		tmp = [int(k) for k in i]
		screen[tmp] = 250
	ret = normalize(screen)
	ret = np.reshape(ret,(200,200,1))
	# print np.shape(ret)
	return ret


if __name__ == "__main__":
	# this takes about 15 epochs to converge to something that performs decently.
	# feel free to play with the parameters below.

	# training parameters
	num_epochs = 10
	num_steps_train = 5  # steps per epoch of training
	num_steps_test = 3
	update_frequency = 4  # step frequency of model training/updates

	# agent settings
	batch_size = 32
	num_frames = 10  # number of frames in a 'state'
	frame_skip = 2
	# percentage of time we perform a random action, help exploration.
	epsilon = 0.15
	epsilon_steps = 30000  # decay steps
	epsilon_min = 0.1
	lr = 0.01
	discount = 0.95  # discount factor
	rng = np.random.RandomState(24)

	# memory settings
	max_memory_size = 100000
	min_memory_size = 1000  # number needed before model training starts

	epsilon_rate = (epsilon - epsilon_min) / epsilon_steps

	# PLE takes our game and the state_preprocessor. It will process the state
	# for our agent.
	game = WaterWorld()
	env = PLE(game, fps=60, state_preprocessor=nv_state_preprocessor)

	agent = Agent(env, batch_size, num_frames, frame_skip, lr,
				  discount, rng)
	agent.build_model2()

	memory = ReplayMemory(max_memory_size, min_memory_size)

	env.init()

	for epoch in range(1, num_epochs + 1):
		steps, num_episodes = 0, 0
		losses, rewards = [], []
		env.display_screen = True

		# training loop
		while num_episodes < num_steps_train:
			episode_reward = 0.0
			agent.start_episode()

			while env.game_over() == False :#and steps < num_steps_train:
				state = env.getGameState()
				screen = env.getScreenGrayscale()
				screen = preprocess(state,screen)
				# screen = np.reshape(screen,(200,200,1))
				# print((screen[0]))
				#screen = screen[:,np.newaxis]
				reward, action = agent.act(screen, epsilon=epsilon)
				memory.add([screen, action, reward, env.game_over()])
				if steps % update_frequency == 0:
					loss = memory.train_agent_batch(agent)

					if loss is not None:
						losses.append(loss)
						epsilon = np.max([epsilon_min, epsilon - epsilon_rate])

				episode_reward += reward
				steps += 1

			if num_episodes % 5 == 0:
				print "Episode {:01d}: Reward {:0.1f}".format(num_episodes, episode_reward)
			# break
			rewards.append(episode_reward)
			num_episodes += 1
			agent.end_episode()
		# break
		print "\nTrain Epoch {:02d}: Epsilon {:0.4f} | Avg. Loss {:0.3f} | Avg. Reward {:0.3f}".format(epoch, epsilon, np.mean(losses), np.sum(rewards) / num_episodes)

		steps, num_episodes = 0, 0
		losses, rewards = [], []

		# display the screen
		env.display_screen =  True

		# slow it down so we can watch it fail!
		env.force_fps = False

		# testing loop
		while num_episodes < num_steps_test:
			episode_reward = 0.0
			agent.start_episode()

			while env.game_over() == False:# and steps < num_steps_test:
				state = env.getGameState()
				screen = env.getScreenGrayscale()
				screen = preprocess(state,screen)
				# screen = np.reshape(screen,(200,200,1))
				#screen = screen[:,np.newaxis]
				reward, action = agent.act(screen, epsilon=0.05)

				episode_reward += reward
				steps += 1

				# done watching after 500 steps.
				if steps > 500:
					env.force_fps = True
					env.display_screen = False

			if num_episodes % 1 == 0:
				print "Episode {:01d}: Reward {:0.1f}".format(num_episodes, episode_reward)

			rewards.append(episode_reward)
			num_episodes += 1
			agent.end_episode()

		print "Test Epoch {:02d}: Best Reward {:0.3f} | Avg. Reward {:0.3f}".format(epoch, np.max(rewards), np.sum(rewards) / num_episodes)
	agent.save_weights('./test6.csv')
	print "\nTraining complete. Will loop forever playing!"
	loop_play_forever(env, agent)
