# thanks to @edersantana and @fchollet for suggestions & help.

import numpy as np
from ple import PLE  # our environment
#from ple.games.catcher import Catcher
from ple.games.waterworld import WaterWorld
import math
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import Conv2D

from example_support import ExampleAgent, ReplayMemory, loop_play_forever


class Agent(ExampleAgent):
    """
        Our agent takes 1D inputs which are flattened.
        We define a full connected model below.
    """

    def __init__(self, *args, **kwargs):
        ExampleAgent.__init__(self, *args, **kwargs)

        # self.state_dim = self.env.getGameStateDims()
        # self.state_shape = np.prod((num_frames,) + self.state_dim)
        # self.input_shape = (batch_size, self.state_shape)

    def build_model(self):
        model = Sequential()
        #model.add(Dense(
        #    input_dim=self.state_shape, output_dim=256, activation="relu", init="he_uniform"
        #))
        model.add(Dense(kernel_initializer="he_uniform", activation="relu", input_dim=self.state_shape, units=256))
        #model.add(Dense(
        #    512, activation="relu", init="he_uniform"
        #))
        model.add(Dense(512, kernel_initializer="he_uniform", activation="relu"))
        #model.add(Dense(
        #    self.num_actions, activation="linear", init="he_uniform"
        #))
        model.add(Dense(self.num_actions, kernel_initializer="he_uniform", activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
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

        model.compile(loss="mse", optimizer=self.optimizer)

        self.model = model
    def load_weights(self,filepath):
        self.model.load_weights(filepath)
    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)


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
    tmp = state['creep_pos']
    all_pos = np.array([])
    for color in colors:
        for position in tmp[color]:
            k = np.array(position)/256.0
            all_pos = np.append(all_pos,k)
    
    time = state['time']
    # if color == 0: # Green
    green = 5*math.cos(time)
    # elif color == 1: # Red
    sign = 1.0
    tmp = math.cos(time)+math.sin(time)
    if tmp < 0 :
        sign = -1.0
    red = 5*sign*(tmp**2)/2
    # elif color == 2: # Yellow
    yellow = 5*math.sin(time)
    # print time
    score = [green,red,yellow]
    all_dist = np.array([])
    tmp = state['creep_dist']

    # print(time)
    for color in colors:
        k = np.array(tmp[color])/362.1
        all_dist = np.append(all_dist,k)
    # all_pos = np.array(state['creep_pos'].values())/256.0
    #print(all_pos)
    all_state = np.append([x_pos,y_pos,x_vel,y_vel,time,green,red,yellow],all_pos)
    # print(all_state)
    # state = np.array([state.values()]) / max_values
    # print(all_state)
    return all_state.flatten()

if __name__ == "__main__":
    # this takes about 15 epochs to converge to something that performs decently.
    # feel free to play with the parameters below.
    # training parameters
    num_epochs = 120
    num_steps_train = 15000  # steps per epoch of training
    num_steps_test = 3000
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

    # memory = ReplayMemory(max_memory_size, min_memory_size)

    game = WaterWorld()
    env = PLE(game, fps=60)#, state_preprocessor=nv_state_preprocessor)

    agent = Agent(env, batch_size, num_frames, frame_skip, lr,
                discount, rng)
    agent.build_model2()
    agent.load_weights('test5.csv')
    env.init()
    loop_play_forever(env, agent)
