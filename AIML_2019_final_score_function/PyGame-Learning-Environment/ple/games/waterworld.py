import pygame
import sys
import math
import time
from .score import *
#import .base
from .base.pygamewrapper import PyGameWrapper

from .utils.vec2d import vec2d
from .utils import percent_round_int
from pygame.constants import K_w, K_a, K_s, K_d
from .primitives import Player, Creep


class WaterWorld(PyGameWrapper):
	"""
	Based Karpthy's WaterWorld in `REINFORCEjs`_.

	.. _REINFORCEjs: https://github.com/karpathy/reinforcejs

	Parameters
	----------
	width : int
		Screen width.

	height : int
		Screen height, recommended to be same dimension as width.

	num_creeps : int (default: 9)
		The number of creeps on the screen at once.
	"""

	def __init__(self,
				 width=256,
				 height=256,
				 num_creeps=9):

		actions = {
			"up": K_w,
			"left": K_a,
			"right": K_d,
			"down": K_s
		}



		PyGameWrapper.__init__(self, width, height, actions=actions)
		# rewards = {
		# 	"green":score(0,pygame.time.get_ticks()),
		# 	"red":score(1,pygame.time.get_ticks()),
		# 	"yellow":score(2,pygame.time.get_ticks())
		# }
		# self.adjustRewards(rewards)
		self.BG_COLOR = (255, 255, 255)
		self.N_CREEPS = num_creeps
		self.CREEP_TYPES = ["G", "R", "Y"]
		self.CREEP_COLORS = [(50, 150, 50), (150, 50, 50), (150,150,50)]
		radius = percent_round_int(width, 0.025)
		self.CREEP_RADII = [radius, radius, radius]
		self.CREEP_REWARD = [
			self.rewards["green"],
			self.rewards["red"],
			self.rewards["yellow"]]
		self.CREEP_SPEED = 0.05 * width
		self.AGENT_COLOR = (20, 20, 140)
		self.AGENT_SPEED = 0.05 * width
		self.AGENT_RADIUS = radius * 2
		self.AGENT_INIT_POS = (self.width / 2, self.height / 2)
		self.stepnum = 0
		self.creep_counts = {
			"G": 0,
			"R": 0,
			"Y": 0
		}

		self.dx = 0
		self.dy = 0
		self.player = None
		self.creeps = None

	def _handle_player_events(self):
		self.dx = 0
		self.dy = 0
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()

			if event.type == pygame.KEYDOWN:
				key = event.key

				if key == self.actions["left"]:
					self.dx -= self.AGENT_SPEED

				if key == self.actions["right"]:
					self.dx += self.AGENT_SPEED

				if key == self.actions["up"]:
					self.dy -= self.AGENT_SPEED

				if key == self.actions["down"]:
					self.dy += self.AGENT_SPEED

	def _add_creep(self,creeptype):
		creep_type = creeptype#self.rng.choice([0, 1, 2])
		#creep_type = self.rng.choice(0)

		creep = None
		pos = (0, 0)
		dist = 0.0

		while dist < 1.5:
			radius = self.CREEP_RADII[creep_type] * 1.5
			pos = self.rng.uniform(radius, self.height - radius, size=2)
			dist = math.sqrt(
				(self.player.pos.x - pos[0])**2 + (self.player.pos.y - pos[1])**2)

		creep = Creep(
			self.CREEP_COLORS[creep_type],
			self.CREEP_RADII[creep_type],
			pos,
			self.rng.choice([-1, 1], 2),
			self.rng.rand() * self.CREEP_SPEED,
			self.CREEP_REWARD[creep_type],
			self.CREEP_TYPES[creep_type],
			self.width,
			self.height,
			self.rng.rand()
		)

		self.creeps.add(creep)
		self.creep_counts[self.CREEP_TYPES[creep_type]] += 1

	def getGameState(self):
		"""

		Returns
		-------

		dict
			* player x position.
			* player y position.
			* player x velocity.
			* player y velocity.
			* player distance to each creep


		"""

		state = {
			"player_x": self.player.pos.x,
			"player_y": self.player.pos.y,
			"player_velocity_x": self.player.vel.x,
			"player_velocity_y": self.player.vel.y,
			"creep_dist": {
				"G": [],
				"R": [],
				"Y": []
			},
			"creep_pos": {
				"G": [],
				"R": [],
				"Y": []
			},
			#"time" : pygame.time.get_ticks()/1000.0
			"time" : self.stepnum 
		}

		for c in self.creeps:
			dist = math.sqrt((self.player.pos.x - c.pos.x) **
							 2 + (self.player.pos.y - c.pos.y)**2)
			state["creep_dist"][c.TYPE].append(dist)
			state["creep_pos"][c.TYPE].append([c.pos.x, c.pos.y])

		return state

	def getScore(self):
		# print("score:",self.score)
		return self.score

	def game_over(self):
		"""
			Return bool if the game has 'finished'
		"""
		return (self.creep_counts['G']==0 or self.stepnum > 2500)

	def init(self):
		"""
			Starts/Resets the game to its inital state
		"""
		self.creep_counts = {"G": 0, "R": 0, "Y": 0 }

		if self.player is None:
			self.player = Player(
				self.AGENT_RADIUS, self.AGENT_COLOR,
				self.AGENT_SPEED, self.AGENT_INIT_POS,
				self.width, self.height
			)

		else:
			self.player.pos = vec2d(self.AGENT_INIT_POS)
			self.player.vel = vec2d((0.0, 0.0))

		if self.creeps is None:
			self.creeps = pygame.sprite.Group()
		else:
			self.creeps.empty()

		for i in [0,1,2]:
			for j in range(self.N_CREEPS//3):
				self._add_creep(i)
		# for i in range(self.N_CREEPS):
		# 	self._add_creep()

		self.score = 0
		self.ticks = 0
		self.lives = -1
		self.stepnum = 0

	def step(self, dt):
		"""
			Perform one step of game emulation.
		"""
		dt /= 1000.0
		self.stepnum += 1
		self.screen.fill(self.BG_COLOR)
		self.score += self.rewards["tick"]
		self._handle_player_events()
		self.player.update(self.dx, self.dy, dt)

		hits = pygame.sprite.spritecollide(self.player, self.creeps, True)
		for creep in hits:
			self.creep_counts[creep.TYPE] -= 1
			#self.score += creep.reward
			#add_score = score(creep.reward,pygame.time.get_ticks()/1000.0)
			#print "creep:{}".format(creep.TYPE)
			if creep.TYPE == 'G': temp = 0
			if creep.TYPE == 'R': temp = 1
			if creep.TYPE == 'Y': temp = 2
			add_score = score(temp, self.stepnum/ 100)
			self.score += add_score
			self._add_creep(self.CREEP_TYPES.index(creep.TYPE))
	

		#if self.creep_counts["G"] == 0:
		#	self.score += self.rewards["win"]

		self.creeps.update(dt)

		self.player.draw(self.screen)
		self.creeps.draw(self.screen)

if __name__ == "__main__":
	import numpy as np

	pygame.init()
	game = WaterWorld(width=256, height=256, num_creeps=10)
	game.clock = pygame.time.Clock()
	game.rng = np.random.RandomState(24)
	game.init()

	while True:
		dt = game.clock.tick_busy_loop(30)
		game.step(dt)
		pygame.display.update()
