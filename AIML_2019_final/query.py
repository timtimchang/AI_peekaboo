import numpy as np
import sys
import random

distributions = np.array([])

def load_distribution():
	with open('distribution.txt') as f:
		d = f.readlines()
		distribution = np.concatenate((distribution,d),axis = 0)	


def query_agent(agent_num,t):
	credit = random.radom()
	agent_credit = distribution[agent_num][t]

	if (credit > agent_credit):
		return 1 # for true action
	else:
		return 0 # for fake action


