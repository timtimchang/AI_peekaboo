from ple import PLE
from ple.games.waterworld import WaterWorld
from tests.test_ple_doom import NaiveAgent as MyAgent

game = WaterWorld()
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

myAgent = MyAgent(p.getActionSet())

nb_frames = 1000
reward = 0.0

for f in range(nb_frames):
	if p.game_over(): #check if the game is over
		p.reset_game()

	obs = p.getScreenRGB()
	action = myAgent.pickAction(reward, obs)
	reward = p.act(action)
	#print("reward:",reward)
