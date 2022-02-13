import time
import gym
from gym import spaces
import numpy as np
from gym_game.envs import pong

class PongEnv(gym.Env):

  metadata = {'render.modes': ['human']}

def __init__(self):
        super(PongEnv, self).__init__()

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
                low=0, high=1, shape=(600, 800, 3), dtype=np.uint8
            )
        self.game=pong

  def step(self, action):
      if self.game.checkforscore == 1:
          reward = -1
          return reward
      elif self.game.checkforscore == -1:
        reward = -1
        return reward

  def reset(self):
    pong.gamerunning = False
    return pong.get_gamestate()

  def render(self, mode='human'):
    if mode == "human":
        time.sleep(0.1)
        self.game.render(mode=mode)

  def seed(self, seed=None):
      np.random.seed(seed)