from __future__ import absolute_import, division, print_function

import numpy as np

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec

import PyPipeline as pp
import time

class MarioEnvironment(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(4,), dtype=np.float32, minimum=0, maximum=1, name='action')

    min_distance = [0] * 19
    max_distance = [12] * 19

    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(24,), dtype=np.float32, 
        minimum=[-4.5, -4.5, -4.5, -4.5, 0] + min_distance, 
        maximum=[4.5, 4.5, 4.5, 4.5, 1] + max_distance, 
        name='observation')

    self.OBS_DISTANCE = 'distance'
    self.OBS_MARIO_POSITION = 'marioPosition'
    self.OBS_MARIO_ROTATION = 'marioRotation'
    self.OBS_COIN_POSITION = 'coinPosition'
    self.OBS_SMALL_COINS_COLLECTED = 'smallCoinsCollected' 
    self.OBS_SMALL_COIN_DISTANCES = 'smallCoinDistances' 

    self.INDEX_DISTANCE = 0
    self.INDEX_SMALL_COINS_COLLECTED = 1
    self.INDEX_MARIO_X = 2
    self.INDEX_MARIO_Y = 3
    self.INDEX_COIN_X = 4
    self.INDEX_COIN_Y = 5
    self.INDEX_MARIO_ROTATION = 6
    self.INDEX_SMALL_COIN_DISTANCE = 7

    self.start_time = time.time()
    self.sleep_time = 0.2
    self.game_duration = 8
    self.prev_vector_obs = np.array([0] * 24, dtype=np.float32)
    self.prev_distance = 12
    self.min_distance = 12
    self.collected_coins = 0


  def reset(self):
    """Return initial_time_step."""
    self._current_time_step = self._reset()
    return self._current_time_step


  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self.start_time = time.time()
    pp.write_gameover()
    time.sleep(self.sleep_time)

    self.prev_vector_obs = np.array([0] * 26, dtype=np.float32)
    self.prev_vector_obs[0] = 12

    obs = self.get_observation()
    self.prev_distance = obs.pop(0)
    self.min_distance = self.prev_distance
    self.collected_coins = obs.pop(0)
    
    return ts.restart(obs)

  def _step(self, action):
    pp.write_action(action)
    time.sleep(self.sleep_time)
    time_elapsed = time.time() - self.start_time

    prev_distance = self.prev_distance
    obs = self.get_observation()
    distance = obs.pop(0)
    latest_collected_coins = obs.pop(0)
    self.prev_vector_obs = obs
    
    did_collect = False
    if distance < 0.9:
      did_collect = True

    reward = self.calculate_reward(distance, prev_distance, latest_collected_coins)
    self.collected_coins = latest_collected_coins
    discount = (self.game_duration - time_elapsed) / self.game_duration
    
    timestep = None
    if time_elapsed > self.game_duration or did_collect:
      self.reset()
      timestep = ts.termination(obs, reward=reward)
    else:
      timestep = ts.transition(obs, reward=reward, discount=discount)        

    return timestep


  def calculate_reward(self, distance, prev_distance, latest_collected_coins):
    reward = 0

    if distance < 0.9:
      reward = 5000

    if distance < self.min_distance:
        reward += 100
        self.min_distance = distance
    elif distance < prev_distance:
        reward += 10
    else:
      reward -= 10

    collected_coin_diff = latest_collected_coins - self.collected_coins
    if collected_coin_diff > 0:
      reward += collected_coin_diff * 500

    return reward


  def get_observation(self):
    obs_dict = pp.read_observation()
    obs = [self.prev_distance] + [self.collected_coins] + self.prev_vector_obs

    if self.OBS_DISTANCE in obs_dict:
      obs[self.INDEX_DISTANCE] = obs_dict[self.OBS_DISTANCE]
      obs[self.INDEX_SMALL_COINS_COLLECTED] = obs_dict[self.OBS_SMALL_COINS_COLLECTED]
      obs[self.INDEX_MARIO_X] = obs_dict[self.OBS_MARIO_POSITION][0]
      obs[self.INDEX_MARIO_Y] = obs_dict[self.OBS_MARIO_POSITION][1]
      obs[self.INDEX_COIN_X] = obs_dict[self.OBS_COIN_POSITION][0]
      obs[self.INDEX_COIN_Y] = obs_dict[self.OBS_COIN_POSITION][1]
      distances = obs_dict[self.OBS_SMALL_COIN_DISTANCES]
      for d in distances:
        obs.append(d)

    return obs