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

import UnityInterface as ui

class MarioEnvironment(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(4,), dtype=np.float32, minimum=0, maximum=1, name='action')

    self.COUNT_SMALL_COIN_DISTANCES = 19
    self.OBSERVATION_COUNT = 22
    self.MAX_DISTANCE = 12
    self.COUNT_PREV_MARIO_POS = 0
    min_distance = [0] * self.COUNT_SMALL_COIN_DISTANCES
    max_distance = [self.MAX_DISTANCE] * self.COUNT_SMALL_COIN_DISTANCES

    min_prev_pos = [-4.5] * self.COUNT_PREV_MARIO_POS
    max_prev_pos = [4.5] * self.COUNT_PREV_MARIO_POS

    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(self.OBSERVATION_COUNT,), dtype=np.float32, 
        minimum=[-4.5, -4.5, 0] + min_prev_pos + min_distance, 
        maximum=[4.5, 4.5, 1] + max_prev_pos + max_distance, 
        name='observation')

    self.OBS_DISTANCE = 'distance' 
    self.OBS_MARIO_POSITION = 'marioPosition'
    self.OBS_MARIO_ROTATION = 'marioRotation'
    self.OBS_SMALL_COINS_COLLECTED = 'smallCoinsCollected' 
    self.OBS_SMALL_COIN_DISTANCES = 'smallCoinDistances' 

    self.INDEX_MARIO_X = 0
    self.INDEX_MARIO_Y = 1
    self.INDEX_MARIO_ROTATION = 2
    self.INDEX_PREV_MARIO_POSITIONS = 3
    self.INDEX_SMALL_COIN_DISTANCE = self.INDEX_PREV_MARIO_POSITIONS + self.COUNT_PREV_MARIO_POS - 1

    self.START_GAME_DURATION = 8
    self.BONUS_GAME_DURATION = 0

    self.start_time = time.time()
    self.sleep_time = 0.1
    self.game_duration = self.START_GAME_DURATION

    self.prev_vector_obs = np.array([0] * self.OBSERVATION_COUNT, dtype=np.float32)
    self.collected_coins = 0
    self.position_history = []
    self.prev_distance = self.MAX_DISTANCE
    self.min_distance = self.MAX_DISTANCE

    self.MAX_POSITION_HISTORY = 30

    self.reset_type = 1
    self.total_reward = 0

    self.interface = ui.UnityInterface.getInstance(1)
    self.env_index = self.interface.init_count


  def reset(self):
    """Return initial_time_step."""
    self._current_time_step = self._reset()
    self.total_reward = 0
    return self._current_time_step


  def action_spec(self):
    return self._action_spec


  def observation_spec(self):
    return self._observation_spec


  def _reset(self):
    self.start_time = time.time()
    self.game_duration = self.START_GAME_DURATION
    pp.write_gameover(self.reset_type)
    time.sleep(self.sleep_time)

    self.prev_vector_obs = np.array([0] * self.OBSERVATION_COUNT, dtype=np.float32)

    obs, small_coins_collected, distance = self.get_observation()
    self.prev_vector_obs = obs
    self.prev_distance = distance
    self.min_distance = self.MAX_DISTANCE
    self.collected_coins = small_coins_collected
    self.position_history = []
    
    return ts.restart(obs)


  def _step(self, action):
    # pp.write_actions(action)
    self.interface.set_action(action, self.env_index)

    time.sleep(self.sleep_time)
    time_elapsed = time.time() - self.start_time

    obs, small_coins_collected, distance = self.get_observation()
    self.prev_vector_obs = obs
    
    latest_collected_coins = small_coins_collected

    mario_x = obs[self.INDEX_MARIO_X]
    mario_y = obs[self.INDEX_MARIO_Y]

    reward = self.calculate_reward(
      distance=distance,
      prev_distance=self.prev_distance,
      latest_collected_coins=latest_collected_coins,
      mario_position=(mario_x, mario_y))
    self.prev_distance = distance

    self.collected_coins = latest_collected_coins
    discount = ((self.game_duration - time_elapsed) / self.game_duration) + 0.1

    self.total_reward += reward * discount

    timestep = None
    if time_elapsed > self.game_duration:
      # print()
      # print(self.total_reward)
      timestep = ts.termination(obs, reward=reward)
    else:
      timestep = ts.transition(obs, reward=reward, discount=discount)        

    return timestep

  def small_coin_reward(self, distance):
    return 1

  def calculate_reward(self,
                       distance,
                       prev_distance,
                       latest_collected_coins,
                       mario_position):
    
    reward = 0
    diff = prev_distance - distance

    reward = diff * 100
    if diff < 0:
      reward *= 3

    for position in self.position_history:
      distance = self.get_distance(mario_position, position)
      if distance < 0.05:
        reward -= 1

    self.position_history.append(mario_position)
    if len(self.position_history) > self.MAX_POSITION_HISTORY:
      del self.position_history[0]
    
    collected_coin_diff = latest_collected_coins - self.collected_coins
    if collected_coin_diff > 0:
      self.min_distance = self.MAX_DISTANCE
      self.prev_distance = self.MAX_DISTANCE
      reward = 100

    return reward

  
  def get_distance(self, positionA, positionB):
    return (positionA[0] - positionB[0])**2 + (positionA[1] - positionB[1])**2


  def get_observation(self):
    obs_dict = self.interface.observations[self.env_index]

    obs = self.prev_vector_obs
    small_coins_collected = self.collected_coins
    distance = self.prev_distance
    if self.OBS_MARIO_ROTATION in obs_dict:
      distances = obs_dict[self.OBS_SMALL_COIN_DISTANCES]
      for index in range(len(distances)):
        coin_obs_index = self.INDEX_SMALL_COIN_DISTANCE + index
        obs[coin_obs_index] = distances[index]

      prev_pos_x = obs[self.INDEX_MARIO_X]
      prev_pos_y = obs[self.INDEX_MARIO_Y]
      for index in range(0, self.COUNT_PREV_MARIO_POS, 2):
        mario_pos_index_x = self.INDEX_PREV_MARIO_POSITIONS + index
        mario_pos_index_y = mario_pos_index_x + 1

        tmp_x = obs[mario_pos_index_x]
        tmp_y = obs[mario_pos_index_y]
        obs[mario_pos_index_x] = prev_pos_x
        obs[mario_pos_index_y] = prev_pos_y

        prev_pos_x = tmp_x
        prev_pos_y = tmp_y

      small_coins_collected = obs_dict[self.OBS_SMALL_COINS_COLLECTED]
      obs[self.INDEX_MARIO_X] = obs_dict[self.OBS_MARIO_POSITION][0]
      obs[self.INDEX_MARIO_Y] = obs_dict[self.OBS_MARIO_POSITION][1]
      obs[self.INDEX_MARIO_ROTATION] = obs_dict[self.OBS_MARIO_ROTATION]    

      distance = obs_dict[self.OBS_DISTANCE]

    return obs, small_coins_collected, distance