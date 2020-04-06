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

    self.COUNT_SMALL_COIN_DISTANCES = 19
    min_distance = [0] * self.COUNT_SMALL_COIN_DISTANCES
    max_distance = [12] * self.COUNT_SMALL_COIN_DISTANCES

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

    self.MAX_DISTANCE = 12
    self.START_GAME_DURATION = 6
    self.BONUS_GAME_DURATION = 2

    self.start_time = time.time()
    self.sleep_time = 0.2
    self.game_duration = 6
    self.prev_vector_obs = np.array([0] * 26, dtype=np.float32)
    self.prev_vector_obs[self.INDEX_DISTANCE] = self.MAX_DISTANCE
    self.prev_distance = 12
    self.min_distance = 12
    self.collected_coins = 0

    self.reset_type = 1


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
    self.game_duration = self.START_GAME_DURATION
    pp.write_gameover(self.reset_type)
    time.sleep(self.sleep_time)

    self.prev_vector_obs = np.array([0] * 26, dtype=np.float32)
    self.prev_vector_obs[self.INDEX_DISTANCE] = self.MAX_DISTANCE

    obs = list(self.get_observation())
    self.prev_distance = obs.pop(0)
    self.min_distance = self.prev_distance
    self.collected_coins = obs.pop(0)
    
    obs = np.array(obs, dtype=np.float32)
    return ts.restart(obs)


  def _step(self, action):
    pp.write_action(action)
    time.sleep(self.sleep_time)
    time_elapsed = time.time() - self.start_time

    scd_start = self.INDEX_SMALL_COIN_DISTANCE
    scd_end = scd_start + self.COUNT_SMALL_COIN_DISTANCES

    prev_distance = self.prev_vector_obs[self.INDEX_DISTANCE]
    prev_small_coin_distances = list(self.prev_vector_obs)[scd_start:scd_end]

    obs = list(self.get_observation())
    self.prev_vector_obs = np.array(obs, dtype=np.float32)
    
    small_coin_distances = obs[scd_start:scd_end]
    distance = obs.pop(0)
    latest_collected_coins = obs.pop(0)
    
    did_collect = False
    if distance < 0.95:
      did_collect = True

    reward = self.calculate_reward(
      distance=distance, 
      prev_distance=prev_distance, 
      latest_collected_coins=latest_collected_coins, 
      small_coin_distances=small_coin_distances,
      prev_small_coin_distances=prev_small_coin_distances)

    self.collected_coins = latest_collected_coins
    discount = (self.game_duration - time_elapsed*0.8) / self.game_duration

    timestep = None
    obs = np.array(obs, dtype=np.float32)
    if time_elapsed > self.game_duration or did_collect:
      self.reset()
      timestep = ts.termination(obs, reward=reward)
    else:
      timestep = ts.transition(obs, reward=reward, discount=discount)        

    return timestep


  def calculate_reward(self, 
                        distance, 
                        prev_distance, 
                        latest_collected_coins, 
                        small_coin_distances, 
                        prev_small_coin_distances):
    reward = 0

    if distance < 0.85:
      reward = 2500

    reward += 5 * (prev_distance - distance)

    for index in range(len(small_coin_distances) - 1):
      scd = small_coin_distances[index]
      p_scd = prev_small_coin_distances[index]
      reward += 5 * (p_scd - scd)

    collected_coin_diff = latest_collected_coins - self.collected_coins
    if collected_coin_diff > 0:
      # Collecting a small coin resets the timer
      self.game_duration += self.BONUS_GAME_DURATION 
      reward += collected_coin_diff * 1000

    return reward


  def get_observation(self):
    obs_dict = pp.read_observation()
    obs = self.prev_vector_obs

    if self.OBS_DISTANCE in obs_dict:
      obs[self.INDEX_DISTANCE] = obs_dict[self.OBS_DISTANCE]
      obs[self.INDEX_SMALL_COINS_COLLECTED] = obs_dict[self.OBS_SMALL_COINS_COLLECTED]
      obs[self.INDEX_MARIO_X] = obs_dict[self.OBS_MARIO_POSITION][0]
      obs[self.INDEX_MARIO_Y] = obs_dict[self.OBS_MARIO_POSITION][1]
      obs[self.INDEX_MARIO_ROTATION] = obs_dict[self.OBS_MARIO_ROTATION]
      obs[self.INDEX_COIN_X] = obs_dict[self.OBS_COIN_POSITION][0]
      obs[self.INDEX_COIN_Y] = obs_dict[self.OBS_COIN_POSITION][1]
      
      distances = obs_dict[self.OBS_SMALL_COIN_DISTANCES]
      for index in range(len(distances)):
        coin_obs_index = self.INDEX_SMALL_COIN_DISTANCE + index
        obs[coin_obs_index] = distances[index]

    return obs