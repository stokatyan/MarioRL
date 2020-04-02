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
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(5,), dtype=np.float32, 
        minimum=[0.0, -4.5, -4.5, -4.5, -4.5], 
        maximum=[12.0, 4.5, 4.5, 4.5, 4.5], 
        name='observation')
    self.OBS_DISTANCE = 'distance'
    self.OBS_MARIO_POSITION = 'marioPosition'
    self.OBS_COIN_POSITION = 'coinPosition'

    self.INDEX_DISTANCE = 0
    self.INDEX_MARIO_X = 1
    self.INDEX_MARIO_Y = 2
    self.INDEX_COIN_X = 3
    self.INDEX_COIN_Y = 4

    self.start_time = time.time()
    self.sleep_time = 0.2
    self.game_duration = 8
    self.prev_vector_obs = np.array([12, 0, 0, 0, 0], dtype=np.float32)
    self.min_distance = 12


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

    self.prev_vector_obs = np.array([12, 0, 0, 0, 0], dtype=np.float32)
    self.min_distance = 12

    obs = self.get_observation()

    return ts.restart(obs)

  def _step(self, action):
    # one_hot_action = self.get_one_hot(action)[0]
    pp.write_action(action)

    time.sleep(self.sleep_time)

    prev_distance = self.prev_vector_obs[0]
    obs = self.get_observation()
    distance = obs[0]
    self.prev_vector_obs = obs

    did_collect = False
    if distance < 0.9:
      did_collect = True

    timestep = None
    time_elapsed = time.time() - self.start_time
    reward = 0

    if distance < self.min_distance:
        reward = 100
        self.min_distance = distance
    elif distance < prev_distance:
        reward += 20
    else:
      reward -= 30

    discount = (self.game_duration - time_elapsed) / self.game_duration
    
    if time_elapsed > self.game_duration:
      self.reset()
      timestep = ts.termination(obs, reward=reward)
    elif did_collect:
      self.reset()
      timestep = ts.termination(obs, reward=500)
    else:
      timestep = ts.transition(obs, reward=reward, discount=discount)        

    return timestep


  def get_observation(self):
    obs_dict = pp.read_observation()
    obs = self.prev_vector_obs

    if self.OBS_DISTANCE in obs_dict:
      obs[self.INDEX_DISTANCE] = obs_dict[self.OBS_DISTANCE]
      obs[self.INDEX_MARIO_X] = obs_dict[self.OBS_MARIO_POSITION][0]
      obs[self.INDEX_MARIO_Y] = obs_dict[self.OBS_MARIO_POSITION][1]
      obs[self.INDEX_COIN_X] = obs_dict[self.OBS_COIN_POSITION][0]
      obs[self.INDEX_COIN_Y] = obs_dict[self.OBS_COIN_POSITION][1]

    return obs


  def get_one_hot(self, target):
    res = np.eye(4)[target]
    return res