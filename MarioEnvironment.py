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
        shape=(1,), dtype=np.int32, minimum=0, maximum=3, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(100, 65, 3), dtype=np.float32, minimum=0, maximum=1, name='observation')
    self.OBS_DISTANCE = 'distance'
    self.start_time = time.time()
    self.sleep_time = 0.02
    self.game_duration = 8
    self.prev_vector_obs = [12]
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

    self.prev_vector_obs = [12]
    self.min_distance = 12

    obs = self.get_im_observation()

    return ts.restart(obs)

  def _step(self, action):
    one_hot_action = self.get_one_hot(action)[0]
    pp.write_action(one_hot_action)

    time.sleep(self.sleep_time)

    prev_distance = self.prev_vector_obs[0]
    vector_obs = self.get_observation()
    distance = vector_obs[0]
    
    self.prev_vector_obs = vector_obs

    did_collect = False
    if distance < 1:
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

    print(f'd: {distance}, p: {prev_distance}, reward: {reward}')

    discount = 1 # (self.game_duration - time_elapsed) / self.game_duration
    obs = self.get_im_observation()

    # print(f'distance: {distance}, minDist: {self.min_distance}, reward: {reward}')

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
      obs[0] = obs_dict[self.OBS_DISTANCE]

    return obs

  def get_im_observation(self):
    image_obs = pp.read_screenshot()
    obs = np.array(image_obs, dtype=np.float32)
    return obs

  def get_one_hot(self, target):
    res = np.eye(4)[target]
    return res

#   def init_prev_observations(self, image_obs):
#     if self.prev_img_obs0 is None:
#       self.prev_img_obs0 = image_obs
#       self.prev_img_obs1 = image_obs
#     elif self.prev_img_obs1 is None:
#       self.prev_img_obs1 = self.prev_img_obs0

#   def update_prev_observations(self, image_obs):
#     self.prev_img_obs1 = self.prev_img_obs0
#     self.prev_img_obs0 = image_obs
  
#   def create_stacked_obs(self, image_obs):
#     self.init_prev_observations(image_obs)

#     current_obs = np.array([image_obs], dtype=np.float32)
#     prev_obs0 = np.array([self.prev_img_obs0], dtype=np.float32)
#     prev_obs1 = np.array([self.prev_img_obs1], dtype=np.float32)

#     obs = np.concatenate((current_obs, prev_obs0, prev_obs1), axis=1)[0]

#     return obs

#   def stacked_image_obs(self):
#     image_obs = self.get_im_observation()
#     obs = self.create_stacked_obs(image_obs)
#     self.update_prev_observations(image_obs)
#     return obs