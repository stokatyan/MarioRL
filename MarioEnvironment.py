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

import ReadWrite as rw
import time

class MarioEnvironment(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(60, 200, 1), dtype=np.float32, minimum=0, maximum=1, name='observation')
    self.ATT_VELOCITY = 'velocity'
    self.ATT_TOPDISTANCE = 'topDistance'
    self.ATT_BOTTOMDISTANCE = 'bottomDistance'
    self.start_time = time.time()
    self.sleep_time = 0.02
    self.game_duration = 8
    self.last_observation = [0, 1, 1]
    self.prev_img_obs0 = None
    self.prev_img_obs1 = None


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
    rw.write_reset_game()
    time.sleep(self.sleep_time)

    obs = self.stacked_image_obs()

    return ts.restart(obs)

  def _step(self, action):
    rw.write_thrust(action[0])

    time.sleep(self.sleep_time)
    observation = self.get_observation()
    min_distance = min(observation[1], observation[2])

    did_collide = False
    if min_distance < 0.055:
      did_collide = True

    timestep = None
    time_elapsed = time.time() - self.start_time
    reward = 200
    discount = 1 # (self.game_duration - time_elapsed) / self.game_duration

    obs = self.stacked_image_obs()

    if time_elapsed > self.game_duration or did_collide:
      self.reset()
      timestep = ts.termination(obs, reward=reward)
    else:
      timestep = ts.transition(obs, reward=reward, discount=discount)        

    return timestep

  def get_observation(self):
    obs_dict = rw.read_observation()
    obs = self.last_observation

    if self.ATT_VELOCITY in obs_dict:
        obs[0] = obs_dict[self.ATT_VELOCITY]
    if self.ATT_TOPDISTANCE in obs_dict:
        obs[1] = obs_dict[self.ATT_TOPDISTANCE]
    if self.ATT_BOTTOMDISTANCE in obs_dict:
        obs[2] = obs_dict[self.ATT_BOTTOMDISTANCE]

    self.last_observation = obs
    return self.last_observation

  def get_im_observation(self):
    im_obs = rw.read_screenshot()
    return im_obs

  def init_prev_observations(self, image_obs):
    if self.prev_img_obs0 is None:
      self.prev_img_obs0 = image_obs
      self.prev_img_obs1 = image_obs
    elif self.prev_img_obs1 is None:
      self.prev_img_obs1 = self.prev_img_obs0

  def update_prev_observations(self, image_obs):
    self.prev_img_obs1 = self.prev_img_obs0
    self.prev_img_obs0 = image_obs
  
  def create_stacked_obs(self, image_obs):
    self.init_prev_observations(image_obs)

    current_obs = np.array([image_obs], dtype=np.float32)
    prev_obs0 = np.array([self.prev_img_obs0], dtype=np.float32)
    prev_obs1 = np.array([self.prev_img_obs1], dtype=np.float32)

    obs = np.concatenate((current_obs, prev_obs0, prev_obs1), axis=1)[0]

    return obs

  def stacked_image_obs(self):
    image_obs = self.get_im_observation()
    obs = self.create_stacked_obs(image_obs)
    self.update_prev_observations(image_obs)
    return obs