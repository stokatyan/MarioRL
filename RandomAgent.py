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

import time
import PyPipeline as pp
from random import randrange
import random


physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 


tf.compat.v1.enable_v2_behavior()

sleep_time = 0.1

count = 0

while count < 5:
  print('New Game')
  pp.write_gameover()
  time.sleep(sleep_time)
  start_time = time.time()
  end_time = time.time()

  while end_time - start_time < 10:
    action = [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]
    pp.write_action(action)
    
    time.sleep(sleep_time)
    end_time = time.time()

  count += 1

pp.write_gameover()