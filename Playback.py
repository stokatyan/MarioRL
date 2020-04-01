from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import matplotlib
import matplotlib.pyplot as plt

import MarioEnvironment
import SACAgent


physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 


tf_agent = SACAgent.create_agent()
tf_agent.initialize()

eval_env = SACAgent.eval_env
eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)

replay_buffer = SACAgent.create_replay_buffer(tf_agent)

checkpoint_dir = "checkpoint"
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=tf_agent,
    policy=tf_agent.policy,
    replay_buffer=replay_buffer,
    global_step=SACAgent.global_step
)

train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()

print('Running old model ...')

time_step = eval_env.reset()
while True:
  action_step = eval_policy.action(time_step)
  time_step = eval_env.step(action_step.action)
