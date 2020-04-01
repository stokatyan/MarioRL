from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tf_agents.policies import greedy_policy

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


tf_agent, checkpointer = SACAgent.restore_agent(ckpt_dir="checkpoint")

eval_env = SACAgent.eval_env
eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)

print(f'Running agent from restored checkpoint: {checkpointer._manager.latest_checkpoint} ...')

time_step = eval_env.reset()
while True:
  action_step = eval_policy.action(time_step)
  time_step = eval_env.step(action_step.action)
