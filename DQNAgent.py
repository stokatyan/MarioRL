from __future__ import absolute_import, division, print_function

import numpy as np
import os

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import matplotlib
import matplotlib.pyplot as plt

import time
from random import randrange

import MarioEnvironment


physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 


num_iterations = 150000 # @param {type:"integer"}

initial_collect_steps = 10000 # @param {type:"integer"} 
collect_steps_per_iteration = 5 # @param {type:"integer"}
replay_buffer_capacity = 50000 # @param {type:"integer"}

batch_size = 1000 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}
gradient_clipping = None # @param

fc_layer_params = (50, 50)

log_interval = 500 # @param {type:"integer"}
learning_rate = 1e-3

num_eval_episodes = 5 # @param {type:"integer"}
eval_interval = 2000 # @param {type:"integer"}

train_py_env = MarioEnvironment.MarioEnvironment()
eval_py_env = MarioEnvironment.MarioEnvironment()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()

# q_net = q_network.QNetwork(
    # observation_spec,
    # action_spec,
#     fc_layer_params=fc_layer_params)
q_net = q_rnn_network.QRnnNetwork(
    observation_spec,
    action_spec,
)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)


global_step = tf.compat.v1.train.get_or_create_global_step()

tf_agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    action_spec,
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step)

tf_agent.initialize()


eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
collect_policy = tf_agent.collect_policy


def compute_avg_return(environment, policy, num_episodes=5):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    policy_state = policy.get_initial_state(environment.batch_size)

    episode_return = 0.0

    while not time_step.is_last():
      action, policy_state, info = policy.action(time_step, policy_state)
      time_step = environment.step(action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

print('\nCollecting Initial Steps ...')
initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps)
initial_collect_driver.run()

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

iterator = iter(dataset)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)


checkpoint_dir = "checkpoint"
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=tf_agent,
    policy=tf_agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)

train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()

# TRAINING THE AGENT

tf_agent.train = common.function(tf_agent.train)
collect_driver.run = common.function(collect_driver.run)

# Evaluate the agent's policy once before training.

returns = []


def train():
  print('Computing Initial Average Return ...')
  avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
  print(f'Initial Average return: {avg_return}')
  returns.append(avg_return)

  print('\nTraining ...\n')
  for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
      collect_driver.run()

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = tf_agent.train(experience)

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
      print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
      print(f'Saving at step: {step} ...')
      train_checkpointer.save(global_step=global_step)
      print('Evaluating ...')
      avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
      print('step = {0}: Average Return = {1}'.format(step, avg_return))
      returns.append(avg_return)

train()

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()
plt.show()

print('Running old model ...')

time_step = eval_env.reset()
while True:
  action_step = eval_policy.action(time_step)
  time_step = eval_env.step(action_step.action)
