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


physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 


train_py_env = MarioEnvironment.MarioEnvironment()
eval_py_env = MarioEnvironment.MarioEnvironment()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()
global_step = tf.compat.v1.train.get_or_create_global_step()


def normal_projection_net(action_spec,init_means_output_factor=0.1):
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=True)


def restore_agent(ckpt_dir="checkpoint"):
  tf_agent = create_agent()
  tf_agent.initialize()

  replay_buffer = create_replay_buffer(tf_agent)

  train_checkpointer = create_checkpointer(
      max_to_keep=1,
      agent=tf_agent,
      replay_buffer=replay_buffer,
      ckpt_dir="checkpoint"
  )

  train_checkpointer.initialize_or_restore()
  return tf_agent, train_checkpointer


def create_agent():
  critic_learning_rate = 3e-4 # @param {type:"number"}
  actor_learning_rate = 3e-4 # @param {type:"number"}
  alpha_learning_rate = 3e-4 # @param {type:"number"}
  target_update_tau = 0.005 # @param {type:"number"}
  target_update_period = 1 # @param {type:"number"}
  gamma = 0.99 # @param {type:"number"}
  reward_scale_factor = 1.0 # @param {type:"number"}
  gradient_clipping = None # @param

  actor_fc_layer_params = (50, 50)
  critic_joint_fc_layer_params = (50, 50)

  critic_net = critic_network.CriticNetwork(
    (observation_spec, action_spec),
    observation_fc_layer_params=None,
    action_fc_layer_params=None,
    joint_fc_layer_params=critic_joint_fc_layer_params)

  actor_net = actor_distribution_network.ActorDistributionNetwork(
    observation_spec,
    action_spec,
    fc_layer_params=actor_fc_layer_params,
    continuous_projection_net=normal_projection_net)

  
  tf_agent = sac_agent.SacAgent(
      train_env.time_step_spec(),
      action_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=actor_learning_rate),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=critic_learning_rate),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=alpha_learning_rate),
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      train_step_counter=global_step)
  
  return tf_agent


def create_replay_buffer(agent):
  replay_buffer_capacity = 10000 # @param {type:"integer"}

  return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=train_env.batch_size,
      max_length=replay_buffer_capacity)


def create_checkpointer(max_to_keep, agent, replay_buffer, ckpt_dir="checkpoint"):
  return common.Checkpointer(
    ckpt_dir=ckpt_dir,
    max_to_keep=max_to_keep,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
  )


def compute_avg_return(environment, policy, num_episodes=5):
  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


def train():
  num_iterations = 30000 # @param {type:"integer"}

  initial_collect_steps = 1000 # @param {type:"integer"} 
  collect_steps_per_iteration = 1 # @param {type:"integer"}

  batch_size = 150 # @param {type:"integer"}
  log_interval = 500 # @param {type:"integer"}

  num_eval_episodes = 5 # @param {type:"integer"}
  eval_interval = 5000 # @param {type:"integer"}

  tf_agent = create_agent()
  tf_agent.initialize()

  eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
  collect_policy = tf_agent.collect_policy

  replay_buffer = create_replay_buffer(tf_agent)

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

  train_checkpointer = create_checkpointer(
    max_to_keep=10, 
    agent=tf_agent, 
    replay_buffer=replay_buffer)

  train_checkpointer.initialize_or_restore()

  # TRAINING THE AGENT

  tf_agent.train = common.function(tf_agent.train)
  collect_driver.run = common.function(collect_driver.run)

  # Evaluate the agent's policy once before training.

  returns = []

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


if __name__ == "__main__":
    train()