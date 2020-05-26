from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import parallel_py_environment

import matplotlib
import matplotlib.pyplot as plt

import MarioEnvironment
import PyPipeline as pp
import tqdm
import time


physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 

constructors = [
  MarioEnvironment.MarioEnvZero, 
  MarioEnvironment.MarioEnvOne,
  MarioEnvironment.MarioEnvTwo,
  MarioEnvironment.MarioEnvThree,
  MarioEnvironment.MarioEnvFour,
  MarioEnvironment.MarioEnvFive,
  MarioEnvironment.MarioEnvSix,
  MarioEnvironment.MarioEnvSeven,
  MarioEnvironment.MarioEnvEight,
  MarioEnvironment.MarioEnvNine,
  ]

eval_constructors = [
  MarioEnvironment.MarioEnvZero, 
  MarioEnvironment.MarioEnvOne,
  MarioEnvironment.MarioEnvTwo,
  MarioEnvironment.MarioEnvThree,
  MarioEnvironment.MarioEnvFour,
]

train_py_env = parallel_py_environment.ParallelPyEnvironment(constructors, start_serially=True, blocking=False)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

eval_py_env = parallel_py_environment.ParallelPyEnvironment(eval_constructors, start_serially=True, blocking=False)
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

  input_fc_layer_params = (200, 100)
  lstm_size = (40,)
  output_fc_layer_params = (200, 100)
  joint_fc_layer_params = (200, 100)
  

  actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
    observation_spec,
    action_spec,
    input_fc_layer_params=input_fc_layer_params,
    lstm_size=lstm_size,
    output_fc_layer_params=output_fc_layer_params,
    continuous_projection_net=normal_projection_net)

  critic_net = critic_rnn_network.CriticRnnNetwork(
    (observation_spec, action_spec),
    observation_fc_layer_params=input_fc_layer_params,
    lstm_size=lstm_size,
    output_fc_layer_params=output_fc_layer_params,
    joint_fc_layer_params=joint_fc_layer_params)
  
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
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      train_step_counter=global_step)
  
  return tf_agent


def create_replay_buffer(agent):
  replay_buffer_capacity = 100000 # @param {type:"integer"}

  return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=train_env.batch_size,
      max_length=replay_buffer_capacity,
      dataset_window_shift=1)


def create_checkpointer(max_to_keep, agent, replay_buffer, ckpt_dir="checkpoint"):
  return common.Checkpointer(
    ckpt_dir=ckpt_dir,
    max_to_keep=max_to_keep,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
  )


def compute_avg_return(environment, policy, num_episodes):
  total_return = 0.0
  for _ in range(num_episodes):
    pp.write_gameover(2)
    time.sleep(0.5)
    time_step = environment.reset()
    policy_state = policy.get_initial_state(environment.batch_size)
    episode_return = 0.0

    while not environment._envs[0]._current_time_step.is_last():
      action_step, policy_state, _ = policy.action(time_step, policy_state)
      time_step = environment.step(action_step)
      for i in range(len(time_step.reward)):
        episode_return += time_step.reward[i] * time_step.discount[i]
    total_return += episode_return

  avg_return = total_return / num_episodes
  avg_return = avg_return / len(environment._envs)
  return avg_return.numpy()


def train():
  num_iterations = 10000 # @param {type:"integer"}
  train_steps_per_iteration = 1
  collect_episodes_per_iteration = 1
  initial_collect_episodes = 1

  batch_size = 18000 # @param {type:"integer"}
  max_train_size = 3000
  train_splits = batch_size / max_train_size

  num_eval_episodes = 4 # @param {type:"integer"}
  eval_interval = 100 # @param {type:"integer"}
  train_sequence_length = 55

  tf_agent = create_agent()
  tf_agent.initialize()

  replay_buffer = create_replay_buffer(tf_agent)

  eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
  initial_collect_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec())
  collect_policy = tf_agent.collect_policy

  train_checkpointer = create_checkpointer(
    max_to_keep=60, 
    agent=tf_agent, 
    replay_buffer=replay_buffer)

  train_checkpointer.initialize_or_restore()
    
  initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      train_env,
      initial_collect_policy,
      observers=[replay_buffer.add_batch],
      num_episodes=1)

  collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      train_env,
      collect_policy,
      observers=[replay_buffer.add_batch],
      num_episodes=collect_episodes_per_iteration)

  tf_agent.train = common.function(tf_agent.train)
  collect_driver.run = common.function(collect_driver.run)
  initial_collect_driver.run = common.function(initial_collect_driver.run)

  print('\nCollecting Initial Steps ...')
  for _ in range(10):
    for _ in range(initial_collect_episodes):
      initial_collect_driver.run(time_step=None)

  # Prepare replay buffer as dataset with invalid transitions filtered.
  def _filter_invalid_transition(trajectories, unused_arg1):
    # Reduce filter_fn over full trajectory sampled. The sequence is kept only
    # if all elements except for the last one pass the filter. This is to
    # allow training on terminal steps.
    return tf.reduce_all(~trajectories.is_boundary()[:-1])

  # Dataset generates trajectories with shape [Bx2x...]
  filtered_dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=train_sequence_length+1).unbatch().filter(
            _filter_invalid_transition).batch(batch_size).prefetch(5)
  filtered_iterator = iter(filtered_dataset)

  # Evaluate the agent's policy once before training.

  returns = [0]

  print('\nTraining ...\n')

  def train_step():
    filtered_exp, _ = next(filtered_iterator)

    for index in range(0, int(train_splits)):
      print(f'sub train step: {index}/{int(train_splits)} ...    ', end="\r", flush=True)
      split_start = index * max_train_size
      split_end = split_start + max_train_size
      traj = trajectory.Trajectory(
        step_type=filtered_exp.step_type[split_start:split_end],
        observation=filtered_exp.observation[split_start:split_end],
        action=filtered_exp.action[split_start:split_end],
        next_step_type=filtered_exp.next_step_type[split_start:split_end],
        reward=filtered_exp.reward[split_start:split_end],
        discount=filtered_exp.discount[split_start:split_end],
        policy_info = filtered_exp.policy_info[split_start:split_end]
      )
      _ = tf_agent.train(traj)
  

  train_step = common.function(train_step)

  for iteration_count in range(num_iterations):

    progress = (iteration_count % eval_interval) + 1

    print(f'progress: {progress}/{eval_interval} ...    ', end="\r", flush=True)

    time_step = None
    pp.write_gameover(1)
    time.sleep(0.5)
    policy_state = collect_policy.get_initial_state(train_env.batch_size)
    collect_driver.run(
        time_step=time_step,
        policy_state=policy_state,
    )

    pp.write_no_action()
    
    for index in range(train_steps_per_iteration):
      print(f'training: {index}/{train_steps_per_iteration} ...    ', end="\r", flush=True)
      train_step()

    step = tf_agent.train_step_counter.numpy()

    if (iteration_count + 1) % eval_interval == 0:
      print(f'Saving at step: {step} ...')
      train_checkpointer.save(global_step=global_step)
      print(f'Evaluating iteration: {iteration_count + 1}')
      avg_return = compute_avg_return(train_env, eval_policy, num_eval_episodes)
      print('step = {0}: Average Return = {1}'.format(step, avg_return))
      returns.append(avg_return)


  steps = range(0, num_iterations + 1, eval_interval)
  plt.plot(steps, returns)
  plt.ylabel('Average Return')
  plt.xlabel('Step')
  plt.ylim()
  plt.show()

  print('Running latest model ...')

  # time_step = train_env.reset()
  # policy_state = eval_policy.get_initial_state(train_env.batch_size)
  # while True:
  #   action_step, policy_state, _ = eval_policy.action(time_step, policy_state)
  #   time_step = train_env.step(action_step)


if __name__ == "__main__":
    train()