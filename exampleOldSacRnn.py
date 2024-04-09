# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Train and Eval RNN SAC.

To run:

```bash
tensorboard --logdir $HOME/tmp/sac_rnn/dm/CartPole-Balance/ --port 2223 &

python tf_agents/agents/sac/examples/v2:train_eval_rnn --\
  --root_dir=$HOME/tmp/sac_rnn/dm/CartPole-Balance/ \
  --alsologtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_dm_control
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

import MarioEnvironment

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS


@gin.configurable
def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
  del init_action_stddev
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=True)


_DEFAULT_REWARD_SCALE = 0


@gin.configurable
def train_eval(
    root_dir,
    env_name='cartpole',
    task_name='balance',
    observations_whitelist='position',
    eval_env_name=None,
    num_iterations=1000000,
    # Params for networks.
    actor_fc_layers=(400, 300),
    actor_output_fc_layers=(100,),
    actor_lstm_size=(40,),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(300,),
    critic_output_fc_layers=(100,),
    critic_lstm_size=(40,),
    num_parallel_environments=1,
    # Params for collect
    initial_collect_episodes=1,
    collect_episodes_per_iteration=1,
    replay_buffer_capacity=1000000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=256,
    train_sequence_length=20,
    critic_learning_rate=3e-4,
    actor_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    td_errors_loss_fn=tf.math.squared_difference,
    gamma=0.99,
    reward_scale_factor=_DEFAULT_REWARD_SCALE,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=30,
    eval_interval=10000,
    # Params for summaries and logging
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=50000,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):
  """A simple train and eval for RNN SAC on DM control."""
  root_dir = 'checkpoint'

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):

    train_py_env = MarioEnvironment.MarioEnvironment()
    eval_py_env = MarioEnvironment.MarioEnvironment()
    eval_py_env.reset_type = 2

    tf_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        input_fc_layer_params=actor_fc_layers,
        lstm_size=actor_lstm_size,
        output_fc_layer_params=actor_output_fc_layers,
        continuous_projection_net=normal_projection_net)

    critic_net = critic_rnn_network.CriticRnnNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        lstm_size=critic_lstm_size,
        output_fc_layer_params=critic_output_fc_layers)

    tf_agent = sac_agent.SacAgent(
        time_step_spec,
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
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()

    # Make the replay buffer.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size*num_parallel_environments,
        max_length=replay_buffer_capacity)

    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())
    collect_policy = tf_agent.collect_policy

    train_checkpointer = common.Checkpointer(
        ckpt_dir=root_dir,
        agent=tf_agent,
        global_step=global_step)

    train_checkpointer.initialize_or_restore()

    initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=initial_collect_episodes)

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=collect_episodes_per_iteration)

    # if use_tf_functions:
    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    initial_collect_driver.run()

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=train_sequence_length + 1).prefetch(3)
    iterator = iter(dataset)

    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)

    train_step = common.function(train_step)

    for iteration_count in range(num_iterations):
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state,
        )
        print(f'Training iteration: {iteration_count}...')
        train_step()
        print('Done Training.')

        global_step_val = global_step.numpy()
        if global_step_val % train_checkpoint_interval == 0:
          train_checkpointer.save(global_step=global_step_val)


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  train_eval(None)


if __name__ == '__main__':
  app.run(main)