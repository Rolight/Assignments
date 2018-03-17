import tensorflow as tf
import numpy as np
import gym

import load_policy
import tf_util
from BCModel import BCModel


class BCModelWithDAgger(BCModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_data = None
        self.ouput_data = None
        self.expert_policy_file = './experts/%s.pkl' % self.env_name
        self.policy_fn = load_policy.load_policy(self.expert_policy_file)

    def DAgger(self, train_inputs, train_outputs, max_steps=1000, rollouts=10):
        if not self.env:
            self.env = gym.make(self.env_name)
        dagger_obs, dagger_actions = [], []
        with tf.Session():
            tf_util.initialize()
            for _ in range(rollouts):
                obs = self.env.reset()
                done = False
                steps = 0
                while not done and steps < max_steps:
                    action = self.get_action(obs)
                    obs, _, done, _ = self.env.step(action)
                    dagger_obs.append(obs)
                    dagger_actions.append(self.policy_fn(obs[None, :]))

        dagger_actions = np.array(dagger_actions)
        dagger_actions = np.reshape(
            dagger_actions, (dagger_actions.shape[0], dagger_actions.shape[2]))
        train_inputs = np.concatenate((train_inputs, dagger_obs))
        train_outputs = np.concatenate((train_outputs, dagger_actions))
        return train_inputs, train_outputs
