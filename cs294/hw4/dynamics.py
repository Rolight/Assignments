import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network


def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out


class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.env = env
        self.batch_size = batch_size
        self.sess = sess
        self.normalization = normalization
        self.iter = iterations

        self.observation_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=[None, env.observation_space.shape[0]]
        )
        self.action_placeholder = tf.placeholder(
            dtype=tf.float32, shape=[None, env.action_space.shape[0]])
        self.label_placeholder = tf.placeholder(
            dtype=tf.float32, shape=[None, env.observation_space.shape[0]])

        eps = tf.constant(1e-7)
        mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action = self.normalization
        observation_norm = (self.observation_placeholder -
                            mean_obs) / (std_obs + eps)
        action_norm = (self.action_placeholder -
                       mean_action) / (std_action + eps)
        inputs = tf.concat([observation_norm, action_norm], axis=1)

        self.delta_predict = build_mlp(
            input_placeholder=inputs,
            output_size=env.observation_space.shape[0],
            scope="nndym",
            n_layers=n_layers,
            size=size,
            activation=activation,
            output_activation=output_activation
        )

        self.delta_predict = self.delta_predict * std_deltas + mean_deltas
        self.loss = tf.reduce_mean(
            tf.square(self.delta_predict - self.label_placeholder))
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        observations = np.concatenate([p['observations'] for p in data])
        next_observations = np.concatenate(
            [p['next_observations'] for p in data])
        actions = np.concatenate([p['actions'] for p in data])

        deltas = next_observations - observations

        train_indicies = np.arange(observations.shape[0])
        for iter_step in range(self.iter):
            np.random.shuffle(train_indicies)
            for i in range(len(train_indicies) // self.batch_size):
                start_id = i * self.batch_size
                train_ids = train_indicies[start_id: start_id+self.batch_size]
                obs_batch = observations[train_ids]
                action_batch = actions[train_ids]
                deltas_batch = deltas[train_ids]
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                    self.observation_placeholder: obs_batch,
                    self.action_placeholder: action_batch,
                    self.label_placeholder: deltas_batch
                })
            print('on iter %d, loss = %.7f' % (iter_step, loss))

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        delta = self.sess.run(self.delta_predict, feed_dict={
            self.observation_placeholder: states,
            self.action_placeholder: actions,
        })
        return states + delta
