import numpy as np
from cost_functions import trajectory_cost_fn
import time


class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        self.env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.env.action_space.sample()


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ YOUR CODE HERE
                Note: be careful to batch your simulations through the model for speed """

        observations = np.empty(
            (self.num_simulated_paths, self.horizon, self.env.observation_space.shape[0]))
        next_observations = np.empty(
            (self.num_simulated_paths, self.horizon, self.env.observation_space.shape[0]))

        actions = [
            [self.env.action_space.sample()
             for _ in range(self.horizon)]
            for _ in range(self.num_simulated_paths)
        ]
        actions = np.array(actions)

        last_state = np.array([state for _ in range(self.num_simulated_paths)])
        for idx in range(self.horizon):
            action_batch = actions[:, idx]
            next_state = self.dyn_model.predict(last_state, action_batch)
            observations[:, idx, :] = last_state
            next_observations[:, idx, :] = next_state
            last_state = next_state

        costs = np.array([trajectory_cost_fn(
            self.cost_fn, observations[i], actions[i],
            next_observations[i])
            for i in range(self.num_simulated_paths)
        ])

        min_cost_path_id = np.argmin(costs)
        return actions[min_cost_path_id][0]
