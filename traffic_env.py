import math

import gymnasium as gym
import numpy as np

from ray.rllib.utils.spaces import simplex
from ray.tune.registry import register_env

from RL4CC.environment.base_environment import BaseEnvironment

def TrafficManagementEnvCreator(env_config):
    env = gym.wrappers.TimeLimit(TrafficManagementEnv(env_config),
                                 max_episode_steps=100)

    return env

register_env("TrafficManagementEnv", TrafficManagementEnvCreator)

class TrafficManagementEnv(BaseEnvironment):
    def __init__(self, config):
        """
        State:
            - Nr. of input requests (integer),
            - Queue capacit (integer),
            - Forward capacity (integer),
            - Congested cause 1: queue is full? (boolean),
            - Congested cause 2: forward capacity has been exceeded? (boolean).

        config is a dictionary with the env's configuration. If a property is
        not set, the default value is used.
        """
        self.action_space = simplex.Simplex(shape=(3,))
        self.observation_space = gym.spaces.Box(low=np.array([50, 0, 0, 0, 0], dtype=np.float32),
                                                high=np.array([150, 100, 100, 1, 1], dtype=np.float32),
                                                dtype=np.float32)

        # Minimium and maximum possible reward. Overwrites parent attribute.
        self.reward_range = (-np.inf, np.inf)

        # Constants.
        self.max_cpu_capacity = config.get("cpu_capacity", 1000)
        self.max_queue_capacity = config.get("queue_capacity", 100)
        self.max_memory_capacity = config.get("memory_capacity", 8000)
        self.max_forward_capacity = config.get("forward_capacity", 100)

        # Parameters used to generate input requests and forward capacity at
        # each time step.
        self.average_requests = config.get("average_requests", 100)
        self.amplitude_requests = config.get("amplitude_requests", 50)
        self.period = config.get("period", 50)
        self.scenario = config.get("scenario", "scenario1")
        self.seed = config.get("seed", 0)

        # Max steps for the environment.
        self.max_steps = config.get("max_steps", 100)

        # Call 'print' function on each step.
        self.debug = config.get("debug", False)

        # Create the RNG used to generate input requests and forward capacity.
        self.rng = np.random.default_rng(seed=self.seed)

        # Reset the environment.
        self.reset(seed=self.seed)

    def reset(self, *, seed=None, options=None):

        # Congested flags.
        self.congested_queue_full = False
        self.congested_forward_exceed = False

        # Steps in congested state.
        self.congested_steps = 0

        # Number of forwarded requests exceeding the forwarding capacity.
        self.forward_exceed = 0

        # The capacity of the queue workload for a single time step. The queue
        # is empty at the start.
        self.queue_workload = []
        self.queue_capacity = self.max_queue_capacity

        # Counters.
        self.total_managed_requests = 0
        self.total_rejected_requests = 0

        # Current step (from 0 to self.max_steps).
        self.current_step = 0

        # Seed.
        if seed is not None:
            self.seed = seed

        # Recreate the RNG from start.
        self.rng = np.random.default_rng(seed=self.seed)

        # Set initial input requests and forward capacity.
        self.input_requests, self.forward_capacity = self._get_requests_capacity()

        # Initial observation and info.
        obs = self._observation()
        info = self._additional_info()

        return obs, info

    def _observation(self):
        """Builds and returns the observation for the current state.

        For more information see gymnasium.Env.step.
        """
        obs = np.array([self.input_requests,
                        self.queue_capacity,
                        self.forward_capacity,
                        self.congested_queue_full,
                        self.congested_forward_exceed])

        return obs

    def _additional_info(self, actions=(), reward=0):
        """Builds and returns the info dictionary for the current step.

        For more information see gymnasium.Env.step.
        """
        info = {}
        # These two keys are required by BaseCallback (from RL4CC).
        info["current_time"] = self.current_step
        info["reward"] = reward

        info["congested"] = self.congested_queue_full or self.congested_forward_exceed

        info["actions"] = {}
        info["actions"]["local"] = actions[0] if len(actions) == 3 else 0
        info["actions"]["forwarded"] = actions[1] if len(actions) == 3 else 0
        info["actions"]["rejected"] = actions[2] if len(actions) == 3 else 0

        return info

    def step(self, action):
        # Apply the action (a distribution of probabilities) to the input
        # requests and obtain the distribution of requests (number of requests
        # to process locally, forward or reject).
        local, forwarded, rejected = self.process_actions(action)
        self.total_managed_requests += local + forwarded + rejected

        if self.debug:
            print(f'State of the system at time step {self.current_step} of {self.max_steps}')
            print(f'Current status (before applying action):')
            print(f'  Step: {self.current_step}')
            print(f'  Congested? {self.congested_queue_full or self.congested_forward_exceed}')
            print(f'  Queue capacity: {self.queue_capacity}')
            print(f'  Input requests: {self.input_requests}')
            print(f'  Forward capacity: {self.forward_capacity}')
            print(f'  Steps in congestion / not in congestion: {self.congested_steps} / {self.current_step - self.congested_steps}')
 
        # In some cases, the agent will try to forward more requests than are
        # available. We need to track this so that it can be transformed into a
        # negative reward for the agent and to enable the congestion state at
        # the node.
        #
        # The 'max' function is necessary because negative values mean that the
        # forward capacity will not be exceeded.
        self.forward_exceed = max(0, forwarded - self.forward_capacity)

        # Process requests locally, building a CPU workload that contains the
        # requests processed by the node in this time step. The method also
        # updates the queue workload and returns the number of rejected requests
        # that cannot be processed locally (both by the CPU workload because
        # there is no CPU and RAM available, and by the queue workload because
        # it is full).
        cpu_workload, new_rejected = self._manage_workload(local)
        rejected += new_rejected
        local -= new_rejected

        self.total_rejected_requests += rejected

        # Calculate reward for the agent.
        reward = self._calculate_reward(local, forwarded, rejected)

        # Update the observation_space. Note that cpu_workload is not used.
        terminated = self._update_observation_space()

        obs = self._observation()
        info = self._additional_info(actions=(local, forwarded, rejected), reward=reward)
        truncated = False

        if self.debug:
            print(f'Action chosen:')
            print(f'  Local requests processed:', local)
            print(f'  Forwarded requests:', forwarded)
            print(f'  Rejected requests:', rejected)
            print(f'Reward: {reward}')
            print(f'Current status (after applying action):')
            print(f'  Congested? {self.congested_queue_full or self.congested_forward_exceed}')
            print(f'  Queue capacity: {self.queue_capacity}')
            print(f'  Forward exceed: {self.forward_exceed}')

        return obs, reward, terminated, truncated, info

    def process_actions(self, action_distribution):
        '''process_actions returns the number of requests processed for each
        action (local processing, forward or reject), distributing the current
        input requests according to the given actions distribution.'''
        # Extract the three actions from the action distribution
        prob_local, prob_forwarded, prob_rejected = action_distribution

        # Get the corresponding number of requests for each action. Note: the
        # number of requests is a discrete number, so there is a fraction of the
        # action probabilities that is left out of the calculation.
        actions = [
                int(prob_local * self.input_requests), # local requests
                int(prob_forwarded * self.input_requests), # forwarded requests
                int(prob_rejected * self.input_requests)] # rejected requests
        processed_requests = sum(actions)

        # There is a fraction of unprocessed input requests. We need to fix this
        # problem by assigning the remaining requests to the higher fraction for
        # the three action probabilities, because that action is the one that
        # loses the most.
        if processed_requests < self.input_requests:
            # Extract the fraction for each action probability.
            fractions = [prob_local * self.input_requests - actions[0],
                         prob_forwarded * self.input_requests - actions[1],
                         prob_rejected * self.input_requests - actions[2]]

            # Get the highest fraction index and and assign remaining requests
            # to that action.
            max_fraction_index = np.argmax(fractions)
            actions[max_fraction_index] += self.input_requests - processed_requests

        assert sum(actions) == self.input_requests
        return tuple(actions)

    def _calculate_reward(self, local, forwarded, rejected):
        '''It returns the agent's reward for the current state of the
        environment, using the given number of local, forwarded and rejected
        requests.'''
        # The queue factor represents how much queue capacity is available for
        # local processing. The range is 0 to 1. A low value means the queue is
        # almost full, while a high value means the queue is almost empty. The
        # same applies to the forward factor.
        queue_factor = self.queue_capacity / self.max_queue_capacity
        forward_factor = self.forward_capacity / self.max_forward_capacity

        # The reward is the sum of three different sub-rewards, one for each
        # action taken by the agent. The state of congestion affects how the
        # reward is calculated. Each sub-reward has a constant multiplicative
        # coefficient.
        if not self.congested_queue_full and not self.congested_forward_exceed:
            # If the queue is not congested, local processing is prioritised in
            # a way that is proportional to the capacity of the queue (the queue
            # factor). So if the queue is close to full, local processing is
            # discouraged.
            reward_local = 3 * local * queue_factor
            reward_forwarded = 1 * forwarded * (1 - queue_factor) * forward_factor
            reward_rejected = -10 * rejected * forward_factor * queue_factor

            reward = reward_local + reward_forwarded + reward_rejected - 2 * self.forward_exceed
        else:
            # If congested, local processing is discouraged, while forwarding is
            # slightly encouraged only if available, then rejection is
            # encouraged (if forwarding is not possible). Note that
            # over-forwarding is dangerous in this case.
            reward_local = -10 * local
            reward_forwarded = -2 * forwarded * forward_factor
            reward_rejected = 2 * rejected * (1 - forward_factor)

            reward = reward_local + reward_forwarded + reward_rejected - 500 - 2 * self.forward_exceed

        return reward

    def sample_workload(self, requests):
        '''sample_workload returns a sample of the workload for the given number
        of requests. Each sample is a dictionary with the following keys:

            - 'class': the class of the request ('A', 'B' or 'C'),
            - 'cpu': CPU shares required to process the request,
            - 'memory': the RAM memory in MB required for the elaboration,
            - 'position': the position of the request in the list.'''

        # Each class has his own parameters for the normal distributions. The
        # standard deviation is fixed to 2.5.
        classes = {'A': {'cpu_mean': 5.5,  'cpu_min': 1,  'cpu_max': 10,
                         'ram_mean': 12.5, 'ram_min': 1,  'ram_max': 25},
                   'B': {'cpu_mean': 15,   'cpu_min': 11, 'cpu_max': 20,
                         'ram_mean': 38,   'ram_min': 26, 'ram_max': 50},
                   'C': {'cpu_mean': 25.5, 'cpu_min': 21, 'cpu_max': 30,
                         'ram_mean': 63,   'ram_min': 51, 'ram_max': 75}}
        std_dev = 2.5

        workload = []
        for i in range(requests):
            # The class is selected from a uniform distribution, while CPU usage
            # and memory are selected from two normal distributions.
            sample = self.rng.uniform()
            if sample < 0.33:
                req = 'A'
            elif sample < 0.67:
                req = 'B'
            else:
                req = 'C'

            # 'class' key.
            load = {'class': req}

            # 'cpu' key.
            shares = self.rng.normal(classes[req]['cpu_mean'], std_dev)
            load['cpu'] = np.clip(shares, classes[req]['cpu_min'], classes[req]['cpu_max'])

            # 'memory' key.
            memory = self.rng.normal(classes[req]['ram_mean'], std_dev)
            load['memory'] = np.clip(memory, classes[req]['ram_min'], classes[req]['ram_max'])

            # 'position' key.
            load['position'] = i

            workload.append(load)

        return workload

    def _manage_workload(self, local):
        '''It returns the CPU workload and the number of rejected requests. It
        also updates the queue workload.

        The CPU workload is a list of requests representing the requests
        processed by the node in a single time step. First the requests already
        in the queue workload are processed, then new requests are sampled to be
        handled locally (number provided by the 'local' argument) and they are
        moved to the CPU workload (if CPU and RAM is available) or to the queue
        workload, or they are rejected (if the queue is full).'''
        # List of requests to be processed in the CPU, the size depends on the
        # current CPU and RAM capacity.
        cpu_workload = []

        # Current CPU and RAM capacity. We assume that the CPU and RAM are fully
        # available at each time step.
        cpu_capacity = self.max_cpu_capacity
        ram_capacity = self.max_memory_capacity

        # First, requests already in the queue are processed. Requests will be
        # moved from queue workload to CPU workload if CPU and RAM shares are
        # sufficient.
        for request in self.queue_workload.copy():
            if cpu_capacity >= request['cpu'] and ram_capacity >= request['memory']:
                cpu_capacity -= request['cpu']
                ram_capacity -= request['memory']

                cpu_workload.append(request)
                self.queue_workload.remove(request)

        # Second, get a sample of the new workload generated and then process
        # these requests:
        #
        #   1. If there is sufficient CPU and RAM available, move the request to
        #   the CPU workload.
        #   2. Otherwise, move the request to the queued workload.
        #   3. If the queue workload is full (has reached its maximum capacity),
        #   the request will be rejected.
        rejected_requests = 0
        for request in self.sample_workload(local):
            if cpu_capacity >= request['cpu'] and ram_capacity >= request['memory']:
                cpu_capacity -= request['cpu']
                ram_capacity -= request['memory']

                cpu_workload.append(request)
            elif len(self.queue_workload) < self.max_queue_capacity:
                self.queue_workload.append(request)
            else:
                rejected_requests += 1
 
        return cpu_workload, rejected_requests

    def _get_requests_capacity(self):
        # Update input requests and forward capacity for the next time step.
        match self.scenario:
            # RANDOM - RANDOM scenario
            case 'scenario1':
                average_capacity = 50
                std_dev = 10
                input_requests = int(self.rng.normal(self.average_requests, std_dev))
                forward_capacity = int(self.rng.normal(average_capacity, std_dev))

            # SINUSOIDE NOISY - SINUSOIDE NOISY scenario
            case 'scenario2':
                average_capacity = 50
                amplitude_capacity = 50

                noise_ratio = .1
                base_input = self.average_requests + self.amplitude_requests * math.sin(2 * math.pi * self.current_step / self.period)
                noisy_input = base_input + noise_ratio * self.rng.normal(0, self.amplitude_requests)
                input_requests = int(noisy_input)

                base_capacity = average_capacity + amplitude_capacity * math.sin(2 * math.pi * self.current_step / self.period)
                noisy_capacity = base_capacity + noise_ratio * self.rng.normal(0, amplitude_capacity)
                forward_capacity = int(noisy_capacity)

            # SINUSOIDE - SINUSOIDE scenario
            case 'scenario3':
                input_requests = int(self.average_requests + self.amplitude_requests * math.sin(2 * math.pi * self.current_step / self.period))
                forward_capacity = int(25 + 75 * (1 + math.sin(2 * math.pi * self.current_step / self.period)) / 2)

            case _:
                assert False, f"Unreachable code with scenario {self.scenario!r}"

        return input_requests, forward_capacity


    def _update_observation_space(self):
        """It updates the observation space for the next step."""
        # Update input requests and forward capacity.
        self.input_requests, self.forward_capacity = self._get_requests_capacity()

        # Update queue capacity.
        self.queue_capacity = self.max_queue_capacity - len(self.queue_workload)
        assert self.queue_capacity >= 0, "Queue capacity can't be negative!"

        # If the queue capacity is zero, it means that the queue workload is
        # full, no more requests can be queued locally: there is congestion at
        # the node.
        if self.queue_capacity == 0:
            self.congested_queue_full = 1
        else:
            self.congested_queue_full = 0

        # If the forward exceed is greater than zero, it means that there are
        # more requests being forwarded than the forward capacity: there is
        # congestion at the node.
        if self.forward_exceed > 0:
            self.congested_forward_exceed = 1
        else:
            self.congested_forward_exceed = 0

        # Count the number of steps in congested state.
        if self.congested_queue_full or self.congested_forward_exceed:
            self.congested_steps += 1

        # Go to the next step until termination.
        self.current_step += 1
        if self.current_step == self.max_steps:
            done = True
        else:
            done = False

        return done


