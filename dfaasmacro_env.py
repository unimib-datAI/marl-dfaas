from load_generator import LoadGenerator, load_requests_traces

from RL4CC.environment import BaseMultiAgentEnvironment
from RL4CC.callbacks import BaseCallbacks

from dfaas_env import (
  _convert_arrival_rate_dist, 
  _distribute_rejects,
  _total_network_delay
)
import perfmodel

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.env.env_context import EnvContext
from gymnasium.spaces import Dict, Box
from copy import deepcopy
import networkx as nx
import pandas as pd
import numpy as np


class DFaaSMacroEnvironment(BaseMultiAgentEnvironment):
  
  def load_configuration(self, env_config: EnvContext) -> int:
    """
    Initialize environment loading info from the provided configuration dict
    """
    # time management and list of agents
    seed = super().load_configuration(env_config)
    # evaluation environment (?)
    self.evaluation_env = env_config.get("is_evaluation", False)
    # workload configuration
    env_limits = env_config.pop("limits", {"load": {}})
    self.trace_type = env_limits["load"].pop("trace_type", "fixed_sum")
    self.requests_trace = None
    if self.evaluation_env:
      reqt = load_requests_traces(
        env_limits["load"].pop("load_existing")
      )[0]
      self.requests_trace = {
        f: {
          f"node_{a}": v for a,v in function_trace.items()
        } for f, function_trace in reqt.items()
      }
    self.workload_limits = {
      "0": {
        agent: {"min": 0, "max": 1200} for agent in self.agents
      }
    }
    # -- load trace generator
    self.LG = LoadGenerator(average_requests = 100, amplitude_requests = 50)
    # reward weights
    env_instance = env_config.pop("instance")[None]
    self.alpha = env_instance["alpha"]
    self.beta = env_instance["beta"]
    self.gamma = env_instance["gamma"]
    # maximum utilization
    self.max_utilization = env_instance["max_utilization"]
    # network configuration
    def load_dfaasconfig(env_config):
      from dfaas_env_config import DFaaSConfig
      return DFaaSConfig.from_dict(env_config)
    config = load_dfaasconfig(env_config)
    config.build_config()
    self.network = config._network
    for u in config.network_links:
      for v in config.network_links[u]:
        self.network[u][v]["access_delay_ms"] = config.network_links[u][v][
          "access_delay_ms"
        ]
        self.network[u][v]["bandwidth_mbps"] = config.network_links[u][v][
          "bandwidth_mbps"
        ]
    # Freeze to prevent further modification.
    self.network = nx.freeze(self.network)
    # agents neighbors
    self.agent_neighbors = {
      agent: list(self.network.neighbors(agent)) for agent in self.agents
    }
    # response time threshold
    self.response_time_threshold = config.response_time_threshold
    # maximum number of replicas
    self.max_n_replicas = {
      agent: config.perfmodel_params[agent]["maximum_concurrency"] \
        for agent in self.agents
    }
    # warm/cold service time
    self.service_times = {
      agent: {
        "warm": config.perfmodel_params[agent]["warm_service_time"],
        "cold": config.perfmodel_params[agent]["cold_service_time"]
      } for agent in self.agents
    }
    # idle time before kill
    self.idle_time_before_kill = {
      agent: config.perfmodel_params[agent]["idle_time_before_kill"] \
        for agent in self.agents
    }
    # i/o data
    self.data_size = {
      "input": config.request_input_data_size_bytes,
      "output": config.request_output_data_size_bytes
    }
    return seed
  
  def define_observation_spaces(self):
    """
    Define the environment observation space(s)
    """
    self.observation_space = Dict({
      agent: Dict({
        # input rate (of the current and previous step)
        "input_rate": Box(
          low = int(self.workload_limits["0"][agent]["min"]),
          high = int(self.workload_limits["0"][agent]["max"]),
          shape=(1,),
          dtype = np.int32
        ),
        "previous_input_rate": Box(
          low = int(self.workload_limits["0"][agent]["min"]),
          high = int(self.workload_limits["0"][agent]["max"]),
          shape=(1,),
          dtype = np.int32
        ),
        # rate of requests forwarded to all neighbors in the previous step 
        # (and those that the neighbor rejected)
        **{
          f"previous_fwd_to_{neighbor}": Box(
            low = 0,
            high = int(self.workload_limits["0"][agent]["max"]),
            shape=(1,),
            dtype = np.int32
          ) for neighbor in self.agent_neighbors[agent]
        },
        **{
          f"previous_fwd_to_{neighbor}_rejected": Box(
            low = 0,
            high = int(self.workload_limits["0"][agent]["max"]),
            shape=(1,),
            dtype = np.int32
          ) for neighbor in self.agent_neighbors[agent]
        },
        # # CPU utilization
        # "previous_cpu_utilization": Box(
        #   low = 0.0, high = 1.0, shape=(1,), dtype = np.float32
        # )
      }) for agent in self.agents
    })
    # initialize a random dummy observation to return at the end
    self._dummy_terminal_obs = self.observation_space.sample()
  
  def define_action_spaces(self):
    """
    Define the environment action space(s)
    ---
    proportion of enqueued, fowarded (to each neighbor) and rejected requests
    """
    self.action_space = Dict({
      agent: Simplex(
        shape = (1 + len(self.agent_neighbors[agent]) + 1,)
      ) for agent in self.agents
    })
  
  def init_agent_metrics(self):
    self.info = {}
    for agent in self.agents:
      avg_workload = (
        self.workload_limits["0"][agent]["max"] - \
          self.workload_limits["0"][agent]["min"]
      ) // 2
      # -- input rate and local processing latency
      self.info[agent] = {
        "input_rate": self.workload_limits["0"][agent]["min"] + avg_workload,
        "cpu_utilization": 0.0,
        # -- action
        "action": [],
        "loc": 0,
        "fwd": 0,
        "total_fwd": 0,
        "rej": 0
      }
      # -- forward and rejections
      for neighbor in self.agent_neighbors[agent]:
        self.info[agent][f"fwd_to_{neighbor}"] = 0
        self.info[agent][f"fwd_to_{neighbor}_rejected"] = 0

  def observation(self):
    """
    Return the next observation (and the corresponding info dictionary)
    """
    # define observation
    obs = {agent: {} for agent in self.agents}
    obs_info = {agent: {} for agent in self.agents}
    for agent in self.agents:
      # -- input rate
      obs[agent]["input_rate"] = np.array(
        [self.requests_trace[0][agent][self.current_time]], dtype = np.int32
      )
      obs_info[agent]["input_rate"] = int(
        self.requests_trace[0][agent][self.current_time]
      )
      # -- previous input rate
      obs[agent]["previous_input_rate"] = np.array(
        [self.info[agent]["input_rate"]], dtype = np.int32
      )
      # # -- previous cpu utilization
      # obs[agent]["previous_cpu_utilization"] = np.array(
      #   [self.info[agent]["cpu_utilization"]], dtype = np.float32
      # )
      # -- previous forward to neighbors
      for neighbor in self.agent_neighbors[agent]:
        obs[agent][f"previous_fwd_to_{neighbor}"] = np.array(
          [self.info[agent][f"fwd_to_{neighbor}"]], dtype = np.int32
        )
        obs[agent][f"previous_fwd_to_{neighbor}_rejected"] = np.array(
          [self.info[agent][f"fwd_to_{neighbor}_rejected"]], dtype = np.int32
        )
    obs_info["__common__"] = {
      "current_time": self.current_time
    }
    # update info
    old_info = deepcopy(self.info)
    for agent, agent_info in old_info.items():
      for key, val in agent_info.items():
        if key in obs_info[agent]:
          self.info[agent][f"previous_{key}"] = val
          self.info[agent][key] = obs_info[agent][key]
    self.info["__common__"] = obs_info["__common__"]
    return obs, self.info
  
  def reset(self, seed: int = None, options = None):
    # initialize info dictionary
    self.init_agent_metrics()
    # random number generator
    self.rng = np.random.default_rng(seed = seed)
    # pick the next load trace
    if not self.evaluation_env:
      self.requests_trace = {}
      wlm = {
        0: {
          0: 72.563,
          1: 72.563,
          2: 72.563,
          3: 72.563
        }
      }
      for function, function_limits in wlm.items():
        new_trace_type = self.trace_type
        if self.trace_type == "fixed_sum_minmax" and function%2 == 0: # min for even
          new_trace_type = "fixed_sum_min"
        elif self.trace_type == "fixed_sum_minmax" and function%2 != 0: # max for odd
          new_trace_type = "fixed_sum_max"
        reqt = self.LG.generate_traces(
          max_steps = self.max_time, 
          limits = function_limits,
          rng = self.rng,
          trace_type = new_trace_type #f"manual{function}"#
        )
        self.requests_trace[function] = {
          f"node_{a}": reqt[a] for a in reqt
        }
    # restart time
    self.current_time = self.min_time
    # define observation
    obs, obs_info = self.observation()
    return obs, obs_info
  
  def simulate_action(self, action_dict):
    # tot_incoming_rate: total incoming requests for each agent (enqueued 
    # requests + those sent by neighbors)
    tot_incoming_requests = {agent: [] for agent in action_dict}
    senders = {agent: [] for agent in action_dict}
    tot_rejects = {agent: 0 for agent in action_dict}
    for agent, action_dist in action_dict.items():
      # -- convert the action proportions into actual number of requests
      action = _convert_arrival_rate_dist(
        self.info[agent]["input_rate"], action_dist
      )
      self.info[agent]["action"] = action_dist
      self.info[agent]["loc"] = action[0]
      self.info[agent]["fwd"] = action[1:-1]
      self.info[agent]["total_fwd"] = sum(action[1:-1])
      self.info[agent]["rej"] = action[-1]
      # -- local processing
      tot_incoming_requests[agent].append(action[0])
      # -- forward
      for neigh_idx, neighbor in enumerate(self.agent_neighbors[agent]):
        tot_incoming_requests[neighbor].append(action[1 + neigh_idx])
        senders[neighbor].append(agent)
        self.info[agent][f"fwd_to_{neighbor}"] = action[1 + neigh_idx]
      # -- rejections (by action)
      tot_rejects[agent] += action[-1]
    # simulate
    for agent in tot_incoming_requests:
      incoming_rate_total = sum(tot_incoming_requests[agent])
      result_props, _ = perfmodel.get_sls_warm_count_dist(
        incoming_rate_total,
        self.service_times[agent]["warm"],
        self.service_times[agent]["cold"],
        self.idle_time_before_kill[agent],
        maximum_concurrency = self.max_n_replicas[agent],
        faster_solution = (self.max_n_replicas[agent] <= 30)
      )
      # cpu utilization and number of replicas
      self.info[agent]["cpu_utilization"] = float(
        result_props["avg_utilization"]
      )
      self.info[agent]["n_replicas"] = int(
        result_props["avg_running_count"]
      )
      # distribute the rejection rate to the agent and all its neighbors and 
      # compute the local / offloading latency
      rejection_rate = result_props["rejection_rate"]
      rejects = _distribute_rejects(
        rejection_rate, 
        tot_incoming_requests[agent]
      )
      avg_resp_time = result_props.get("avg_resp_time", 0.0)
      tot_rejects[agent] += rejects[0]
      self.info[agent]["local_rejected"] = tot_rejects[agent]
      self.info[agent]["avg_resp_time_loc"] = avg_resp_time
      if tot_incoming_requests[agent][0] <= 0.0:
        self.info[agent]["reject_rate"] = 0.0
      else:
        self.info[agent]["reject_rate"] = float(
          rejects[0] / tot_incoming_requests[agent][0]
        )
      # -- neighbors
      for sender, reject in zip(senders[agent], rejects[1:]):
        tot_rejects[sender] += reject
        self.info[sender][f"fwd_to_{agent}_rejected"] = reject
        self.info[sender][
          f"avg_resp_time_fwd_to_{agent}"
        ] = float(
          avg_resp_time + _total_network_delay(
            self.network[sender][agent]["access_delay_ms"],
            self.data_size["input"],
            self.network[sender][agent]["bandwidth_mbps"][self.current_time]
          ) + _total_network_delay(
            self.network[agent][sender]["access_delay_ms"],
            self.data_size["output"],
            self.network[agent][sender]["bandwidth_mbps"][self.current_time]
          )
        )
  
  def step(self, action_dict):
    """
    Applies the action chosen by each agent, moves to the next state and 
    computes the reward
    ---
    Returns a tuple containing:
      1) new observations for each ready agent, 
      2) reward values for each ready agent. If the episode is just started, 
      the value will be None. 
      3) Terminated values for each ready agent. The special key “__all__” 
      (required) is used to indicate env termination. 
      4) Truncated values for each ready agent. 
      5) Info values for each agent id (may be empty dicts)
    """
    # apply action
    self.simulate_action(action_dict)
    # compute reward
    reward = self.compute_reward()
    # update time
    self.current_time += self.time_step
    # check if we are in the last step of the episod should be truncated
    done = {
      agent: self.current_time >= self.max_time for agent in self.agents
    }
    done["__all__"] = all(done.values())
    truncated = done
    # define observation
    obs = None
    obs_info = {}
    if self.current_time < self.max_time:
      obs, obs_info = self.observation()
    else:
      # -- ignore the last step
      obs = self._dummy_terminal_obs
    return obs, reward, done, truncated, obs_info

  def compute_reward(self):
    reward = {}
    # for each agent, compare response time with threshold and compute utility
    for idx, agent in enumerate(self.agents):
      # -- local processing
      loc_utility = 0.0
      if self.info[agent]["loc"] > 0:
        rt_loc = self.info[agent]["avg_resp_time_loc"]
        rej_loc = self.info[agent]["local_rejected"] - self.info[agent]["rej"]
        if rej_loc <= 0:
          loc_utility = self.alpha[(idx+1,1)] * self.info[agent]["action"][0]
        else:
          loc_utility = self.alpha[(idx+1,1)] * (
            self.info[agent]["action"][0] - rej_loc / self.info[agent]["loc"]
          )
      # -- forward
      fwd_utility = 0.0
      for neighbor_idx, neighbor in enumerate(self.agent_neighbors[agent]):
        if self.info[agent]["fwd"][neighbor_idx] > 0:
          rt_fwd = self.info[agent][f"avg_resp_time_fwd_to_{neighbor}"]
          rej_fwd = self.info[agent][f"fwd_to_{neighbor}_rejected"]
          if rej_fwd <= 0.0:
            fwd_utility += self.beta[(idx+1,neighbor_idx+1,1)] * self.info[agent]["action"][neighbor_idx + 1]
          else:
            fwd_utility += self.beta[(idx+1,neighbor_idx+1,1)] * (
              self.info[agent]["action"][neighbor_idx + 1] - rej_fwd / self.info[
                agent
              ]["fwd"][neighbor_idx]
            )
      # -- reject
      rej_penalty = -self.gamma[(idx+1,1)] * self.info[agent]["action"][-1]
      reward[agent] = (
        float(loc_utility) + float(fwd_utility) + float(rej_penalty)
      )
    return reward


class DFaaSMacroCallbacks(BaseCallbacks):
  """
  User defined callbacks for the DFaaS environment.

  These callbacks can be used with other environments, both multi-agent and
  single-agent.

  See the Ray's API documentation for DefaultCallbacks, the custom class
  overrides (and uses) only a subset of callbacks and keyword arguments.
  """
  def on_episode_start(
      self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
    """
    Callback run right after an episode has started.
    Only the episode and base_env keyword arguments are used, other
    arguments are ignored.
    """
    try:
      env = base_env.envs[0]
    except AttributeError:
      # With single-agent environment the wrapper env is an instance of
      # VectorEnvWrapper and it doesn't have envs attribute. With
      # multi-agent the wrapper is MultiAgentEnvWrapper.
      env = base_env.get_sub_environments()[0]
    self.RELEVANT_KEYS = set()
    for agent in env.agents + ["__common__"]:
      for key in env.info[agent]:
        self.RELEVANT_KEYS.add(key)
    super().on_episode_start(
      worker = worker, 
      base_env = base_env, 
      policies = policies, 
      episode = episode, 
      env_index = env_index, 
      **kwargs
    )
  
  def on_episode_end(
      self, *, worker, base_env, policies, episode, env_index, **kwargs,
    ):
    try:
      env = base_env.envs[0]
      for agent in env.agents:
        for key in self.RELEVANT_KEYS:
          if f"{key}_{agent}" in episode.user_data:
            _ = episode.hist_data.pop(f"{key}_{agent}")
            if len(episode.user_data[f"{key}_{agent}"]) > 0:
              episode.hist_data[f"{key}-{agent}"] = episode.user_data[
                f"{key}_{agent}"
              ][:-1]
              episode.custom_metrics[
                f"{key}-{agent}_avg"
              ] = np.mean(episode.user_data[f"{key}_{agent}"][:-1])
    except AttributeError:
      for key in self.RELEVANT_KEYS:
        episode.hist_data[key] = episode.user_data[key]
        episode.custom_metrics[f"{key}_avg"] = np.mean(episode.user_data[key])
    # add worker index
    episode.hist_data["worker_index"] = episode.user_data["worker_index"]
