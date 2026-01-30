from RL4CC.environment import BaseMultiAgentEnvironment

from dfaas_env import (
  _convert_arrival_rate_dist, 
  _distribute_rejects,
  _total_network_delay
)
import perfmodel

from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.env.env_context import EnvContext
from gymnasium.spaces import Dict, Box
import pandas as pd
import numpy as np


class DFaaSMetricsEnvironment(BaseMultiAgentEnvironment):
  
  def load_configuration(self, env_config: EnvContext) -> int:
    """
    Initialize environment loading info from the provided configuration dict
    """
    # time management and list of agents
    seed = super().load_configuration(env_config)
    # load datasets
    self.joined_metrics = pd.read_csv(env_config["joined_metrics"])
    self.joined_metrics_avg = pd.read_csv(env_config["joined_metrics_avg"])
    if self.max_time > self.joined_metrics_avg["cp_bucket"].max():
      max_cp_bucket = self.joined_metrics_avg["cp_bucket"].max()
      raise ValueError(
        f"max_time must be <= max_cp_bucket ({self.max_time}>{max_cp_bucket})"
      )
    # workload limits
    self.workload_limits = env_config["limits"]
    # agents neighbors
    self.agent_neighbors = env_config["neighborhood"]
    # maximum number of replicas
    self.max_n_replicas = env_config["max_n_replicas"]
    # warm/cold service time
    self.service_times = env_config["service_times"]
    # idle time before kill
    self.idle_time_before_kill = env_config["idle_time_before_kill"]
    # response time threshold
    self.response_time_threshold = env_config["response_time_threshold"]
    # i/o data
    self.data_size = env_config["data_size"]
    # evaluation environment (?)
    self.evaluation_env = env_config.get("evaluation", False)
    return seed
  
  def define_observation_spaces(self):
    """
    Define the environment observation space(s)
    """
    self.observation_space = Dict({
      agent: Dict({
        # input rate (of the current and previous step)
        "input_rate": Box(
          low = self.workload_limits["0"][agent]["min"],
          high = self.workload_limits["0"][agent]["max"],
          dtype = np.int32
        ),
        "previous_input_rate": Box(
          low = self.workload_limits["0"][agent]["min"],
          high = self.workload_limits["0"][agent]["max"],
          dtype = np.int32
        ),
        # rate of requests forwarded to all neighbors in the previous step 
        # (and those that the neighbor rejected)
        **{
          f"previous_fwd_to_{neighbor}": Box(
            low = self.workload_limits["0"][agent]["min"],
            high = self.workload_limits["0"][agent]["max"],
            dtype = np.int32
          ) for neighbor in self.agent_neighbors[agent]
        },
        **{
          f"previous_fwd_to_{neighbor}_rejected": Box(
            low = self.workload_limits["0"][agent]["min"],
            high = self.workload_limits["0"][agent]["max"],
            dtype = np.int32
          ) for neighbor in self.agent_neighbors[agent]
        },
        # average latency of enqueued / forwarded requests
        "previous_avg_resp_time_loc": Box(
          low = 0.0, high = self.response_time_threshold, dtype = np.float32
        ),
        **{
          f"previous_avg_resp_time_fwd_to_{neighbor}": Box(
            low = 0.0, high = self.response_time_threshold, dtype = np.float32
          ) for neighbor in self.agent_neighbors[agent]
        },
        # CPU utilization
        "cpu_utilization": Box(low = 0.0, high = 1.0, dtype = np.float32),
        # number of replicas
        "n_replicas": Box(
          low = 0, high = self.max_n_replicas[agent], dtype = np.int32
        )
      }) for agent in self.agents
    })
  
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
      self.info[agent] = {
        "input_rate": 0
      }
      for neighbor in self.agent_neighbors[agent]:
        self.info[agent][f"fwd_to_{neighbor}"] = 0.0
        self.info[agent][f"fwd_to_{neighbor}_rejected"] = 0.0

  def observation(self):
    """
    Return the next observation (and the corresponding info dictionary)
    """
    # extract metrics for the current control period
    cp_metrics = pd.DataFrame()
    sample_idx = None
    if self.evaluation_env:
      cp_metrics = self.current_metrics_avg[
        self.current_metrics_avg["cp_bucket"] == self.current_time
      ]
    else:
      cp_metrics = self.current_metrics[
        self.current_metrics["cp_bucket"] == self.current_time
      ]
      sample_idx = self.rng.integers(low = 0, high = len(cp_metrics))
      cp_metrics = cp_metrics.iloc[sample_idx]
    # define observation
    obs = {}
    obs_info = {}
    for agent in self.agents:
      # -- input rate
      obs[agent]["input_rate"] = np.array(
        cp_metrics["http_reqs"].values(), dtype = np.int32
      )
      obs_info[agent]["input_rate"] = cp_metrics["http_reqs"].values()
      # -- previous input rate
      obs[agent]["previous_input_rate"] = np.array(
        self.info[agent]["input_rate"], dtype = np.int32
      )
      # -- previous average latency
      obs[agent]["previous_avg_resp_time_loc"] = np.array(
        self.info[agent]["avg_resp_time_loc"], dtype = np.int32
      )
      # -- previous forward to neighbors and forward latency
      for neighbor in self.agent_neighbors[agent]:
        obs[agent][f"previous_fwd_to_{neighbor}"] = np.array(
          self.info[agent][f"fwd_to_{neighbor}"], dtype = np.float32
        )
        obs[agent][f"previous_fwd_to_{neighbor}_rejected"] = np.array(
          self.info[agent][f"fwd_to_{neighbor}_rejected"], dtype = np.float32
        )
        obs[agent][f"previous_avg_resp_time_fwd_to_{neighbor}"] = np.array(
          self.info[agent][f"avg_resp_time_fwd_to_{neighbor}"], 
          dtype = np.float32
        )
    obs_info["__common__"] = {
      "current_time": self.current_time,
      "trace_idx": self.current_metrics["trace_idx"].iloc[0],
      "sample_idx": sample_idx
    }
    # update info
    self.info = obs_info
    return obs, obs_info
  
  def reset(self, seed: int = None, options = None):
    # initialize info dictionary
    self.init_agent_metrics()
    # random number generator
    self.rng = np.random.default_rng(seed = seed)
    # extract the next load trace index
    trace_idx = self.rng.choice(self.joined_metrics["trace_idx"].unique())
    # -- prepare subset of current metrics
    self.current_metrics = self.joined_metrics[
      self.joined_metrics["trace_idx"] == trace_idx
    ]
    self.current_metrics_avg = self.current_metrics_avg[
      self.current_metrics_avg["trace_idx"] == trace_idx
    ]
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
      self.info[agent]["fwd"] = sum(action[1:-1])
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
      # -- neighbors
      for sender, reject in zip(senders, rejects[1:]):
        tot_rejects[sender] += reject
        self.info[sender][f"fwd_to_{agent}_rejected"] = reject
        self.info[sender][
          f"avg_resp_time_fwd_to_{agent}"
        ] = avg_resp_time + _total_network_delay(
          self.network[sender][agent]["access_delay_ms"],
          self.data_size["input"],
          self.network[agent][neighbor]["bandwidth_mbps"][self.current_time]
        ) + _total_network_delay(
          self.network[sender][agent]["access_delay_ms"],
          self.data_size["output"],
          self.network[agent][neighbor]["bandwidth_mbps"][self.current_time]
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
    obs, obs_info = self.observation()
    return obs, reward, done, truncated, obs_info

  def compute_reward(self):
    reward = {}
    # for each agent, compare response time with threshold and compute utility
    for agent in self.agents:
      # -- local processing
      loc_utility = 0.0
      rt_loc = self.info["avg_resp_time_loc"][agent]
      rej_loc = self.info[agent]["local_rejected"] - self.info[agent]["rej"]
      if rt_loc < self.response_time_threshold and rej_loc <= 0:
        loc_utility = 1.0 * self.info[agent]["action"][0]
      elif rt_loc < self.response_time_threshold and rej_loc > 0:
        loc_utility = 1.0 * (
          self.info[agent]["action"][0] - rej_loc / self.info[agent]["loc"]
        )
      else:
        loc_utility = -0.75 * self.info[agent]["action"][0]
      # -- forward
      fwd_utility = 0.0
      for neighbor_idx, neighbor in enumerate(self.agent_neighbors[agent]):
        rt_fwd = self.info[agent][f"avg_resp_time_fwd_to_{neighbor}"]
        rej_fwd = self.info[agent][f"fwd_to_{neighbor}_rejected"]
        if rt_fwd < self.response_time_threshold and rej_fwd <= 0.0:
          fwd_utility += 1.0 * self.info[agent]["action"][neighbor_idx + 1]
        elif rt_fwd < self.response_time_threshold and rej_fwd > 0.0:
          fwd_utility += 1.0 * (
            self.info[agent]["action"][neighbor_idx + 1] - rej_fwd / self.info[
              agent
            ]["fwd"][neighbor_idx + 1]
          )
        else:
          fwd_utility += -1.0 * self.info[agent]["action"][neighbor_idx + 1]
      # -- reject
      rej_penalty = -0.5 * self.info[agent]["action"][-1]
      reward[agent] = (
        float(loc_utility) + float(fwd_utility) + float(rej_penalty)
      )
    return reward
  
