import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class TrafficManagementCallbacks(DefaultCallbacks):
    """Custom callbacks for the TrafficManagementEnv training.

    See API documentation for DefaultCallbacks, I use only a subset of the
    callbacks and keyword arguments."""

    def __init__(self):
        self.keys = ["current_time",
                     "reward",
                     "reward_components",
                     "congested",
                     "congested_queue_full",
                     "congested_forwarding_exceed",
                     "rejected_reqs",
                     "input_requests",
                     "queue_capacity",
                     "forwarding_capacity",
                     "actions"
                     ]

    def on_episode_start(self, *, episode, base_env, **kwargs):
        """Callback run right after an Episode has started.

        Only the episode and base_env keyword argument are used, other arguments
        will be ignored."""
        # Make sure this episode has just been started (only initial obs logged
        # so far).
        assert episode.length <= 0, "'on_episode_start()' callback should be called right after env reset!"

        # See API documentation for EpisodeV2 class.
        for key in self.keys:
            episode.user_data[key] = []
            episode.hist_data[key] = []

        # The way to get the first observation/additional information is
        # complicated because of the Ray API. However, we need to save the first
        # observation because it contains the initial data. The other values
        # (actions...) will be ignored.
        #
        # Note: the same API is for multi-agent, so we get a list and have only
        # the first environment.
        info = base_env.get_sub_environments()[0].unwrapped._additional_info()

        self._save_data(episode, info)

    def on_episode_step(self, *, episode, **kwargs):
        """Called on each episode step (after the action has been logged).

        Only the episode keyword argument is used, other arguments will be
        ignored."""
        # Make sure this episode is ongoing.
        assert episode.length > 0, "'on_episode_step()' callback should not be called right after env reset!"

        info = episode.last_info_for()

        self._save_data(episode, info)

    def on_episode_end(self, *, episode, **kwargs):
        """Called when an episode is done (after terminated/truncated have been
        logged).

        Only the episode keyword argument is used, other arguments will be
        ignored."""
        for key in self.keys:
            episode.hist_data[key] = episode.user_data[key]

        # I need to calculate the percentage of rejected requests per step in
        # relation to the input requests. The lists have to be shifted because
        # the action is done on the previous observation, so we have an initial
        # zero action and a last input requests to remove.
        rejected_reqs = episode.hist_data["rejected_reqs"][1:]
        input_requests = episode.hist_data["input_requests"][:-1]
        rejected_reqs_percent = []
        for idx in range(len(rejected_reqs)):
            rejected_reqs_percent.append(rejected_reqs[idx] / input_requests[idx] * 100)

        # Save the list of percent in hist_data. Note that this list is aligned
        # for each step (starting from 1).
        episode.hist_data["rejected_reqs_percent"] = rejected_reqs_percent

        rejected_reqs_total_percent = sum(rejected_reqs) / sum(input_requests) * 100

        episode.custom_metrics["reward_mean"] = np.mean(episode.hist_data["reward"])
        episode.custom_metrics["congested_steps"] = np.sum(episode.hist_data["congested"])
        episode.custom_metrics["rejected_reqs_total"] = np.sum(episode.hist_data["rejected_reqs"])
        episode.custom_metrics["rejected_reqs_total_percent"] = rejected_reqs_total_percent

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Called at the end of Algorithm.train()."""
        # Final checker to verify the callbacks are executed.
        result["callbacks_ok"] = True

    def _save_data(self, episode, info):
        current_time = info["current_time"]
        reward = info["reward"]
        reward_components = info["reward_components"]
        congested = info["congested"]
        rejected_reqs = info["actions"]["rejected"]
        obs = info["obs"]
        input_requests = obs[0]
        queue_capacity = obs[1]
        forwarding_capacity = obs[2]
        congested_queue_full = obs[3]
        congested_forwarding_exceed = obs[4]
        action = tuple(info["actions"].values())

        episode.user_data["current_time"].append(current_time)
        episode.user_data["reward"].append(reward)
        episode.user_data["reward_components"].append(reward_components)
        episode.user_data["congested"].append(congested)
        episode.user_data["congested_queue_full"].append(congested_queue_full)
        episode.user_data["congested_forwarding_exceed"].append(congested_forwarding_exceed)
        episode.user_data["rejected_reqs"].append(rejected_reqs)
        episode.user_data["input_requests"].append(input_requests)
        episode.user_data["queue_capacity"].append(queue_capacity)
        episode.user_data["forwarding_capacity"].append(forwarding_capacity)
        episode.user_data["actions"].append(action)
