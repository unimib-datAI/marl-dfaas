import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class TrafficManagementCallbacks(DefaultCallbacks):
    """Custom callbacks for the TrafficManagementEnv training.

    See API documentation for DefaultCallbacks, I use only a subset of the
    callbacks and keyword arguments."""

    def __init__(self):
        self.keys = ["current_time",
                     "reward",
                     "congested",
                     "rejected_reqs",
                     "input_requests",
                     "queue_capacity",
                     "forwarding_capacity",
                     "actions"
                     ]

    def on_episode_start(self, *, episode, **kwargs):
        """Callback run right after an Episode has started.

        Only the episode keyword argument is used, other arguments will be
        ignored."""
        # Make sure this episode has just been started (only initial obs logged
        # so far).
        assert episode.length <= 0, "'on_episode_start()' callback should be called right after env reset!"

        # See API documentation for Episode class.
        for key in self.keys:
            episode.user_data[key] = []
            episode.hist_data[key] = []

    def on_episode_step(self, *, episode, **kwargs):
        """Called on each episode step (after the action has been logged).

        Only the episode keyword argument is used, other arguments will be
        ignored."""
        # Make sure this episode is ongoing.
        assert episode.length > 0, "'on_episode_step()' callback should not be called right after env reset!"

        info = episode.last_info_for()

        current_time = info["current_time"]
        reward = info["reward"]
        congested = info["congested"]
        rejected_reqs = info["actions"]["rejected"]
        obs = info["obs"]
        input_requests = obs[0]
        queue_capacity = obs[1]
        forwarding_capacity = obs[2]
        action = tuple(info["actions"].values())

        episode.user_data["current_time"].append(current_time)
        episode.user_data["reward"].append(reward)
        episode.user_data["congested"].append(congested)
        episode.user_data["rejected_reqs"].append(rejected_reqs)
        episode.user_data["input_requests"].append(input_requests)
        episode.user_data["queue_capacity"].append(queue_capacity)
        episode.user_data["forwarding_capacity"].append(forwarding_capacity)
        episode.user_data["actions"].append(action)

    def on_episode_end(self, *, episode, **kwargs):
        """Called when an episode is done (after terminated/truncated have been
        logged).

        Only the episode keyword argument is used, other arguments will be
        ignored."""
        for key in self.keys:
            episode.hist_data[key] = episode.user_data[key]

        episode.custom_metrics["reward_mean"] = np.mean(episode.hist_data["reward"])
        episode.custom_metrics["congested_steps"] = np.sum(episode.hist_data["congested"])
        episode.custom_metrics["rejected_reqs_total"] = np.sum(episode.hist_data["rejected_reqs"])

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Called at the end of Algorithm.train()."""
        # Final checker to verify the callbacks are executed.
        result["callbacks_ok"] = True
