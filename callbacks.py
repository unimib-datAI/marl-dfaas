import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class TrafficManagementCallbacks(DefaultCallbacks):
    """Custom callbacks for the TrafficManagementEnv training.

    See API documentation for DefaultCallbacks, I use only a subset of the
    callbacks and keyword arguments."""

    def on_episode_start(self, *, episode, **kwargs):
        """Callback run right after an Episode has started.

        Only the episode keyword argument is used, other arguments will be
        ignored."""
        # Make sure this episode has just been started (only initial obs logged
        # so far).
        assert episode.length <= 0, "'on_episode_start()' callback should be called right after env reset!"

        # See API documentation for Episode class.
        episode.user_data["current_time"] = []
        episode.hist_data["current_time"] = []

        episode.user_data["reward"] = []
        episode.hist_data["reward"] = []

        episode.user_data["congested"] = []
        episode.hist_data["congested"] = []

        episode.user_data["rejected_reqs"] = []
        episode.hist_data["rejected_reqs"] = []

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

        episode.user_data["current_time"].append(current_time)
        episode.user_data["reward"].append(reward)
        episode.user_data["congested"].append(congested)
        episode.user_data["rejected_reqs"].append(rejected_reqs)

    def on_episode_end(self, *, episode, **kwargs):
        """Called when an episode is done (after terminated/truncated have been
        logged).

        Only the episode keyword argument is used, other arguments will be
        ignored."""
        episode.hist_data["current_time"] = episode.user_data["current_time"]
        episode.hist_data["reward"] = episode.user_data["reward"]
        episode.hist_data["congested"] = episode.user_data["congested"]
        episode.hist_data["rejected_reqs"] = episode.user_data["rejected_reqs"]

        episode.custom_metrics["reward_mean"] = np.mean(episode.hist_data["reward"])
        episode.custom_metrics["congested_steps"] = np.sum(episode.hist_data["congested"])
        episode.custom_metrics["rejected_reqs_total"] = np.sum(episode.hist_data["rejected_reqs"])

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Called at the end of Algorithm.train()."""
        # Final checker to verify the callbacks are executed.
        result["callbacks_ok"] = True
