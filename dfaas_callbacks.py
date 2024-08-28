from ray.rllib.algorithms.callbacks import DefaultCallbacks


class DFaaSCallbacks(DefaultCallbacks):
    """User defined callbacks for the DFaaS environment.

    See the Ray's API documentation for DefaultCallbacks, the custom class
    overrides (and uses) only a subset of callbacks and keyword arguments."""

    def on_episode_start(self, *, episode, base_env, **kwargs):
        """Callback run right after an episode has started.

        Only the episode and base_env keyword arguments are used, other
        arguments are ignored."""
        # Make sure this episode has just been started (only initial obs logged
        # so far).
        assert episode.length <= 0, f"'on_episode_start()' callback should be called right after env reset! {episode.length = }"

        env = base_env.envs[0]

        # Initialize the dictionaries and lists.
        episode.user_data["observation_queue_capacity"] = {agent: [] for agent in env.agent_ids}
        episode.user_data["observation_input_requests"] = {agent: [] for agent in env.agent_ids}
        episode.user_data["observation_forward_capacity"] = {agent: [] for agent in env.agent_ids}
        episode.user_data["original_input_requests"] = []
        episode.user_data["action_local"] = {agent: [] for agent in env.agent_ids}
        episode.user_data["action_forward"] = {agent: [] for agent in env.agent_ids}
        episode.user_data["action_reject"] = {agent: [] for agent in env.agent_ids}
        episode.user_data["reward"] = {agent: [] for agent in env.agent_ids}
        episode.hist_data["seed"] = [env.seed]

        # The way to get the info data is complicated because of the Ray API.
        # However, we need to save the first observation because it contains the
        # initial data.
        info = env._additional_info()

        # Track common info for all agents.
        for agent in env.agent_ids:
            # Note that each element is a np.ndarray of size 1. It must be
            # unwrapped!
            episode.user_data["observation_queue_capacity"][agent].append(info[agent]["observation"]["queue_capacity"].item())
            episode.user_data["observation_input_requests"][agent].append(info[agent]["observation"]["input_requests"].item())

        # Track forwarded capacity only for node_0.
        episode.user_data["observation_forward_capacity"]["node_0"].append(info["node_0"]["observation"]["forward_capacity"].item())

        # Track the original input requests only for node_1.
        episode.user_data["original_input_requests"].append(info["node_1"]["original_input_requests"])

    def on_episode_step(self, *, episode, base_env, **kwargs):
        """Called on each episode step (after the action has been logged).

        Only the episode and base_env keyword arguments are used, other
        arguments are ignored"""
        # Make sure this episode is ongoing.
        assert episode.length > 0, f"'on_episode_step()' callback should not be called right after env reset! {episode.length = }"

        env = base_env.envs[0]

        info = env._additional_info()

        # Track common info for all agents.
        for agent in env.agent_ids:
            episode.user_data["action_local"][agent].append(info[agent]["action"]["local"])
            episode.user_data["action_reject"][agent].append(info[agent]["action"]["reject"])
            episode.user_data["reward"][agent].append(info[agent]["reward"])

        # Track forwarded requests only for node_0.
        episode.user_data["action_forward"]["node_0"].append(info["node_0"]["action"]["forward"])

        # If it is the last step, skip the observation because it will not be
        # paired with the next action.
        if env.current_step < env.max_steps:
            for agent in env.agent_ids:
                episode.user_data["observation_queue_capacity"][agent].append(info[agent]["observation"]["queue_capacity"].item())
                episode.user_data["observation_input_requests"][agent].append(info[agent]["observation"]["input_requests"].item())

            # Track forwarded capacity only for node_0.
            episode.user_data["observation_forward_capacity"]["node_0"].append(info["node_0"]["observation"]["forward_capacity"].item())

            # Track the original input requests only for node_1.
            episode.user_data["original_input_requests"].append(info["node_1"]["original_input_requests"])

    def on_episode_end(self, *, episode, **kwargs):
        """Called when an episode is done (after terminated/truncated have been
        logged).

        Only the episode keyword arguments is used, other arguments are
        ignored."""
        # Note that this has to be a list of length 1 because there can be
        # multiple episodes in a single iteration, so at the end Ray will append
        # the list to a general list for the iteration.
        episode.hist_data["observation_queue_capacity"] = [episode.user_data["observation_queue_capacity"]]
        episode.hist_data["observation_input_requests"] = [episode.user_data["observation_input_requests"]]
        episode.hist_data["observation_forward_capacity"] = [episode.user_data["observation_forward_capacity"]]
        episode.hist_data["original_input_requests"] = [episode.user_data["original_input_requests"]]
        episode.hist_data["action_local"] = [episode.user_data["action_local"]]
        episode.hist_data["action_forward"] = [episode.user_data["action_forward"]]
        episode.hist_data["action_reject"] = [episode.user_data["action_reject"]]
        episode.hist_data["reward"] = [episode.user_data["reward"]]

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Called at the end of Algorithm.train()."""
        # Final checker to verify the callbacks are executed.
        result["callbacks_ok"] = True

        # The problem here is that Ray cumulates the values of the keys under
        # hist_stats across iterations, but I do not want this behavior.
        # Solution: keep only the values generated by episodes in this
        # iteration.
        episodes = result["episodes_this_iter"]
        for key in result["hist_stats"]:
            result["hist_stats"][key] = result["hist_stats"][key][-episodes:]

        # Because they are repeated by Ray within the result dictionary.
        del result["sampler_results"]
