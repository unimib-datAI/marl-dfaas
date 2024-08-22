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

        episode.user_data["input_requests"] = {"node_0": [], "node_1": []}

        # The way to get the info data is complicated because of the Ray API.
        # However, we need to save the first observation because it contains the
        # initial data.
        info = base_env.envs[0]._additional_info()

        # Track the input requests for each agent.
        turn = info["__common__"]["turn"]
        episode.user_data["input_requests"][turn].append(info[turn]["input_requests"])

    def on_episode_step(self, *, episode, base_env, **kwargs):
        """Called on each episode step (after the action has been logged).

        Only the episode and base_env keyword arguments are used, other
        arguments are ignored"""
        # Make sure this episode is ongoing.
        assert episode.length > 0, f"'on_episode_step()' callback should not be called right after env reset! {episode.length = }"

        info = base_env.envs[0]._additional_info()
        turn = info["__common__"]["turn"]
        episode.user_data["input_requests"][turn].append(info[turn]["input_requests"])

    def on_episode_end(self, *, episode, base_env, **kwargs):
        """Called when an episode is done (after terminated/truncated have been
        logged).

        Only the episode and base_env keyword arguments are used, other
        arguments are ignored"""
        env = base_env.envs[0]

        # Because of the way the environment is designed, an agent will have an
        # extra input request that must be dropped because it's the last
        # observation of the environment (it won't have the next action).
        for agent_id in episode.user_data["input_requests"]:
            input_requests = episode.user_data["input_requests"][agent_id]
            if len(input_requests) > env.node_max_steps:
                # Remove the last input request by moving the list one to the
                # right.
                input_requests = input_requests[:-1]
                episode.user_data["input_requests"][agent_id] = input_requests

        # Save the input requests for each agent for this episode. Note that
        # this has to be a list of length 1 because there can be multiple
        # episodes in a single iteration, so at the end Ray will append the list
        # to a general list for the iteration.
        episode.hist_data["input_requests"] = [episode.user_data["input_requests"]]

        # Save seed for this episode.
        episode.hist_data["seed"] = [env.seed]

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Called at the end of Algorithm.train()."""
        # Final checker to verify the callbacks are executed.
        result["callbacks_ok"] = True
