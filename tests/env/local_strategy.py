# This is a test script that instantiates an environment and each agent always
# chooses to process incoming requests locally.
import sys
import os

# Add the current directory (where Python is called) to sys.path. This is
# required to load modules in the project root directory (like dfaas_utils.py).
sys.path.append(os.getcwd())

import dfaas_env  # pylint: disable=import-error


def main():
    env = dfaas_env.DFaaS()

    env.reset(seed=42)
    print("Step 0 (reset)")
    for agent in env.agents:
        print(f"  {agent} (observation)")
        print(
            "      input_requests =",
            env.info["observation_input_requests"][agent][env.current_step],
        )
        print(
            "          queue_size =",
            env.info["observation_queue_size"][agent][env.current_step],
        )
    print()

    for i in range(10):
        action = {agent: [1.0, 0.0, 0.0] for agent in env.agents}
        env.step(action)

        # Note that after "step()" the current step is moved forward, but the
        # action is performed on the previous step.

        for agent in env.agents:
            print(f"  {agent} (action)")
            print(
                "        action_local =",
                env.info["action_local"][agent][env.current_step - 1],
            )
            print(
                "          queue_size =",
                env.info["queue_size"][agent][env.current_step - 1],
            )
            print(
                "     processed_local =",
                env.info["processed_local"][agent][env.current_step - 1],
            )
        print()

        print("Step", i + 1)
        for agent in env.agents:
            print(f"  {agent} (observation)")
            print(
                "      input_requests =",
                env.info["observation_input_requests"][agent][env.current_step],
            )
            print(
                "          queue_size =",
                env.info["observation_queue_size"][agent][env.current_step],
            )
        print()

    print("Terminated")


if __name__ == "__main__":
    main()
