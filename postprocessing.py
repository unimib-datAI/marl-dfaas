from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import json
import ast
import os


def load_progress_file(
    exp_folder: str, last_iter: int
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Load results from the progress.csv file
  """
  # load progress file
  progress = pd.read_csv(os.path.join(exp_folder, "progress.csv"))
  # build dataframes
  all_hist_stats = pd.DataFrame()
  all_episode_hist_stats = pd.DataFrame()
  for it in progress["training_iteration"]:
    print(f"Iter {it}")
    # process results only if they are not already available
    if it > last_iter:
      row = progress[progress["training_iteration"] == it]
      # hist stats
      hist_stats_cols = [
        c for c in row.columns if c.startswith("env_runners/hist_stats/")
      ]
      hist_stats_dict = {
        c.split("/")[-1]: ast.literal_eval(
          row[c].iloc[0]
        ) for c in hist_stats_cols
      }
      hist_stats = pd.DataFrame({
        k: v for k, v in hist_stats_dict.items() if "episode_" not in k
      })
      episode_hist_stats = pd.DataFrame({
        k: v for k, v in hist_stats_dict.items() if "episode_" in k
      })
      # add information about the episode number
      hist_stats["episode"] = [-1] * len(hist_stats)
      # for _, temp in hist_stats.groupby("current_time"):
      #   hist_stats.loc[temp.index, "episode"] = range(len(temp))
      # concatenate
      hist_stats["iter"] = [it] * len(hist_stats)
      hist_stats["step"] = range(len(hist_stats))
      episode_hist_stats["iter"] = [it] * len(episode_hist_stats)
      all_hist_stats = pd.concat(
        [all_hist_stats, hist_stats], ignore_index = True
      )
      all_episode_hist_stats = pd.concat(
        [all_episode_hist_stats, episode_hist_stats], ignore_index = True
      )
  return all_hist_stats, all_episode_hist_stats


def plot_action(
    df: pd.DataFrame, agents: list, plot_folder: str = None, suffix: str = ""
  ):
  _, axs = plt.subplots(nrows = len(agents), ncols = 1, figsize = (30,8))
  for idx, agent in enumerate(agents):
    # -- input load
    df[f"observation_input_rate-{agent}"].plot(
      linewidth = 2,
      marker = ".",
      color = "k",
      ax = axs[idx],
      label = None,
      legend = False
    )
    # -- action
    df[[
      f"action_local-{agent}", 
      f"action_forward-{agent}", 
      f"action_reject-{agent}"
    ]].plot.bar(
      stacked = True,
      ax = axs[idx],
      label = None,
      legend = False
    )
    axs[idx].set_ylabel(agent)
    axs[idx].grid(visible = True, axis = "y")
  if plot_folder is not None:
    plt.savefig(
      os.path.join(plot_folder, f"actions{suffix}.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.title(title_key)
    plt.show()


def plot_moving_average(
    data: pd.DataFrame, 
    columns: list, 
    window: int, 
    plot_folder: str = None,
    title_key: str = "training",
    y_threshold: float = None
  ):
  """
  Plot the moving average over the given window of the results in the listed 
  columns
  """
  avg = data[columns].rolling(
    window = window,
    min_periods = 1
  ).mean()
  min_iter = data["iter"].min()
  max_iter = data["iter"].max()
  # plot
  ax = avg.plot()
  if y_threshold is not None:
    ax.axhline(y_threshold, color = "k", linestyle = "dashed", linewidth = 2)
  plt.grid()
  if plot_folder is not None:
    plt.savefig(
      os.path.join(
        plot_folder, 
        f"{title_key}_moving_average_{min_iter}_{max_iter}.png"
      ),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.title(title_key)
    plt.show()


def sum_latency(df: pd.DataFrame, agent: str) -> pd.Series:
  return df[
    f"response_time_avg_forwarded-{agent}"
  ] + df[
    f"network_forward_delay_avg-{agent}"
  ] + df[
    f"network_return_delay_avg-{agent}"
  ]


def unpack_step_values(
    all_hist_stats: pd.DataFrame
  ) -> Tuple[pd.DataFrame, set]:
  to_convert = all_hist_stats.select_dtypes("object").copy(deep = True)
  new_df = pd.DataFrame()
  all_agents = set()
  for col in to_convert:
    print(col)
    df_col = pd.DataFrame()
    if isinstance(to_convert[col].iloc[0], dict):
      agents = list(to_convert[col].iloc[0].keys())
      for iteration in to_convert[col].index:
        df_dict = {}
        n_steps = 0
        for agent in agents:
          df_dict[f"{col}-{agent}"] = to_convert.loc[iteration,col][agent]
          n_steps = len(to_convert.loc[iteration,col][agent])
          all_agents.add(agent)
        df_dict["step"] = range(n_steps)
        df_dict["iter"] = iteration
        # merge
        df_col = pd.concat([df_col, pd.DataFrame(df_dict)], ignore_index=True)
    else:
      print(". here")
    # join
    if len(new_df) == 0:
      new_df = df_col
    else:
      new_df = df_col.join(
        new_df.set_index(["iter", "step"], drop = True),
        on = ["iter", "step"],
        how = "inner"
      )
  return new_df, all_agents


def main(
    exp_folder: str, 
    moving_average_window: int = 10, 
    last_iter: int = 0,
    plot_iterations: list = []
  ):
  # create folder to store plots
  plot_folder = os.path.join(exp_folder, "plots")
  os.makedirs(plot_folder, exist_ok = True)
  # load data
  all_hist_stats, all_episode_hist_stats = load_progress_file(
    exp_folder, last_iter
  )
  # unpack by-step values
  all_hist_stats_unpacked, agents = unpack_step_values(all_hist_stats)
  # plot episode reward moving average
  plot_moving_average(
    all_episode_hist_stats, 
    ["episode_reward"], 
    moving_average_window, 
    plot_folder, 
    "episode_total_reward"
  )
  # -- by node
  plot_moving_average(
    all_hist_stats, 
    [f"policy_policy_{a}_reward" for a in agents], 
    moving_average_window, 
    plot_folder, 
    "by_node_reward"
  )
  
  # compute by-episode average
  avg_stats_unpacked = all_hist_stats_unpacked.groupby(
    "iter"
  ).mean().reset_index()
  # plot utility
  plot_moving_average(
    avg_stats_unpacked, 
    [
      f"loc_utility-{a}" for a in agents
    ] + [
      f"fwd_utility-{a}" for a in agents
    ] + [
      f"rej_penalty-{a}" for a in agents
    ], 
    moving_average_window, 
    plot_folder, 
    "utilities"
  )
  # plot response time
  for a in agents:
    avg_stats_unpacked[f"response_time_avg_fwd-{a}"] = sum_latency(
      avg_stats_unpacked, a
    )
  plot_moving_average(
    avg_stats_unpacked, 
    [
      f"response_time_avg_local-{a}" for a in agents
    ] + [
      f"response_time_avg_fwd-{a}" for a in agents
    ], 
    moving_average_window, 
    plot_folder, 
    "response_time_avg",
    y_threshold = 0.5
  )
  # plot rejections
  plot_moving_average(
    avg_stats_unpacked, 
    [
      f"action_reject-{a}" for a in agents
    ] + [
      f"incoming_rate_reject-{a}" for a in agents
    ] + [
      f"forward_reject_rate-{a}" for a in agents
    ], 
    moving_average_window, 
    plot_folder, 
    "reject"
  )
  # plot "average" actions
  plot_action(avg_stats_unpacked, agents, plot_folder)
  # plot actions in specific iterations
  for iteration in plot_iterations:
    plot_action(
      all_hist_stats_unpacked[all_hist_stats_unpacked["iter"] == iteration], 
      agents, 
      plot_folder,
      f"-iter_{iteration}"
    )
  


if __name__ == "__main__":
  exp_folder = "results/DF_20251204_121104_PPO_3agents"
  main(exp_folder)
