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


def plot_action(df: pd.DataFrame, agents: list, plot_folder: str = None):
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
    axs[idx].grid(visible = True, axis = "y")
  if plot_folder is not None:
    plt.savefig(
      os.path.join(plot_folder, "actions.png"),
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
    title_key: str = "training"
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
  avg.plot()
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


def unpack_step_values(all_hist_stats: pd.DataFrame) -> pd.DataFrame:
  to_convert = all_hist_stats.select_dtypes("object").copy(deep = True)
  new_df = pd.DataFrame()
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
  return new_df


def main(exp_folder: str, moving_average_window: int = 10, last_iter: int = 0):
  # create folder to store plots
  plot_folder = os.path.join(exp_folder, "plots")
  os.makedirs(plot_folder, exist_ok = True)
  # load data
  all_hist_stats, all_episode_hist_stats = load_progress_file(
    exp_folder, last_iter
  )
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
    ["policy_policy_node_0_reward", "policy_policy_node_1_reward"], 
    moving_average_window, 
    plot_folder, 
    "by_node_reward"
  )
  # unpack by-step values
  all_hist_stats_unpacked = unpack_step_values(all_hist_stats)
  # compute by-episode average
  avg_stats_unpacked = all_hist_stats_unpacked.groupby("iter").mean().reset_index()
  plot_moving_average(
    avg_stats_unpacked, 
    [
      "loc_utility-node_0", "fwd_utility-node_0", "rej_penalty-node_0",
      "loc_utility-node_1", "fwd_utility-node_1", "rej_penalty-node_1"
    ], 
    moving_average_window, 
    plot_folder, 
    "utilities"
  )
  # plot "average" actions
  plot_action(avg_stats_unpacked, ["node_0", "node_1"], plot_folder)
  


if __name__ == "__main__":
  exp_folder = "results/DF_20251203_172416_PPO_rrof_lbr"
  main(exp_folder)
