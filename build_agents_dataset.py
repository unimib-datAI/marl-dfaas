from datetime import datetime, timedelta
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import json
import os


def add_cp_bucket(prmetrics_avg, k6datelimits):
  df = prmetrics_avg.reset_index().rename(columns={"index": "ts"}).copy()
  df["ts"] = pd.to_datetime(df["ts"])
  df = df.sort_values("ts")
  #
  limits = k6datelimits.reset_index().copy()
  limits["date_min"] = pd.to_datetime(limits["date_min"])
  limits["date_max"] = pd.to_datetime(limits["date_max"])
  limits = limits.sort_values("date_min")
  #
  joined = pd.merge_asof(
    df,
    limits,
    left_on="ts",
    right_on="date_min",
    direction="backward",
    allow_exact_matches=True
  )
  # reject rows that fall outside the interval
  mask = joined["date_min"].notna() & (joined["ts"] <= joined["date_max"])
  joined = joined.loc[mask, :]
  return joined


def average_in_control_period(
    metrics_avg: pd.DataFrame, control_period: str
  ) -> pd.DataFrame:
  metrics_avg["cp_bucket"] = metrics_avg.index.floor(control_period)
  avg_in_cp = metrics_avg.resample(control_period).mean()
  if "iterations" in avg_in_cp:
    avg_in_cp["iterations"] = metrics_avg.resample(
      control_period
    )["iterations"].sum()
  bucket_map = {b: i for i, b in enumerate(avg_in_cp.index)}
  metrics_avg["cp_bucket"] = metrics_avg["cp_bucket"].map(bucket_map)
  avg_in_cp["cp_bucket"] = range(len(avg_in_cp))
  return avg_in_cp, metrics_avg


def average_in_control_period2(
    metrics_avg: pd.DataFrame
  ) -> pd.DataFrame:
  metrics_avg["cp_bucket"] = metrics_avg["load_interval"].astype(int)
  avg_in_cp = metrics_avg.groupby("cp_bucket").mean()
  if "iterations" in avg_in_cp:
    avg_in_cp["iterations"] = metrics_avg.groupby(
      "cp_bucket"
    )["iterations"].sum()
  return avg_in_cp, metrics_avg


def compute_average(metrics_wide: pd.DataFrame) -> pd.DataFrame:
  metrics_avg = metrics_wide.groupby("timestamp").mean()
  if "http_reqs" in metrics_avg:
    metrics_avg["http_reqs"] = metrics_wide.groupby(
      "timestamp"
    )["http_reqs"].sum()
    metrics_avg["date"] = [
      datetime.fromtimestamp(t) for t in metrics_avg.index
    ]
  else:
    metrics_avg["date"] = [
      datetime.strptime(t, "%Y-%m-%dT%H:%M:%S") for t in metrics_avg.index
    ]
  return metrics_avg.set_index("date").sort_index()


def load_metrics(filename):
  metrics = pd.DataFrame()
  try:
    colnames = [
      'metric_name', 'timestamp', 'metric_value', 'check', 'error',
      'error_code', 'expected_response', 'group', 'method', 'name', 'proto',
      'scenario', 'service', 'status', 'subproto', 'tls_version', 'url',
      'extra_tags', 'metadata'
    ]
    metrics = pd.read_csv(
      filename, compression = "gzip", low_memory = True, dtype = {
        k: object if k != "timestamp" and "value" not in k else (
          int if k == "timestamp" else float
        ) for k in colnames
      }
    )
  except pd.errors.ParserError:
    colnames = ['metric', 'type', 'timestamp', 'labels', 'value']
    metrics = pd.read_csv(
      filename, compression = "gzip", delimiter = ";", 
      low_memory = True, dtype = {
        k: object if k != "value" else float for k in metrics
      }
    )
  if "k6" in filename:
    metrics = metrics.loc[
      (
        metrics["metric_name"].str.startswith("http_req")
      ) | (
        metrics["metric_name"].str.startswith("data_")
      ) | (
        metrics["metric_name"].str.startswith("iteration")
      ),
      ["timestamp", "metric_name", "metric_value", "extra_tags"]
    ]
  elif "prometheus" in filename:
    metrics.rename(
      columns = {"metric": "metric_name", "value": "metric_value"},
      inplace = True
    )
  return metrics


def plot_metric(
    metrics_avg: pd.DataFrame, 
    avg_in_cp: pd.DataFrame, 
    metric: str, 
    pred: pd.Series = None
  ):
  _, ax = plt.subplots()
  metrics_avg.plot(
    y = metric,
    marker = ".",
    linewidth = 0.1,
    alpha = 0.6,
    ax = ax,
    grid = True,
    color = mcolors.TABLEAU_COLORS["tab:blue"]
  )
  avg_in_cp.plot(
    y = metric,
    linewidth = 2,
    ax = ax,
    grid = True,
    color = mcolors.TABLEAU_COLORS["tab:blue"],
    label = f"{metric} (avg.)"
  )
  if pred is not None:
    pred.plot(
      linewidth = 2,
      ax = ax,
      grid = True,
      color = mcolors.TABLEAU_COLORS["tab:red"],
      label = f"{metric} (pred.)"
    )
  plt.show()


def reshape(metrics: pd.DataFrame) -> pd.DataFrame:
  # -- add a counter for each metric per timestamp
  cols = ["timestamp", "metric_name"]
  if "stage" in metrics:
    cols += ["stage"]
  metrics["row_id"] = metrics.groupby(cols).cumcount()
  # -- pivot using the row_id to keep duplicates
  cols = ["timestamp", "row_id"]
  if "stage" in metrics:
    cols += ["stage"]
  metrics_wide = (
      metrics
          .pivot(index=cols,
                columns="metric_name",
                values="metric_value")
          .reset_index()
          .drop(columns="row_id")
  )
  metrics_wide.columns.name = None
  return metrics_wide


def main():
  control_period = "1min"
  base_dataset_folder = "/home/federica/mlimage"
  # index traces
  trace_idxs = {}
  all_traces = pd.DataFrame()
  trace_type_to_idx = []
  for foldername in os.listdir(base_dataset_folder):
    if "_node_" in foldername:
      dataset_folder = os.path.join(base_dataset_folder, foldername)
      tokens = foldername.split("_")
      node_idx = int(tokens[-1])
      trace_type = "_".join(tokens[:-2])
      trace_idx = None
      try:
        trace_idx = trace_type_to_idx.index(trace_type)
      except ValueError:
        trace_type_to_idx.append(trace_type)
        trace_idx = len(trace_type_to_idx) - 1
      trace_idx = f"{trace_idx}{node_idx}"  
      # -- k6/prometheus results
      trace_idxs[trace_idx] = {
        "k6": os.path.join(dataset_folder, "k6_results.csv.gz"), 
        "pmt": os.path.join(dataset_folder, "prometheus_metrics.csv.gz"),
        "stages": os.path.join(dataset_folder, "dfaas_node_k6_stages.csv")
      }
      # -- traces
      for filename in os.listdir(dataset_folder):
        if filename.startswith("input_requests") and filename.endswith(".json"):
          tt = "_".join(filename.split("_")[-2:]).replace(".json", "")
          if len(all_traces) == 0 or not (all_traces["trace_type"].eq(tt)).any():  
            trace = {}
            with open(os.path.join(dataset_folder, filename), "r") as ist:
              trace = json.load(ist)["0"]
            trace = pd.DataFrame(trace)
            trace["time"] = trace.index
            trace["trace_type"] = tt
            all_traces = pd.concat([all_traces, trace], ignore_index = True)
  # -- save
  res_dataset_folder = "dataset"
  all_traces.to_csv(
    os.path.join(res_dataset_folder, "all_traces.csv"), index = False
  )
  with open(os.path.join(res_dataset_folder, "trace_idxs.json"), "w") as ist:
    ist.write(json.dumps(trace_idxs, indent = 2))
  with open(
      os.path.join(res_dataset_folder, "trace_type_to_idx.txt"), "w"
    ) as ist:
    for el in trace_type_to_idx:
      ist.write(f"{el}\n")
  # load and join
  all_joined_metrics = pd.DataFrame()
  all_joined_metrics_avg = pd.DataFrame()
  for trace_idx, files in trace_idxs.items():
    print(f"Load trace {trace_idx}")
    # stages
    stages = pd.read_csv(files["stages"])
    # k6 metrics
    k6metrics = load_metrics(files["k6"])
    k6metrics = k6metrics.join(
      stages.set_index("timestamp", drop = True), on = "timestamp"
    )
    k6metrics = k6metrics[k6metrics["stage"] >= 0]
    # k6metrics["extra_tags"] = [
    #   int(t.split("=")[-1]) for t in k6metrics["extra_tags"]
    # ]
    k6metrics_wide = reshape(k6metrics)
    k6metrics_wide["load_interval"] = k6metrics_wide["stage"] // 2
    k6metrics_wide.drop("stage", axis = "columns", inplace = True)
    k6metrics_avg = compute_average(k6metrics_wide)
    k6avg_in_cp, k6metrics_avg = average_in_control_period2(k6metrics_avg)
    k6metrics_avg["date"] = k6metrics_avg.index
    k6datelimits = pd.DataFrame(
      k6metrics_avg.groupby(
        ["cp_bucket","load_interval"]
      )["date"].min()
    ).join(
      pd.DataFrame(
        k6metrics_avg.groupby(
          ["cp_bucket","load_interval"]
        )["date"].max()
      ), 
      lsuffix = "_min",
      rsuffix = "_max"
    ).reset_index()
    # prometheus metrics
    prmetrics = load_metrics(files["pmt"])
    prmetrics = prmetrics[
      (
        prmetrics["labels"].str.contains("function_name=mlimage")
      ) | (
        prmetrics["labels"].str.contains("container=mlimage")
      )
    ]
    prmetrics_wide = reshape(prmetrics)
    prmetrics_avg = compute_average(prmetrics_wide)
    prmetrics_avg.index = [d + timedelta(hours=1) for d in prmetrics_avg.index]
    # add load interval (cp bucket) info and average
    prmetrics_avg = add_cp_bucket(prmetrics_avg, k6datelimits)
    pravg_in_cp, prmetrics_avg = average_in_control_period2(prmetrics_avg)
    # join
    # -- detailed
    start = k6metrics_avg.index[0]
    k6metrics_avg_resampled = k6metrics_avg.resample(
      "5s",
      origin = start,   # anchor at first sample
      label = "left",
      closed = "left"
    ).mean().ffill()
    k6metrics_avg_resampled = k6metrics_avg_resampled.astype(
      {"cp_bucket": int}
    ).reset_index(drop = True)
    joined_metrics = k6metrics_avg_resampled.join(
      prmetrics_avg.reset_index(drop = True).drop(
        columns = ["load_interval", "cp_bucket"]
      ),
      how = "right"
    )
    joined_metrics["trace_idx"] = trace_idx
    all_joined_metrics = pd.concat(
      [all_joined_metrics, joined_metrics], ignore_index = True
    )
    # -- average
    joined_metrics_avg = k6avg_in_cp.join(
      pravg_in_cp.drop(columns = ["load_interval"]),
      how = "right"
    )
    joined_metrics_avg["trace_idx"] = trace_idx
    all_joined_metrics_avg = pd.concat(
      [
        all_joined_metrics_avg, 
        joined_metrics_avg.reset_index()
      ], 
      ignore_index = True
    )
    # save
    all_joined_metrics.to_csv(
      "dataset/mlimage_joined_metrics.csv", index = False
    )
    all_joined_metrics_avg.to_csv(
      "dataset/mlimage_joined_metrics_avg.csv", index = False
    )
  #   # plot
  #   plot_metric(k6metrics_avg, k6avg_in_cp, "http_req_duration")
  #   plot_metric(
  #     k6metrics_avg, k6avg_in_cp, "http_reqs", all_traces[str(trace_idx)]
  #   )
  #   plot_metric(prmetrics_avg, pravg_in_cp, "container_memory_usage_bytes")
  #   plot_metric(prmetrics_avg, pravg_in_cp, "gateway_service_count")
  #   plot_metric(prmetrics_avg, pravg_in_cp, "gateway_function_invocation_started")
  #   plot_metric(prmetrics_avg, pravg_in_cp, "cpu_usage_percent")


  # # stats = k6metrics_wide.loc[
  # #   :,
  # #   (
  # #     k6metrics_wide.columns.str.startswith("http_req_duration")
  # #   ) | (
  # #     k6metrics_wide.columns.str.startswith("timestamp")
  # #   )
  # # ].groupby("timestamp").describe()["http_req_duration"]

  # # ax = stats['mean'].plot(label="Mean latency over time")
  # # stats['std'].plot(label="Std deviation over time", ax = ax)
  # # plt.show()

  # # stats['cv'] = stats['std'] / stats['mean']
  # # stats['cv'].describe()


if __name__ == "__main__":
  main()
