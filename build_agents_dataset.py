from datetime import datetime, timedelta
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


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
  metrics = pd.read_csv(filename, compression = "gzip")
  if "k6" in filename:
    metrics = metrics.loc[
      (
        metrics["metric_name"].str.startswith("http_req")
      ) | (
        metrics["metric_name"].str.startswith("data_")
      ) | (
        metrics["metric_name"].str.startswith("iteration")
      ),
      ["timestamp", "metric_name", "metric_value"]
    ]
  elif "prometheus" in filename:
    metrics.rename(
      columns = {"metric": "metric_name", "value": "metric_value"},
      inplace = True
    )
  return metrics


def plot_metric(
    metrics_avg: pd.DataFrame, avg_in_cp: pd.DataFrame, metric: str
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
  plt.show()


def reshape(metrics: pd.DataFrame) -> pd.DataFrame:
  # -- add a counter for each metric per timestamp
  metrics["row_id"] = (
      metrics
          .groupby(["timestamp", "metric_name"])
          .cumcount()
  )
  # -- pivot using the row_id to keep duplicates
  metrics_wide = (
      metrics
          .pivot(index=["timestamp", "row_id"],
                columns="metric_name",
                values="metric_value")
          .reset_index()
          .drop(columns="row_id")
  )
  metrics_wide.columns.name = None
  return metrics_wide


def main():
  control_period = "1min"
  # k6 metrics
  k6metrics = load_metrics("dataset/k6_results.csv.gz")
  k6metrics_wide = reshape(k6metrics)
  k6metrics_avg = compute_average(k6metrics_wide)
  k6avg_in_cp, k6metrics_avg = average_in_control_period(
    k6metrics_avg, control_period
  )
  # prometheus metrics
  prmetrics = load_metrics("dataset/prometheus_metrics.csv.gz")
  prmetrics_wide = reshape(prmetrics)
  prmetrics_avg = compute_average(prmetrics_wide)
  pravg_in_cp, prmetrics_avg = average_in_control_period(
    prmetrics_avg, control_period
  )
  # plot
  plot_metric(k6metrics_avg, k6avg_in_cp, "http_req_duration")
  plot_metric(k6metrics_avg, k6avg_in_cp, "http_reqs")
  plot_metric(prmetrics_avg, pravg_in_cp, "container_memory_usage_bytes")
  plot_metric(prmetrics_avg, pravg_in_cp, "gateway_service_count")
  plot_metric(prmetrics_avg, pravg_in_cp, "gateway_function_invocation_started")
  # join (keep intervals that where collected by both)
  # -- detailed
  prmetrics_avg.index = [d + timedelta(hours=1) for d in prmetrics_avg.index]
  joined_metrics = k6metrics_avg.resample("5s").mean().join(
    prmetrics_avg.drop("cp_bucket", axis = "columns")
  )
  # -- average
  pravg_in_cp.index = [d + timedelta(hours=1) for d in pravg_in_cp.index]
  joined_metrics_avg = k6avg_in_cp.join(
    pravg_in_cp.drop("cp_bucket", axis = "columns")
  )
  # save
  joined_metrics.to_csv("dataset/joined_metrics.csv")
  joined_metrics_avg.to_csv("dataset/joined_metrics_avg.csv")


if __name__ == "__main__":
  main()
