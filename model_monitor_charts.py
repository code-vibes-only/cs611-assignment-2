import argparse
import glob
import os
import pickle
from datetime import timedelta

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw to files; there is no screen inside the container
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# to call this script: python model_monitor_charts.py --snapshotdate "2025-07-01"

CHART_DIRECTORY = "datamart/monitoring_charts/"
STATUS_COLOURS = {"green": "#0ca30c", "amber": "#fab219", "red": "#d03b3b"}
FEED_STOPPED = pd.Timestamp("2024-07-01").date()  # first application month with no clickstream data

plt.rcParams.update({
    "figure.facecolor": "white", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.8, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "text.color": "#0b0b0b", "font.size": 10, "legend.frameon": False,
})


def time_axis(ax, months, snapshotdate):
    """One tick per plotted month; once the run has reached it, a line at the month the clickstream feed stopped"""
    months = list(months)
    if pd.Timestamp(snapshotdate).date() >= FEED_STOPPED:
        months.append(FEED_STOPPED)
        ax.axvline(FEED_STOPPED, color="#898781", linewidth=1)
        ax.annotate(" clickstream feed stopped", (FEED_STOPPED, 1), xycoords=("data", "axes fraction"),
                    va="top", fontsize=9, color="#52514e")
    ax.set_xlim(min(months) - timedelta(days=20), max(months) + timedelta(days=20))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def threshold(ax, y, label):
    """Horizontal hairline at a status band edge, labelled in the right margin"""
    ax.axhline(y, color="#898781", linewidth=0.8)
    ax.annotate(label, (1, y), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color="#52514e")


def status_points(ax, data):
    """One marker per month, coloured by its status, with the status named in the legend"""
    for status, colour in STATUS_COLOURS.items():
        points = data[data["status"] == status]
        if len(points):
            ax.scatter(points["prediction_month"], points["value"], color=colour, s=60, zorder=3,
                       edgecolors="white", linewidths=1.5, label=status)


def main(snapshotdate):
    # --- empty the chart folder: it holds only the charts as of this month ---
    os.makedirs(CHART_DIRECTORY, exist_ok=True)
    for old_chart in glob.glob(CHART_DIRECTORY + "*.png"):
        os.remove(old_chart)

    # --- load every monitoring partition computed up to this month ---
    last_file = "datamart/gold/model_monitoring/model_monitoring_" + snapshotdate.replace('-', '_') + ".parquet"
    files = [f for f in sorted(glob.glob("datamart/gold/model_monitoring/*.parquet")) if f <= last_file]
    if not files:
        print(f"no monitoring results up to {snapshotdate}, skipping")
        return
    monitoring = pd.concat([pd.read_parquet(f) for f in files])

    # --- performance: AUC and bad rates of each scored month, once its labels have arrived ---
    perf = monitoring[monitoring["metric_group"] == "performance"].sort_values("prediction_month")
    if len(perf):
        auc = perf[perf["metric"] == "auc"]
        with open(f"model_bank/{auc['model_name'].iloc[-1]}.pkl", "rb") as file:
            auc_at_training = pickle.load(file)["results"]["auc_oot"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, height_ratios=[3, 2])
        ax1.plot(auc["prediction_month"], auc["value"], color="#52514e", linewidth=2)
        status_points(ax1, auc)
        threshold(ax1, auc_at_training, f"OOT AUC\nat training\n{auc_at_training:.3f}")
        threshold(ax1, auc_at_training - 0.05, "amber\nbelow")
        threshold(ax1, auc_at_training - 0.10, "red\nbelow")
        time_axis(ax1, auc["prediction_month"], snapshotdate)
        ax1.set_ylim(auc_at_training - 0.20, auc_at_training + 0.05)
        ax1.set_ylabel("AUC")
        ax1.legend(title="AUC status", loc="lower left", ncols=3)
        ax1.set_title(f"Performance by scored month, measured 6 months later when the labels arrive (as of {snapshotdate})",
                      loc="left", fontsize=11)

        for metric, colour, label in [("actual_bad_rate", "#2a78d6", "actual"), ("predicted_bad_rate", "#eb6834", "predicted (mean score)")]:
            rate = perf[perf["metric"] == metric]
            ax2.plot(rate["prediction_month"], rate["value"], color=colour, linewidth=2, marker="o", label=label)
        time_axis(ax2, auc["prediction_month"], snapshotdate)
        ax2.set_ylim(0, 0.45)
        ax2.set_ylabel("Bad rate")
        ax2.legend(loc="lower left", ncols=2)
        fig.tight_layout()
        fig.savefig(CHART_DIRECTORY + "performance.png", dpi=150)
        plt.close(fig)

    # --- stability: PSI of the score distribution against training, with the PSI bands ---
    psi = monitoring[monitoring["metric"] == "psi_score"].sort_values("prediction_month")
    if len(psi):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(psi["prediction_month"], psi["value"], color="#52514e", linewidth=2)
        status_points(ax, psi)
        threshold(ax, 0.1, "amber\nabove 0.1")
        threshold(ax, 0.25, "red\nabove 0.25")
        time_axis(ax, psi["prediction_month"], snapshotdate)
        ax.set_ylim(0, max(0.5, psi["value"].max() * 1.15))
        ax.set_ylabel("PSI")
        ax.legend(title="PSI status", loc="upper left", ncols=3, bbox_to_anchor=(0, 0.9))
        ax.set_title(f"Score stability: PSI of each month's scores against the training scores (as of {snapshotdate})",
                     loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(CHART_DIRECTORY + "score_stability.png", dpi=150)
        plt.close(fig)

    # --- feature stability: CSI status of each feature by month, the 20 most drifted features ---
    csi = monitoring[monitoring["metric"].str.startswith("csi_")]
    if len(csi):
        values = csi.pivot_table(index="metric", columns="prediction_month", values="value")
        order = values.max(axis=1).sort_values(ascending=False).index
        shown, hidden = order[:20], order[20:]
        csi = csi.assign(code=csi["status"].map({"green": 0, "amber": 1, "red": 2}))  # the status stored by monitoring
        codes = csi.pivot(index="metric", columns="prediction_month", values="code").loc[shown]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.pcolormesh(codes.values, cmap=ListedColormap(list(STATUS_COLOURS.values())), vmin=-0.5, vmax=2.5,
                      edgecolors="white", linewidth=2)
        ax.set_xticks([i + 0.5 for i in range(len(codes.columns))], [d.strftime("%Y-%m") for d in codes.columns], rotation=45)
        ax.set_yticks([i + 0.5 for i in range(len(shown))], [m.replace("csi_", "") for m in shown])
        ax.invert_yaxis()
        ax.grid(False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATUS_COLOURS.values()]
        ax.legend(handles, ["green < 0.1", "amber 0.1-0.25", "red > 0.25"], title="CSI status",
                  loc="upper left", bbox_to_anchor=(1.01, 1))
        if len(hidden):
            ax.text(1.01, 0, f"Not shown: {len(hidden)} more features,\nhighest CSI {values.loc[hidden].max().max():.3f}",
                    transform=ax.transAxes, fontsize=9, color="#52514e", va="bottom")
        ax.set_title(f"Feature stability: CSI of the {len(shown)} most drifted features against training (as of {snapshotdate})",
                     loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(CHART_DIRECTORY + "feature_stability.png", dpi=150)
        plt.close(fig)

    print("charts saved to:", CHART_DIRECTORY, sorted(os.listdir(CHART_DIRECTORY)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    main(args.snapshotdate)
