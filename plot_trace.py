"""
Usage:
    python plot_trace.py trace.csv [--run 0] [--x superstep|wall_ms|gvt]
                                   [--downsample N] [-o output.png]
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Trace CSV produced by --trace")
    ap.add_argument("--run", type=int, default=0,
                    help="Which run index to plot (default: 0)")
    ap.add_argument("--x", choices=["superstep", "wall_ms", "gvt"],
                    default="superstep", help="X-axis variable")
    ap.add_argument("--downsample", type=int, default=1,
                    help="Average every N supersteps into one bar (default: 1)")
    ap.add_argument("-o", "--output", default=None,
                    help="Save figure to file instead of showing")
    ap.add_argument("--n-lps", type=int, default=None,
                    help="Override n_lps for thread-cycle normalisation "
                         "(auto-detected from data if omitted)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if "run" in df.columns:
        df = df[df["run"] == args.run].reset_index(drop=True)
    if len(df) == 0:
        print(f"No data for run {args.run}", file=sys.stderr)
        sys.exit(1)

    df["useful"] = df["processed"] - df["rolledback"]
    if args.n_lps is None:
        row0 = df.iloc[0]
        total_tc = row0["processed"] + row0["inactive"]
        ec = row0["exec_cycles"]
        n_lps = int(round(total_tc / ec)) if ec > 0 else 1
        if n_lps == 0:
            n_lps = 1
    else:
        n_lps = args.n_lps

    df["total_thread_cycles"] = n_lps * df["exec_cycles"]

    if args.downsample > 1:
        n = args.downsample
        group = df.index // n
        agg = {
            "superstep": "first",
            "wall_ms": "mean",
            "gvt": "mean",
            "exec_cycles": "sum",
            "processed": "sum",
            "rolledback": "sum",
            "committed": "sum",
            "inactive": "sum",
            "inac_no_event": "sum",
            "inac_window": "sum",
            "inac_handle_fail": "sum",
            "inac_eq_full": "sum",
            "inac_sq_full": "sum",
            "inac_amq_full": "sum",
            "useful": "sum",
            "total_thread_cycles": "sum",
        }
        df = df.groupby(group).agg(agg).reset_index(drop=True)

    x = df[args.x].values
    useful = df["useful"].values.astype(float)
    rolledback = df["rolledback"].values.astype(float)
    inactive = df["inactive"].values.astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    width = np.diff(x, prepend=x[0] - 1).clip(min=0.8) * 0.8 if args.x != "superstep" \
        else 0.8

    ax.bar(x, useful,     width=width, label="Useful (processed - rolledback)",
           color="#2ecc71")
    ax.bar(x, rolledback, width=width, bottom=useful,
           label="Rolled back", color="#e74c3c")
    ax.bar(x, inactive,   width=width, bottom=useful + rolledback,
           label="Inactive", color="#95a5a6")

    ax.set_ylabel("Thread-cycles per superstep")
    ax.set_title("GPUTW Optimistic Sync — Per-Superstep Thread Activity")
    ax.legend(loc="upper right")

    ax2 = axes[1]
    inac_cols = [
        ("inac_no_event",    "#f39c12", "No event"),
        ("inac_window",      "#9b59b6", "Outside window"),
        ("inac_handle_fail", "#e67e22", "Handle fail"),
        ("inac_eq_full",     "#1abc9c", "EQ full"),
        ("inac_sq_full",     "#3498db", "SQ full"),
        ("inac_amq_full",    "#e91e63", "AMQ full"),
    ]
    bottom = np.zeros(len(df))
    for col, color, label in inac_cols:
        vals = df[col].values.astype(float)
        if vals.sum() > 0:
            ax2.bar(x, vals, width=width, bottom=bottom,
                    label=label, color=color)
            bottom += vals

    ax2.set_ylabel("Inactive thread-cycles")
    ax2.set_xlabel(args.x)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
