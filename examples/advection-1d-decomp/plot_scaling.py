#!/usr/bin/env python3
"""Plot scaling results from CSV file."""

import sys
import csv
import matplotlib.pyplot as plt


def load_csv(filename):
    """Load CSV and return list of dicts."""
    with open(filename) as f:
        reader = csv.DictReader(f)
        return [
            {
                "num_threads": int(row["num_threads"]),
                "num_partitions": int(row["num_partitions"]),
                "num_zones": int(row["num_zones"]),
                "use_flux_buffer": int(row["use_flux_buffer"]),
                "Mzps": float(row["Mzps"]),
            }
            for row in reader
        ]


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_scaling.py <scaling_results.csv>")
        sys.exit(1)

    data = load_csv(sys.argv[1])

    # Plot 1: Thread scaling at nz=1e8
    fused_threads = [(r["num_threads"], r["Mzps"]) for r in data
                     if r["num_zones"] == 100000000
                     and r["num_threads"] == r["num_partitions"]
                     and r["use_flux_buffer"] == 0]
    unfused_threads = [(r["num_threads"], r["Mzps"]) for r in data
                       if r["num_zones"] == 100000000
                       and r["num_threads"] == r["num_partitions"]
                       and r["use_flux_buffer"] == 1]
    fused_threads.sort()
    unfused_threads.sort()

    if len(fused_threads) >= 2 or len(unfused_threads) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))

        if fused_threads:
            x = [p[0] for p in fused_threads]
            y = [p[1] for p in fused_threads]
            ax.loglog(x, y, "o-", markersize=8, linewidth=2, label="Fused (no flux buffer)")

        if unfused_threads:
            x = [p[0] for p in unfused_threads]
            y = [p[1] for p in unfused_threads]
            ax.loglog(x, y, "s--", markersize=8, linewidth=2, label="Unfused (with flux buffer)")

        ax.set_xlabel("num_threads = num_partitions", fontsize=12)
        ax.set_ylabel("Mzps", fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title("Thread scaling (num_zones=1e8)", fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig("scaling_threads.pdf")
        print("Saved scaling_threads.pdf")
        plt.close()

    # Plot 2: Problem size scaling (sequential)
    fused_zones = [(r["num_zones"], r["Mzps"]) for r in data
                   if r["num_threads"] == 0
                   and r["num_partitions"] == 1
                   and r["use_flux_buffer"] == 0]
    unfused_zones = [(r["num_zones"], r["Mzps"]) for r in data
                     if r["num_threads"] == 0
                     and r["num_partitions"] == 1
                     and r["use_flux_buffer"] == 1]
    fused_zones.sort()
    unfused_zones.sort()

    if len(fused_zones) >= 2 or len(unfused_zones) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))

        if fused_zones:
            x = [p[0] for p in fused_zones]
            y = [p[1] for p in fused_zones]
            ax.loglog(x, y, "o-", markersize=8, linewidth=2, label="Fused (no flux buffer)")

        if unfused_zones:
            x = [p[0] for p in unfused_zones]
            y = [p[1] for p in unfused_zones]
            ax.loglog(x, y, "s--", markersize=8, linewidth=2, label="Unfused (with flux buffer)")

        ax.set_xlabel("num_zones", fontsize=12)
        ax.set_ylabel("Mzps", fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title("Problem size scaling (sequential, np=1)", fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig("scaling_zones.pdf")
        print("Saved scaling_zones.pdf")
        plt.close()


if __name__ == "__main__":
    main()
