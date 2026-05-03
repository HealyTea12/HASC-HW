import argparse
import json
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OPS_PER_SAMPLE = {
    "BM_f1": 9,
    "BM_f2": 33,
}


def load_benchmark_data(path: str, simd_lanes: int) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    frame = pd.DataFrame(raw["benchmarks"])
    frame = frame[frame["run_type"] == "iteration"].copy()

    extracted = frame["name"].str.extract(r"^(?P<benchmark>[^/]+)/(?P<n>\d+)$")
    frame["benchmark"] = extracted["benchmark"]
    frame["base_benchmark"] = frame["benchmark"].str.replace(r"_v$", "", regex=True)
    frame["n"] = extracted["n"].astype(int)
    frame["real_time_ms"] = frame["real_time"] / 1_000_000.0
    frame["ops_per_sample"] = frame["base_benchmark"].map(OPS_PER_SAMPLE)
    frame["ops_per_sample"] = frame["ops_per_sample"].fillna(0)
    frame["lane_multiplier"] = 1
    frame.loc[frame["benchmark"].str.endswith("_v"), "lane_multiplier"] = simd_lanes
    frame["total_ops"] = frame["n"] * frame["ops_per_sample"] * frame["lane_multiplier"]
    frame["gflops"] = (
        frame["total_ops"] / (frame["real_time"] / 1_000_000_000.0) / 1_000_000_000.0
    )
    return frame


def plot_benchmark_data(frame: pd.DataFrame, output_path: str | None = None) -> None:
    sns.set_theme(style="whitegrid")
    fig, (ax_time, ax_flops) = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(
        data=frame, x="n", y="real_time_ms", hue="benchmark", marker="o", ax=ax_time
    )
    ax_time.set_xscale("log", base=2)
    ax_time.set_yscale("log")
    ax_time.set_xlabel("n")
    ax_time.set_ylabel("real time [ms]")
    ax_time.set_title("Runtime")

    xticks = sorted(frame["n"].unique())
    ax_time.set_xticks(xticks)
    ax_time.set_xticklabels([rf"$2^{{{int(math.log2(value))}}}$" for value in xticks])

    sns.lineplot(
        data=frame, x="n", y="gflops", hue="benchmark", marker="o", ax=ax_flops
    )
    ax_flops.set_xscale("log", base=2)
    ax_flops.set_yscale("log")
    ax_flops.set_xlabel("n")
    ax_flops.set_ylabel("estimated GFLOP/s")
    ax_flops.set_title("Estimated FLOP rate")
    ax_flops.set_xticks(xticks)
    ax_flops.set_xticklabels([rf"$2^{{{int(math.log2(value))}}}$" for value in xticks])

    sns.despine()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Google Benchmark JSON output.")
    parser.add_argument(
        "input",
        nargs="?",
        default="bench_res.txt",
        help="Path to the Google Benchmark JSON file",
    )
    parser.add_argument("--output", default="results.png", help="Output image file")
    parser.add_argument(
        "--show", action="store_true", help="Display the plot instead of saving it"
    )
    parser.add_argument(
        "--simd-lanes",
        type=int,
        default=1,
        help="Number of doubles processed by native_simd<double>; set this to your machine's vector width for vector benchmarks",
    )
    args = parser.parse_args()

    data = load_benchmark_data(args.input, args.simd_lanes)
    print(data[["benchmark", "n", "real_time_ms", "gflops"]].head())

    plot_benchmark_data(data, None if args.show else args.output)


if __name__ == "__main__":
    main()
