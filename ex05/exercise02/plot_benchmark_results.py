#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "benchmark_results.csv"
SVG_PATH = ROOT / "benchmark_comparison.svg"


def read_rows(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["threads"] = int(row["threads"])
        for key in (
            "sequential_time_s",
            "cppthreads_time_s",
            "openmp_time_s",
            "cppthreads_gflops",
            "openmp_gflops",
        ):
            row[key] = float(row[key])
    return sorted(rows, key=lambda row: row["threads"])


def polyline(points, color):
    pairs = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{pairs}" fill="none" stroke="{color}" '
        'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def marker(x, y, color):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>'


def text(x, y, value, size=12, anchor="middle", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="#202124">{value}</text>'
    )


def axis_panel(rows, x, y, width, height, title, ylabel, series):
    threads = [row["threads"] for row in rows]
    xmin, xmax = min(threads), max(threads)
    ymax = max(max(row[key] for row in rows) for _, key, _ in series) * 1.12
    ymin = 0.0

    def sx(t):
        if xmax == xmin:
            return x + width / 2
        return x + (t - xmin) / (xmax - xmin) * width

    def sy(v):
        return y + height - (v - ymin) / (ymax - ymin) * height

    out = [
        text(x + width / 2, y - 28, title, size=17, weight="700"),
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" stroke="#5f6368" stroke-width="1.4"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" stroke="#5f6368" stroke-width="1.4"/>',
        text(x + width / 2, y + height + 48, "Threads", size=13),
        f'<text x="{x - 46}" y="{y + height / 2}" font-size="13" text-anchor="middle" '
        f'fill="#202124" transform="rotate(-90 {x - 46},{y + height / 2})">{ylabel}</text>',
    ]

    for t in threads:
        px = sx(t)
        out.append(f'<line x1="{px:.1f}" y1="{y + height}" x2="{px:.1f}" y2="{y + height + 6}" stroke="#5f6368"/>')
        out.append(text(px, y + height + 24, str(t), size=12))

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = ymin + frac * (ymax - ymin)
        py = sy(value)
        out.append(f'<line x1="{x}" y1="{py:.1f}" x2="{x + width}" y2="{py:.1f}" stroke="#e0e3e7" stroke-width="1"/>')
        out.append(text(x - 9, py + 4, f"{value:.2g}", size=11, anchor="end"))

    for name, key, color in series:
        pts = [(sx(row["threads"]), sy(row[key])) for row in rows]
        out.append(polyline(pts, color))
        for px, py in pts:
            out.append(marker(px, py, color))
        out.append(
            f'<rect x="{x + width - 130}" y="{y + 18 + 24 * series.index((name, key, color))}" '
            f'width="14" height="14" fill="{color}"/>'
        )
        out.append(text(x + width - 108, y + 30 + 24 * series.index((name, key, color)), name, size=12, anchor="start"))

    return "\n".join(out)


def build_svg(rows):
    colors = {
        "Sequential": "#5f6368",
        "cppthreads": "#1a73e8",
        "OpenMP": "#d93025",
    }
    runtime_series = [
        ("Sequential", "sequential_time_s", colors["Sequential"]),
        ("cppthreads", "cppthreads_time_s", colors["cppthreads"]),
        ("OpenMP", "openmp_time_s", colors["OpenMP"]),
    ]
    gflops_series = [
        ("cppthreads", "cppthreads_gflops", colors["cppthreads"]),
        ("OpenMP", "openmp_gflops", colors["OpenMP"]),
    ]

    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="520" viewBox="0 0 1100 520">',
        '<rect width="1100" height="520" fill="#ffffff"/>',
        text(550, 36, "n-body benchmark: cppthreads vs OpenMP (n = 8192, B = 128)", size=21, weight="700"),
        axis_panel(rows, 85, 95, 410, 310, "Runtime", "Seconds", runtime_series),
        axis_panel(rows, 620, 95, 410, 310, "Throughput", "GFLOP/s", gflops_series),
        text(550, 495, f"Source: {CSV_PATH.name}. cppthreads runtime is max(elapsed[]) over worker threads.", size=12),
        "</svg>",
    ]
    return "\n".join(body)


def main():
    rows = read_rows(CSV_PATH)
    SVG_PATH.write_text(build_svg(rows))
    print(f"wrote {SVG_PATH}")


if __name__ == "__main__":
    main()
