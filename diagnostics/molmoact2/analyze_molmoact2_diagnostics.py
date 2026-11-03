"""Compare paired MolmoAct2 diagnostic runs and render layer-wise influence plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--run must be NAME=PATH")
    return name, Path(path)


def _read_run(path: Path) -> dict[int, dict[str, Any]]:
    manifest = path / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing diagnostic manifest: {manifest}")
    result: dict[int, dict[str, Any]] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        chunk = int(record["chunk_index"])
        record["arrays"] = dict(np.load(path / record["npz"]))
        result[chunk] = record
    return result


def _relative_l2(value: np.ndarray, baseline: np.ndarray) -> float:
    numerator = np.linalg.norm(value.astype(np.float64) - baseline.astype(np.float64))
    denominator = max(np.linalg.norm(baseline.astype(np.float64)), 1e-12)
    return float(numerator / denominator)


def _cosine(value: np.ndarray, baseline: np.ndarray) -> float:
    left = value.astype(np.float64).reshape(-1)
    right = baseline.astype(np.float64).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")


def _attention_rows(
    run_name: str,
    chunk: int,
    stage: str,
    values: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, metrics in values.items():
        for metric, value in metrics.items():
            rows.append(
                {
                    "run": run_name,
                    "chunk": chunk,
                    "stage": stage,
                    "layer": int(layer),
                    "region": "roi",
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["run", "chunk", "stage", "layer", "region", "metric", "value"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _matrix(
    rows: list[dict[str, Any]],
    run_names: list[str],
    *,
    stage: str,
    metric: str,
    region: str = "roi",
) -> tuple[np.ndarray, list[int]]:
    selected = [
        row
        for row in rows
        if row["stage"] == stage and row["metric"] == metric and row["region"] == region
    ]
    layers = sorted({int(row["layer"]) for row in selected})
    data = np.full((len(run_names), len(layers)), np.nan, dtype=np.float64)
    for run_idx, run in enumerate(run_names):
        for layer_idx, layer in enumerate(layers):
            samples = [
                float(row["value"])
                for row in selected
                if row["run"] == run and int(row["layer"]) == layer
            ]
            if samples:
                data[run_idx, layer_idx] = float(np.nanmean(samples))
    return data, layers


def _plot(rows: list[dict[str, Any]], run_names: list[str], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    panels = [
        ("vlm_attention", "language_to_roi", "roi", "VLM language → sign attention"),
        ("vlm_hidden", "relative_l2", "roi", "Sign-region activation Δ (relative L2)"),
        ("action_cross_attention", "action_to_roi", "roi", "Action query → sign attention"),
        ("action_hidden", "relative_l2", "all", "Action-expert activation Δ (relative L2)"),
    ]
    width = 1400
    panel_height = max(155, 76 + 24 * len(run_names))
    canvas = Image.new("RGB", (width, panel_height * len(panels)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    def colour(value: float, minimum: float, maximum: float) -> tuple[int, int, int]:
        if not np.isfinite(value):
            return 210, 210, 210
        fraction = 0.5 if maximum <= minimum else (value - minimum) / (maximum - minimum)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        anchors = ((68, 1, 84), (33, 145, 140), (253, 231, 37))
        scaled = fraction * 2
        left = min(int(scaled), 1)
        mix = scaled - left
        return tuple(
            int(anchors[left][channel] * (1 - mix) + anchors[left + 1][channel] * mix)
            for channel in range(3)
        )

    for panel_index, (stage, metric, region, title) in enumerate(panels):
        top = panel_index * panel_height
        data, layers = _matrix(rows, run_names, stage=stage, metric=metric, region=region)
        draw.text((10, top + 8), title, fill="black", font=font)
        if not layers:
            draw.text((180, top + 55), "not captured", fill=(90, 90, 90), font=font)
            continue
        finite = data[np.isfinite(data)]
        minimum = float(finite.min()) if finite.size else 0.0
        maximum = float(finite.max()) if finite.size else 1.0
        x0, y0 = 180, top + 35
        heatmap_width = width - x0 - 120
        cell_width = max(1.0, heatmap_width / len(layers))
        cell_height = 22
        for run_index, run_name in enumerate(run_names):
            row_y = y0 + run_index * cell_height
            draw.text((10, row_y + 5), run_name, fill="black", font=font)
            for layer_index, value in enumerate(data[run_index]):
                left = int(x0 + layer_index * cell_width)
                right = int(x0 + (layer_index + 1) * cell_width)
                draw.rectangle(
                    (left, row_y, max(left, right - 1), row_y + cell_height - 1),
                    fill=colour(float(value), minimum, maximum),
                )
        ticks = np.linspace(0, len(layers) - 1, min(8, len(layers)), dtype=int)
        label_y = y0 + len(run_names) * cell_height + 4
        for tick in ticks:
            x = int(x0 + (tick + 0.5) * cell_width)
            draw.text((x - 5, label_y), str(layers[tick]), fill="black", font=font)
        draw.text((width - 108, y0), f"max {maximum:.3g}", fill="black", font=font)
        draw.text(
            (width - 108, y0 + max(0, len(run_names) * cell_height - 12)),
            f"min {minimum:.3g}",
            fill="black",
            font=font,
        )
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=_parse_run, metavar="NAME=PATH")
    parser.add_argument("--baseline", default="clean")
    parser.add_argument("--output", type=Path, default=Path("diagnostic_comparison"))
    parser.add_argument(
        "--include-diverged-chunks",
        action="store_true",
        help=(
            "compare every common rollout chunk; by default only the first chunk is used because "
            "different policies cause later environment states to diverge"
        ),
    )
    args = parser.parse_args()

    runs = dict(args.run)
    if args.baseline not in runs:
        raise ValueError(f"Baseline {args.baseline!r} is not one of {sorted(runs)}")
    loaded = {name: _read_run(path) for name, path in runs.items()}
    common_chunks = sorted(set.intersection(*(set(run) for run in loaded.values())))
    if not common_chunks:
        raise RuntimeError("The supplied runs have no common chunk indices")
    if not args.include_diverged_chunks:
        common_chunks = common_chunks[:1]
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    action_summary: dict[str, list[float]] = {name: [] for name in runs if name != args.baseline}
    baseline = loaded[args.baseline]
    for run_name, run in loaded.items():
        for chunk in common_chunks:
            record = run[chunk]
            rows.extend(
                _attention_rows(run_name, chunk, "vlm_attention", record.get("vlm_attention", {}))
            )
            rows.extend(
                _attention_rows(
                    run_name,
                    chunk,
                    "action_cross_attention",
                    record.get("action_cross_attention", {}),
                )
            )
            base_arrays = baseline[chunk]["arrays"]
            arrays = record["arrays"]
            for key, stage, region in (
                ("vlm_hidden_roi", "vlm_hidden", "roi"),
                ("vlm_hidden_visual", "vlm_hidden", "visual"),
                ("vlm_hidden_language", "vlm_hidden", "language"),
                ("action_hidden", "action_hidden", "all"),
            ):
                if key not in arrays or key not in base_arrays:
                    continue
                count = min(arrays[key].shape[0], base_arrays[key].shape[0])
                for layer in range(count):
                    relative_l2 = (
                        0.0
                        if run_name == args.baseline
                        else _relative_l2(arrays[key][layer], base_arrays[key][layer])
                    )
                    cosine = (
                        1.0
                        if run_name == args.baseline
                        else _cosine(arrays[key][layer], base_arrays[key][layer])
                    )
                    rows.extend(
                        [
                            {
                                "run": run_name,
                                "chunk": chunk,
                                "stage": stage,
                                "layer": layer,
                                "region": region,
                                "metric": "relative_l2",
                                "value": relative_l2,
                            },
                            {
                                "run": run_name,
                                "chunk": chunk,
                                "stage": stage,
                                "layer": layer,
                                "region": region,
                                "metric": "cosine_similarity",
                                "value": cosine,
                            },
                        ]
                    )
            action_delta = (
                0.0
                if run_name == args.baseline
                else _relative_l2(arrays["actions"], base_arrays["actions"])
            )
            if run_name == args.baseline:
                continue
            action_summary[run_name].append(action_delta)
            rows.append(
                {
                    "run": run_name,
                    "chunk": chunk,
                    "stage": "actions",
                    "layer": -1,
                    "region": "all",
                    "metric": "relative_l2",
                    "value": action_delta,
                }
            )

    _write_csv(args.output / "layer_metrics.csv", rows)
    summary = {
        "baseline": args.baseline,
        "common_chunks": common_chunks,
        "include_diverged_chunks": args.include_diverged_chunks,
        "mean_action_relative_l2": {
            name: float(np.mean(values)) if values else None for name, values in action_summary.items()
        },
        "note": (
            "Attention is descriptive. Treat the action deltas from zero/replace ROI-KV runs as the "
            "causal result. Only the first chunk is compared by default because later rollout states "
            "diverge. ROI grid mapping is approximate unless explicit token indices were supplied."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(rows, list(runs), args.output / "layer_modality_influence.png")
    print(f"Wrote comparison to {args.output}")


if __name__ == "__main__":
    main()
