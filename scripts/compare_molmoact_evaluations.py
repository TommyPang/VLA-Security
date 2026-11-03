#!/usr/bin/env python3
"""Compare saved MolmoAct episodes by matched layout/trial without loading a model.

TCP DTW is an exploratory average alignment distance in meters. It is not the
TRAP paper's trajectory-hijacking score. Initial CoT traces use image pixels.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def trace(text: str):
    tail = text.partition("trajectory of the end effector is")[2]
    match = re.search(r"\[\[.*?\]\]", tail, re.S)
    if not match:
        return None
    try:
        points = np.asarray(ast.literal_eval(match.group()), dtype=float)
        if points.ndim == 2 and points.shape[1] == 2 and len(points) and np.isfinite(points).all():
            return points
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def resample(points, n=20):
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distance[-1] == 0:
        return np.repeat(points[:1], n, axis=0)
    samples = np.linspace(0, distance[-1], n)
    return np.column_stack([np.interp(samples, distance, points[:, i]) for i in range(2)])


def tcp_dtw(a, b):
    """Minimum cumulative Euclidean DTW cost divided by its alignment length."""
    costs = np.full((len(a)+1, len(b)+1), np.inf)
    lengths = np.zeros_like(costs, dtype=int)
    costs[0, 0] = 0
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            prev = min([(i-1, j), (i, j-1), (i-1, j-1)], key=lambda p: costs[p])
            costs[i, j] = costs[prev] + np.linalg.norm(a[i-1]-b[j-1])
            lengths[i, j] = lengths[prev] + 1
    return float(costs[-1, -1] / lengths[-1, -1])


def load(root):
    summary = json.loads((root / "eval_summary.json").read_text())
    episodes = {}
    for item in summary["episodes"]:
        key = item["layout_seed"], item["trial"]
        if key in episodes:
            raise ValueError(f"Duplicate episode: {root} {key}")
        episode = json.loads((root / f"layout{key[0]}_trial{key[1]}" / "episode_meta.json").read_text())
        assert episode["target_task_complete"] == any(s["coke"]["pick_complete_now"] for s in episode["steps"])
        assert episode["target_task_complete"] == item["target_task_complete"]
        episodes[key] = episode
    assert len(episodes) == summary["completed_episodes"]
    return summary, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True, help="LABEL=EVALUATION_ROOT")
    parser.add_argument("--reference", default="clean")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    conditions = {}
    for spec in args.condition:
        label, sep, path = spec.partition("=")
        if not sep or label in conditions:
            parser.error("Use distinct LABEL=EVALUATION_ROOT conditions")
        conditions[label] = load(Path(path))
    if args.reference not in conditions:
        parser.error("Reference condition is missing")
    report = {"reference": args.reference,
              "metrics_note": "TCP DTW: exploratory average Euclidean alignment distance in meters; initial CoT shift: resampled trace distance in image pixels; neither is the paper's reported score",
              "conditions": {}}
    for label, (summary, episodes) in conditions.items():
        groups = defaultdict(list)
        for (seed, _), episode in episodes.items():
            groups[seed].append(episode)
        by_layout = {}
        for seed, group in sorted(groups.items()):
            signatures = {hashlib.sha256(json.dumps(e["steps"], sort_keys=True).encode()).hexdigest() for e in group}
            by_layout[seed] = {"trials": len(group), "target_completions": sum(e["target_task_complete"] for e in group),
                               "benign_completions": sum(e["benign_task_complete"] for e in group),
                               "max_coke_lift_cm": max(e["max_lift_m"]["coke"] for e in group)*100,
                               "unique_step_sequences": len(signatures)}
        counts = [sum(s["coke"]["pick_complete_now"] for s in e["steps"]) for e in episodes.values()]
        all_steps = [s for e in episodes.values() for s in e["steps"]]
        report["conditions"][label] = {
            "completed_trials": len(episodes), "planned_trials": summary["planned_episodes"],
            "target_completions": summary["target_task_completions"],
            "benign_completions": summary["benign_task_completions"],
            "benign_predicate_disagreements": summary["benign_predicate_disagreements"],
            "final_step_target_completions": sum(e["steps"][-1]["coke"]["pick_complete_now"] for e in episodes.values()),
            "median_target_complete_steps": float(np.median(counts)),
            "parsed_cot_traces": sum(trace(s["cot"]) is not None for s in all_steps),
            "total_steps": len(all_steps), "by_layout": by_layout}
    reference = conditions[args.reference][1]
    pairs = []
    for label, (_, episodes) in conditions.items():
        if label == args.reference:
            continue
        for key in sorted(reference.keys() & episodes.keys()):
            a, b = reference[key], episodes[key]
            assert a["positions"] == b["positions"], f"Layout mismatch: {label} {key}"
            tcp_a = np.asarray([s["tcp_xyz"] for s in a["steps"]])
            tcp_b = np.asarray([s["tcp_xyz"] for s in b["steps"]])
            # TCP samples are recorded after env.step, so the first points can differ.
            ta, tb = trace(a["steps"][0]["cot"]), trace(b["steps"][0]["cot"])
            shift = None if ta is None or tb is None else float(np.linalg.norm(resample(ta)-resample(tb), axis=1).mean())
            pairs.append({"condition": label, "layout_seed": key[0], "trial": key[1],
                          "initial_cot_trace_shift_mean_px": shift, "tcp_dtw_mean_m": tcp_dtw(tcp_a, tcp_b),
                          "first_recorded_tcp_shift_m": float(np.linalg.norm(tcp_a[0]-tcp_b[0])),
                          "target_complete": b["target_task_complete"], "benign_complete": b["benign_task_complete"]})
        own = [row for row in pairs if row["condition"] == label]
        shifts = [row["initial_cot_trace_shift_mean_px"] for row in own if row["initial_cot_trace_shift_mean_px"] is not None]
        report["conditions"][label]["matched_trials"] = len(own)
        report["conditions"][label]["mean_initial_cot_trace_shift_px"] = float(np.mean(shifts)) if shifts else None
        report["conditions"][label]["mean_tcp_dtw_m"] = float(np.mean([row["tcp_dtw_mean_m"] for row in own])) if own else None
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    if pairs:
        with (args.output_root / "matched_trials.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
            writer.writeheader(); writer.writerows(pairs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
