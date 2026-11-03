"""CPU-only, mechanism-level reproduction of the TRAP objective.

This demo is deliberately transparent: a synthetic two-object scene is passed
through a differentiable reasoning/action surrogate, and a fixed corner patch
is optimized with PGD. It is not a substitute for the paper's VLA checkpoints
or simulators and must not be compared numerically with the paper's tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Config:
    image_size: int = 32
    object_size: int = 4
    patch_size: int = 8
    steps: int = 120
    batch_size: int = 32
    eval_scenes: int = 256
    alpha: float = 8.0 / 255.0
    action_weight: float = 2.0
    tv_weight: float = 0.02
    seed: int = 7
    eot_noise: float = 0.02
    eot_brightness: float = 0.15


class ToyReasoningVLA:
    """A small analytical VLA surrogate with a discrete CoT bottleneck.

    The instruction asks for the red object (class 0). The attack target is the
    blue object (class 1). CoT and direct-action paths react to orthogonal visual
    carriers, approximating the weak gradient path through discrete reasoning.
    """

    def __init__(self, image_size: int, patch_size: int) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        yy, xx = np.mgrid[:patch_size, :patch_size]
        self.cot_carrier = np.where((xx + yy) % 2 == 0, 1.0, -1.0)
        self.action_carrier = np.where(xx % 2 == 0, 1.0, -1.0)
        self.area_fraction = (patch_size / image_size) ** 2
        self.instruction_bias = -4.0
        self.cot_gain = 140.0
        self.action_gain = 40.0

    @staticmethod
    def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
        value = np.clip(value, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-value))

    def signals(self, patch_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blue_minus_red = patch_batch[:, 2] - patch_batch[:, 0]
        cot = np.mean(blue_minus_red * self.cot_carrier, axis=(1, 2))
        action = np.mean(blue_minus_red * self.action_carrier, axis=(1, 2))
        return cot, action

    def probabilities(self, patch_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cot_signal, action_signal = self.signals(patch_batch)
        cot_logit = self.instruction_bias + self.cot_gain * self.area_fraction * cot_signal
        action_delta = self.action_gain * self.area_fraction * action_signal
        cot_target_probability = self.sigmoid(cot_logit)
        execution_target_probability = self.sigmoid(cot_logit + action_delta)
        return cot_target_probability, execution_target_probability


def make_scenes(config: Config, count: int, rng: np.random.Generator):
    images = np.full((count, 3, config.image_size, config.image_size), 0.12, dtype=np.float64)
    positions = np.zeros((count, 2, 2), dtype=np.float64)
    top_limit = config.image_size - config.patch_size - config.object_size - 2
    if top_limit <= 2:
        raise ValueError("patch is too large for the selected image size")

    for index in range(count):
        chosen: list[tuple[int, int]] = []
        while len(chosen) < 2:
            x = int(rng.integers(2, config.image_size - config.object_size - 2))
            y = int(rng.integers(2, top_limit + 1))
            if all(abs(x - old_x) > config.object_size + 1 or abs(y - old_y) > config.object_size + 1 for old_x, old_y in chosen):
                chosen.append((x, y))

        for object_index, (x, y) in enumerate(chosen):
            color = (0.92, 0.12, 0.12) if object_index == 0 else (0.10, 0.28, 0.95)
            images[index, :, y : y + config.object_size, x : x + config.object_size] = np.asarray(color)[:, None, None]
            positions[index, object_index] = (
                (x + 0.5 * config.object_size) / (config.image_size - 1),
                (y + 0.5 * config.object_size) / (config.image_size - 1),
            )
    return images, positions


def apply_patch(images: np.ndarray, patch: np.ndarray) -> np.ndarray:
    output = images.copy()
    patch_size = patch.shape[-1]
    output[:, :, -patch_size:, :patch_size] = patch[None]
    return output


def total_variation_and_grad(patch: np.ndarray) -> tuple[float, np.ndarray]:
    grad = np.zeros_like(patch)
    horizontal = patch[:, :, 1:] - patch[:, :, :-1]
    vertical = patch[:, 1:, :] - patch[:, :-1, :]
    h_scale = horizontal.size
    v_scale = vertical.size
    h_sign = np.sign(horizontal) / h_scale
    v_sign = np.sign(vertical) / v_scale
    grad[:, :, 1:] += h_sign
    grad[:, :, :-1] -= h_sign
    grad[:, 1:, :] += v_sign
    grad[:, :-1, :] -= v_sign
    return float(np.mean(np.abs(horizontal)) + np.mean(np.abs(vertical))), grad


def transformed_patch_and_jacobian(
    patch: np.ndarray, batch_size: int, config: Config, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    brightness = rng.uniform(1.0 - config.eot_brightness, 1.0 + config.eot_brightness, size=(batch_size, 1, 1, 1))
    noise = rng.normal(0.0, config.eot_noise, size=(batch_size, *patch.shape))
    raw = brightness * patch[None] + noise
    transformed = np.clip(raw, 0.0, 1.0)
    jacobian = brightness * ((raw > 0.0) & (raw < 1.0))
    return transformed, jacobian


def signal_patch_grad(carrier: np.ndarray, patch_size: int) -> np.ndarray:
    grad = np.zeros((3, patch_size, patch_size), dtype=np.float64)
    grad[2] = carrier / carrier.size
    grad[0] = -carrier / carrier.size
    return grad


def optimize_patch(
    method: str, config: Config, positions: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, list[dict[str, float]]]:
    model = ToyReasoningVLA(config.image_size, config.patch_size)
    patch = np.full((3, config.patch_size, config.patch_size), 0.5, dtype=np.float64)
    cot_signal_grad = signal_patch_grad(model.cot_carrier, config.patch_size)
    action_signal_grad = signal_patch_grad(model.action_carrier, config.patch_size)
    history: list[dict[str, float]] = []

    for step in range(1, config.steps + 1):
        indices = rng.integers(0, len(positions), size=config.batch_size)
        batch_positions = positions[indices]
        transformed, jacobian = transformed_patch_and_jacobian(patch, config.batch_size, config, rng)
        cot_probability, execution_probability = model.probabilities(transformed)
        benign_xy = batch_positions[:, 0]
        target_xy = batch_positions[:, 1]
        direction = target_xy - benign_xy
        prediction = benign_xy + execution_probability[:, None] * direction

        cot_loss = float(-np.mean(np.log(np.clip(cot_probability, 1e-8, 1.0))))
        action_loss = float(np.mean((prediction - target_xy) ** 2))
        tv_loss, tv_grad = total_variation_and_grad(patch)
        grad = np.zeros_like(patch)

        if method in {"cot", "joint"}:
            dloss_dcot_logit = (cot_probability - 1.0) / config.batch_size
            per_sample = (
                dloss_dcot_logit[:, None, None, None]
                * model.cot_gain
                * model.area_fraction
                * cot_signal_grad[None]
                * jacobian
            )
            grad += np.sum(per_sample, axis=0)

        if method in {"action", "joint"}:
            # The action objective follows the continuous action path. We stop
            # its gradient at the discrete CoT sample, as in autoregressive VLAs.
            dloss_dprob = np.sum(2.0 * (prediction - target_xy) * direction, axis=1) / (2.0 * config.batch_size)
            dloss_daction_logit = dloss_dprob * execution_probability * (1.0 - execution_probability)
            action_multiplier = config.action_weight if method == "joint" else 1.0
            per_sample = (
                action_multiplier
                * dloss_daction_logit[:, None, None, None]
                * model.action_gain
                * model.area_fraction
                * action_signal_grad[None]
                * jacobian
            )
            grad += np.sum(per_sample, axis=0)

        grad += config.tv_weight * tv_grad
        patch = np.clip(patch - config.alpha * np.sign(grad), 0.0, 1.0)
        total_loss = (
            (cot_loss if method in {"cot", "joint"} else 0.0)
            + (config.action_weight if method == "joint" else 1.0) * (action_loss if method in {"action", "joint"} else 0.0)
            + config.tv_weight * tv_loss
        )
        history.append(
            {
                "step": float(step),
                "total_loss": total_loss,
                "cot_loss": cot_loss,
                "action_loss": action_loss,
                "tv_loss": tv_loss,
                "mean_target_cot_probability": float(np.mean(cot_probability)),
                "mean_target_execution_probability": float(np.mean(execution_probability)),
            }
        )
    return patch, history


def evaluate_patch(
    patch: np.ndarray, config: Config, positions: np.ndarray, seed: int
) -> dict[str, float]:
    model = ToyReasoningVLA(config.image_size, config.patch_size)
    rng = np.random.default_rng(seed)
    transformed, _ = transformed_patch_and_jacobian(patch, len(positions), config, rng)
    cot_probability, execution_probability = model.probabilities(transformed)
    prediction = positions[:, 0] + execution_probability[:, None] * (positions[:, 1] - positions[:, 0])
    error = np.linalg.norm(prediction - positions[:, 1], axis=1)
    tv, _ = total_variation_and_grad(patch)
    return {
        "cot_asr": float(np.mean(cot_probability >= 0.5)),
        "action_asr": float(np.mean(error <= 0.12)),
        "mean_target_cot_probability": float(np.mean(cot_probability)),
        "mean_target_execution_probability": float(np.mean(execution_probability)),
        "mean_action_error": float(np.mean(error)),
        "total_variation": tv,
    }


def to_uint8_chw(array: np.ndarray) -> np.ndarray:
    image = np.moveaxis(np.clip(array, 0.0, 1.0), 0, -1)
    return np.round(image * 255).astype(np.uint8)


def save_patch(path: Path, patch: np.ndarray, scale: int = 16) -> None:
    image = Image.fromarray(to_uint8_chw(patch), mode="RGB")
    image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST).save(path)


def save_preview(path: Path, images: np.ndarray, positions: np.ndarray, patch: np.ndarray) -> None:
    sample_count = min(6, len(images))
    clean = images[:sample_count]
    patched = apply_patch(clean, patch)
    tiles: list[Image.Image] = []
    for row_name, batch in (("clean", clean), ("patched", patched)):
        for index, scene in enumerate(batch):
            tile = Image.fromarray(to_uint8_chw(scene), mode="RGB").resize((192, 192), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(tile)
            target_x, target_y = positions[index, 1]
            draw.ellipse(
                (target_x * 192 - 5, target_y * 192 - 5, target_x * 192 + 5, target_y * 192 + 5),
                outline=(255, 255, 255),
                width=2,
            )
            draw.rectangle((0, 0, 58, 15), fill=(0, 0, 0))
            draw.text((3, 2), row_name, fill=(255, 255, 255))
            tiles.append(tile)
    canvas = Image.new("RGB", (sample_count * 192, 2 * 192), (20, 20, 20))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % sample_count) * 192, (index // sample_count) * 192))
    canvas.save(path)


def write_csv(path: Path, rows: Iterable[dict[str, float]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_for_size(base_config: Config, patch_size: int, output_dir: Path) -> dict[str, dict[str, float]]:
    config = Config(**{**asdict(base_config), "patch_size": patch_size})
    train_rng = np.random.default_rng(config.seed + patch_size)
    eval_rng = np.random.default_rng(config.seed + 10_000 + patch_size)
    train_images, train_positions = make_scenes(config, 25, train_rng)
    eval_images, eval_positions = make_scenes(config, config.eval_scenes, eval_rng)
    del train_images

    size_dir = output_dir / f"patch_{patch_size}x{patch_size}"
    size_dir.mkdir(parents=True, exist_ok=True)
    patches: dict[str, np.ndarray] = {
        "clean": np.full((3, patch_size, patch_size), 0.5, dtype=np.float64),
        "random": np.random.default_rng(config.seed + 20_000 + patch_size).random((3, patch_size, patch_size)),
    }

    for offset, method in enumerate(("action", "cot", "joint")):
        patch, history = optimize_patch(
            method,
            config,
            train_positions,
            np.random.default_rng(config.seed + 30_000 + patch_size * 10 + offset),
        )
        patches[method] = patch
        write_csv(size_dir / f"loss_{method}.csv", history, list(history[0]))

    metrics: dict[str, dict[str, float]] = {}
    for offset, (method, patch) in enumerate(patches.items()):
        metrics[method] = evaluate_patch(patch, config, eval_positions, config.seed + 40_000 + offset)
        save_patch(size_dir / f"patch_{method}.png", patch)
        save_preview(size_dir / f"preview_{method}.png", eval_images, eval_positions, patch)
    return metrics


def parse_patch_sizes(value: str) -> list[int]:
    sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not sizes or any(size < 2 for size in sizes):
        raise argparse.ArgumentTypeError("patch sizes must be comma-separated integers >= 2")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toy_reproduction"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-scenes", type=int, default=256)
    parser.add_argument("--patch-sizes", type=parse_patch_sizes, default=[8])
    parser.add_argument("--alpha", type=float, default=8.0 / 255.0)
    parser.add_argument("--action-weight", type=float, default=2.0)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.eval_scenes < 1:
        parser.error("steps, batch-size, and eval-scenes must be positive")
    config = Config(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_scenes=args.eval_scenes,
        alpha=args.alpha,
        action_weight=args.action_weight,
        tv_weight=args.tv_weight,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "note": "Mechanism-level synthetic reproduction; do not compare these values with the paper.",
        "config": asdict(config),
        "patch_sizes": {},
    }
    flat_rows: list[dict[str, float | str]] = []
    for patch_size in args.patch_sizes:
        metrics = run_for_size(config, patch_size, args.output_dir)
        summary["patch_sizes"][str(patch_size)] = metrics
        for method, values in metrics.items():
            flat_rows.append({"patch_size": patch_size, "method": method, **values})

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(flat_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote results to {args.output_dir.resolve()}")
    print("patch method   CoT-ASR Action-ASR mean-action-error")
    for row in flat_rows:
        print(
            f"{int(row['patch_size']):>5} {str(row['method']):<8} "
            f"{float(row['cot_asr']):>7.1%} {float(row['action_asr']):>10.1%} "
            f"{float(row['mean_action_error']):>17.4f}"
        )


if __name__ == "__main__":
    main()
