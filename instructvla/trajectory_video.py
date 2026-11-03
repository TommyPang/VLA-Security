"""Render saved SimplerEnv episodes as annotated GIFs without loading the VLA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _captioned_frame(frame_path: Path, episode: dict, index: int, total: int, font: ImageFont.ImageFont) -> np.ndarray:
    with Image.open(frame_path) as source:
        frame = source.convert("RGB")
    width, height = frame.size
    canvas = Image.new("RGB", (width, height + 84), (20, 24, 31))
    canvas.paste(frame, (0, 0))
    draw = ImageDraw.Draw(canvas)
    instruction = episode.get("instruction", "")
    condition = "patched" if episode.get("patched") else "clean"
    draw.text((10, height + 5), f"{condition} | seed {episode.get('seed')} | {instruction} | frame {index + 1}/{total}", font=font, fill=(255, 255, 255))

    steps = episode.get("steps", [])
    current = steps[index] if index < len(steps) else {}
    reasoning = str(current.get("assistant_reasoning", "")).replace("\n", " ").strip()
    if len(reasoning) > 72:
        reasoning = reasoning[:69] + "..."
    draw.text((10, height + 31), f"CoT: {reasoning}", font=font, fill=(225, 232, 241))

    # Frame i is the observation before action i; its visible object state is
    # the state recorded after action i-1.
    previous = steps[index - 1] if index > 0 and index - 1 < len(steps) else {}
    coke = previous.get("coke", {})
    orange = previous.get("orange", {})
    state = (
        f"Coke grasp/lift: {int(bool(coke.get('grasped')))} / {int(bool(coke.get('lifted')))}"
        f"    Orange grasp/lift: {int(bool(orange.get('grasped')))} / {int(bool(orange.get('lifted')))}"
    )
    draw.text((10, height + 57), state, font=font, fill=(255, 214, 130) if orange.get("lifted") else (196, 209, 225))
    return np.asarray(canvas)


def render_episode_gif(episode_dir: str | Path, *, episode: dict | None = None, fps: float = 3.0) -> Path:
    """Write episode.gif from step PNGs and episode metadata."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    episode_dir = Path(episode_dir)
    if episode is None:
        episode = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
    frames = sorted((episode_dir / "frames").glob("step_*.png"))
    if not frames:
        raise FileNotFoundError(f"No trajectory frames in {episode_dir / 'frames'}")
    gif_path = episode_dir / "episode.gif"
    font = _font(15)
    imageio.mimsave(
        gif_path,
        [_captioned_frame(path, episode, index, len(frames), font) for index, path in enumerate(frames)],
        format="GIF",
        fps=fps,
        loop=0,
    )
    return gif_path


def write_run_index(run_dir: str | Path) -> Path:
    """List each episode's outcome and GIF in a browsable Markdown table."""
    run_dir = Path(run_dir)
    lines = [
        "# InstructVLA trajectory GIFs",
        "",
        "Orange and coke columns report a physical lift, not a grasp alone.",
        "",
        "| Condition | Seed | Coke lift | Orange lift | First CoT | Trajectory |",
        "| --- | ---: | :---: | :---: | --- | --- |",
    ]
    for meta_path in sorted(run_dir.glob("*/episode_meta.json")):
        episode_dir = meta_path.parent
        episode = json.loads(meta_path.read_text(encoding="utf-8"))
        condition = episode_dir.name.rsplit("_seed", 1)[0]
        ever = episode.get("ever", {})
        first_cot = (episode.get("steps") or [{}])[0].get("assistant_reasoning", "")
        first_cot = str(first_cot).replace("|", "\\|").replace("\n", " ")
        target = "episode.gif" if (episode_dir / "episode.gif").exists() else "frames/"
        lines.append(
            f"| {condition} | {episode.get('seed')} | "
            f"{'yes' if ever.get('coke_lifted') else 'no'} | "
            f"{'yes' if ever.get('orange_lifted') else 'no'} | "
            f"{first_cot} | [view]({episode_dir.name}/{target}) |"
        )
    index_path = run_dir / "trajectory_index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dirs", type=Path, nargs="+", help="Episode folders or run folders containing episode folders")
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()
    for root in args.episode_dirs:
        is_episode = (root / "episode_meta.json").exists()
        episodes = [root] if is_episode else sorted(root.glob("*/episode_meta.json"))
        for item in episodes:
            episode_dir = item if item.is_dir() else item.parent
            print(render_episode_gif(episode_dir, fps=args.fps), flush=True)
        if not is_episode:
            print(write_run_index(root), flush=True)


if __name__ == "__main__":
    main()
