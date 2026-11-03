"""Causal diagnostics for MolmoAct2 LIBERO rollouts.

This module is intentionally installed as a monkeypatch.  ``lerobot-eval`` owns
policy construction and the rollout loop, so a patch at policy construction is
the least invasive way to observe the real inference path.

The recorder captures, for every sampled action chunk:

* VLM hidden-state summaries for the sign ROI, all visual tokens, and all
  non-visual tokens at every language-model layer;
* VLM self-attention mass into and out of the sign ROI;
* action-expert cross-attention mass assigned to the sign ROI;
* action-expert block activations and the generated action chunk; and
* the per-layer VLM K/V values for the ROI, suitable for a paired causal run.

It can additionally zero the sign-region K/V cache or replace it with a cache
captured from a reference run.  These interventions occur at the exact
VLM-to-action-expert interface used by MolmoAct2.

The implementation targets the public Hugging Face/LeRobot MolmoAct2 model and
checks the required interfaces at runtime.  Diagnostics are opt-in and normal
evaluation is unchanged when ``MOLMO_DIAGNOSTICS`` is not set.
"""

from __future__ import annotations

import json
import math
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


_TEXT_ROI_NORM: tuple[float, float, float, float] | None = None
_TEXT_ROI_SOURCE = "unset"


def set_text_roi(
    roi: tuple[float, float, float, float] | None,
    *,
    source: str = "attack-difference",
) -> None:
    """Set the sign bounding box as normalized ``(x0, y0, x1, y1)`` values."""

    global _TEXT_ROI_NORM, _TEXT_ROI_SOURCE
    _TEXT_ROI_NORM = roi
    _TEXT_ROI_SOURCE = source


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_roi(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    fields = [float(item.strip()) for item in value.split(",")]
    if len(fields) != 4:
        raise ValueError("MOLMO_TEXT_ROI must be x0,y0,x1,y1 in normalized image coordinates")
    x0, y0, x1, y1 = fields
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"Invalid normalized MOLMO_TEXT_ROI: {fields}")
    return x0, y0, x1, y1


@dataclass(frozen=True)
class DiagnosticConfig:
    output_dir: Path
    variant: str
    capture_every: int
    max_chunks: int
    intervention: str
    reference_dir: Path | None
    save_roi_kv: bool

    @classmethod
    def from_env(cls) -> "DiagnosticConfig":
        intervention = os.environ.get("MOLMO_DIAG_INTERVENTION", "none").strip().lower()
        if intervention not in {"none", "zero_roi_kv", "replace_roi_kv"}:
            raise ValueError(
                "MOLMO_DIAG_INTERVENTION must be none, zero_roi_kv, or replace_roi_kv"
            )
        reference = os.environ.get("MOLMO_DIAG_REFERENCE_DIR", "").strip()
        if intervention == "replace_roi_kv" and not reference:
            raise ValueError("replace_roi_kv requires MOLMO_DIAG_REFERENCE_DIR")
        capture_every = int(os.environ.get("MOLMO_DIAG_CAPTURE_EVERY", "1"))
        max_chunks = int(os.environ.get("MOLMO_DIAG_MAX_CHUNKS", "-1"))
        if capture_every < 1:
            raise ValueError("MOLMO_DIAG_CAPTURE_EVERY must be at least 1")
        if intervention != "none" and (capture_every != 1 or max_chunks not in {-1, 1}):
            raise ValueError(
                "Causal interventions must cover either the first action chunk or the whole "
                "rollout: use MOLMO_DIAG_CAPTURE_EVERY=1 and "
                "MOLMO_DIAG_MAX_CHUNKS=1 or -1"
            )
        return cls(
            output_dir=Path(os.environ.get("MOLMO_DIAG_DIR", "results/molmoact2_diagnostics")),
            variant=os.environ.get("ATTACK_VARIANT", "unknown"),
            capture_every=capture_every,
            max_chunks=max_chunks,
            intervention=intervention,
            reference_dir=Path(reference) if reference else None,
            save_roi_kv=_env_bool("MOLMO_DIAG_SAVE_ROI_KV", True),
        )


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(device="cpu", dtype=torch.float32).numpy()


def _mean_selected(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=hidden.device, dtype=torch.bool)
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denominator


def _attention_mass(attention: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor) -> float:
    """Mean, over selected queries and heads, of probability mass on selected keys."""

    values: list[torch.Tensor] = []
    for batch_idx in range(attention.shape[0]):
        q_idx = query_mask[batch_idx].nonzero(as_tuple=False).flatten()
        k_idx = key_mask[batch_idx].nonzero(as_tuple=False).flatten()
        if q_idx.numel() == 0 or k_idx.numel() == 0:
            continue
        selected = attention[batch_idx].index_select(-2, q_idx).index_select(-1, k_idx)
        values.append(selected.sum(dim=-1).mean())
    if not values:
        return float("nan")
    return float(torch.stack(values).mean().detach().float().cpu())


def _grid_roi_indices(
    height: int,
    width: int,
    roi: tuple[float, float, float, float],
) -> list[int]:
    if height <= 0 or width <= 0:
        return []
    x0, y0, x1, y1 = roi
    selected: list[int] = []
    for row in range(height):
        cell_y0, cell_y1 = row / height, (row + 1) / height
        for col in range(width):
            cell_x0, cell_x1 = col / width, (col + 1) / width
            if cell_x1 > x0 and cell_x0 < x1 and cell_y1 > y0 and cell_y0 < y1:
                selected.append(row * width + col)
    return selected


def _token_masks(
    backbone: torch.nn.Module,
    call_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    input_ids = call_kwargs.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise RuntimeError("MolmoAct2 diagnostics require batched input_ids")

    image_patch_id = getattr(getattr(backbone, "config", None), "image_patch_id", None)
    if image_patch_id is None:
        raise RuntimeError("MolmoAct2 config exposes no image_patch_id")
    visual_mask = input_ids == int(image_patch_id)
    valid_mask = call_kwargs.get("attention_mask")
    if not torch.is_tensor(valid_mask) or valid_mask.shape != input_ids.shape:
        valid_mask = input_ids != -1
    else:
        valid_mask = valid_mask.to(dtype=torch.bool)
    language_mask = valid_mask & ~visual_mask
    roi_mask = torch.zeros_like(visual_mask)

    explicit = os.environ.get("MOLMO_TEXT_TOKEN_INDICES", "").strip()
    mapping_method = "none"
    if explicit:
        indices = [int(item.strip()) for item in explicit.split(",") if item.strip()]
        for idx in indices:
            if idx < 0 or idx >= input_ids.shape[1]:
                raise ValueError(f"MOLMO_TEXT_TOKEN_INDICES contains out-of-range index {idx}")
            roi_mask[:, idx] = visual_mask[:, idx]
        mapping_method = "explicit-sequence-indices"
    else:
        roi = _TEXT_ROI_NORM or _parse_roi(os.environ.get("MOLMO_TEXT_ROI"))
        grids = call_kwargs.get("image_grids")
        if roi is not None and input_ids.shape[0] == 1 and torch.is_tensor(grids) and grids.numel() >= 4:
            # MolmoAct2 represents each image with two ordered 2-D grids.  The
            # first camera is the first image in the prompt.  Selecting cells
            # that intersect the ROI gives a deterministic approximation. The
            # manifest labels it as approximate so it is not mistaken for an
            # exact inverse of the processor's crop transform.
            first_grid = [int(value) for value in grids[0].detach().cpu().flatten()[:4]]
            local_indices = _grid_roi_indices(first_grid[0], first_grid[1], roi)
            offset = first_grid[0] * first_grid[1]
            local_indices += [
                offset + idx for idx in _grid_roi_indices(first_grid[2], first_grid[3], roi)
            ]
            visual_positions = visual_mask[0].nonzero(as_tuple=False).flatten()
            for local_idx in local_indices:
                if local_idx < visual_positions.numel():
                    roi_mask[0, visual_positions[local_idx]] = True
            mapping_method = "first-image-two-grid-intersection-approx"
        elif roi is not None:
            # Better to record a conservative visual-region fallback than to
            # silently claim pixel-level localization for unsupported batches.
            roi_mask = visual_mask.clone()
            mapping_method = "all-visual-fallback"

    metadata = {
        "mapping_method": mapping_method,
        "roi_normalized": list(_TEXT_ROI_NORM) if _TEXT_ROI_NORM is not None else None,
        "roi_source": _TEXT_ROI_SOURCE,
        "roi_token_count": [int(row.sum().item()) for row in roi_mask],
        "visual_token_count": [int(row.sum().item()) for row in visual_mask],
        "sequence_length": int(input_ids.shape[1]),
    }
    return roi_mask, visual_mask, language_mask, metadata


class _Recorder:
    def __init__(self, policy: torch.nn.Module, config: DiagnosticConfig):
        self.policy = policy
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone = policy._backbone()
        self.action_expert = policy._action_expert()
        self.call_index = 0
        self.captured_count = 0
        self.phase = "idle"
        self.roi_mask: torch.Tensor | None = None
        self.visual_mask: torch.Tensor | None = None
        self.language_mask: torch.Tensor | None = None
        self.roi_metadata: dict[str, Any] = {}
        self.vlm_hidden: dict[int, dict[str, np.ndarray]] = {}
        self.vlm_attention: dict[int, dict[str, float]] = {}
        self.action_hidden_samples: dict[int, list[np.ndarray]] = {}
        self.action_attention_samples: dict[int, list[dict[str, float]]] = {}
        self._handles: list[Any] = []
        self._install_hooks()

    def _install_hooks(self) -> None:
        transformer = getattr(self.backbone, "transformer", None)
        blocks = getattr(transformer, "blocks", None)
        if blocks is None:
            raise RuntimeError("Unsupported MolmoAct2: backbone.transformer.blocks is missing")
        for layer_idx, block in enumerate(blocks):
            self._handles.append(block.register_forward_hook(self._vlm_hidden_hook(layer_idx)))
            self._handles.append(block.self_attn.register_forward_hook(self._vlm_attention_hook(layer_idx)))

        action_blocks = getattr(self.action_expert, "blocks", None)
        if action_blocks is None:
            raise RuntimeError("Unsupported MolmoAct2: action_expert.blocks is missing")
        for layer_idx, block in enumerate(action_blocks):
            self._handles.append(block.register_forward_hook(self._action_hidden_hook(layer_idx)))
            self._handles.append(
                block.cross_attn.register_forward_pre_hook(
                    self._action_attention_hook(layer_idx), with_kwargs=True
                )
            )

    def should_capture(self) -> bool:
        selected = self.call_index % self.config.capture_every == 0
        below_limit = self.config.max_chunks < 0 or self.captured_count < self.config.max_chunks
        return selected and below_limit

    def begin(self, call_kwargs: dict[str, Any]) -> int:
        chunk_index = self.call_index
        self.call_index += 1
        self.captured_count += 1
        self.roi_mask, self.visual_mask, self.language_mask, self.roi_metadata = _token_masks(
            self.backbone, call_kwargs
        )
        self.vlm_hidden = {}
        self.vlm_attention = {}
        self.action_hidden_samples = {}
        self.action_attention_samples = {}
        return chunk_index

    def _vlm_hidden_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if self.phase != "vlm" or self.roi_mask is None:
                return
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                return
            self.vlm_hidden[layer_idx] = {
                "roi": _tensor_to_numpy(_mean_selected(hidden, self.roi_mask)),
                "visual": _tensor_to_numpy(_mean_selected(hidden, self.visual_mask)),
                "language": _tensor_to_numpy(_mean_selected(hidden, self.language_mask)),
            }

        return hook

    def _vlm_attention_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if self.phase != "vlm" or self.roi_mask is None:
                return
            attention = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
            if not torch.is_tensor(attention):
                return
            self.vlm_attention[layer_idx] = {
                "language_to_roi": _attention_mass(attention, self.language_mask, self.roi_mask),
                "visual_to_roi": _attention_mass(attention, self.visual_mask, self.roi_mask),
                "roi_to_language": _attention_mass(attention, self.roi_mask, self.language_mask),
                "roi_to_visual": _attention_mass(attention, self.roi_mask, self.visual_mask),
            }

        return hook

    def _action_hidden_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if self.phase != "action" or not torch.is_tensor(output):
                return
            sample = output.detach().float().mean(dim=(0, 1)).cpu().numpy()
            self.action_hidden_samples.setdefault(layer_idx, []).append(sample)

        return hook

    def _action_attention_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            if self.phase != "action" or self.roi_mask is None:
                return
            x = args[0] if args else kwargs.get("x")
            kv_k = kwargs.get("kv_k")
            attn_mask = kwargs.get("attn_mask")
            if not torch.is_tensor(x) or not torch.is_tensor(kv_k):
                return
            with torch.no_grad():
                batch, target_len, _ = x.shape
                q = module.q_proj(x).view(batch, target_len, module.num_heads, module.head_dim)
                k = module._as_heads(kv_k)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                if module.q_norm is not None:
                    q = module.q_norm(q)
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(
                    module.head_dim
                )
                if torch.is_tensor(attn_mask):
                    scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype)
                attention = torch.softmax(scores, dim=-1)
                sample = {
                    "action_to_roi": _attention_mass(
                        attention,
                        torch.ones(
                            (batch, target_len), device=attention.device, dtype=torch.bool
                        ),
                        self.roi_mask,
                    ),
                    "action_to_visual": _attention_mass(
                        attention,
                        torch.ones(
                            (batch, target_len), device=attention.device, dtype=torch.bool
                        ),
                        self.visual_mask,
                    ),
                }
                self.action_attention_samples.setdefault(layer_idx, []).append(sample)

        return hook

    def _save_reference_kv(
        self,
        chunk_index: int,
        kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        if not self.config.save_roi_kv or self.roi_mask is None:
            return
        if self.roi_mask.shape[0] != 1:
            raise RuntimeError("ROI K/V references currently require eval.batch_size=1")
        indices = self.roi_mask[0].nonzero(as_tuple=False).flatten()
        payload = {
            "format_version": 1,
            "variant": self.config.variant,
            "chunk_index": chunk_index,
            "roi_indices": indices.detach().cpu(),
            "layers": [
                {
                    "key": key.index_select(1, indices).detach().cpu(),
                    "value": value.index_select(1, indices).detach().cpu(),
                }
                for key, value in kv_states
            ],
        }
        torch.save(payload, self.config.output_dir / f"chunk_{chunk_index:05d}_roi_kv.pt")

    def intervene(
        self,
        chunk_index: int,
        kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self._save_reference_kv(chunk_index, kv_states)
        if self.config.intervention == "none":
            return list(kv_states)
        if self.roi_mask is None:
            raise RuntimeError("Cannot intervene before ROI token mapping")
        if self.roi_mask.shape[0] != 1:
            raise RuntimeError("Causal K/V interventions currently require eval.batch_size=1")
        target_indices = self.roi_mask[0].nonzero(as_tuple=False).flatten()
        if target_indices.numel() == 0:
            raise RuntimeError(
                "The sign ROI mapped to zero tokens; set MOLMO_TEXT_TOKEN_INDICES explicitly"
            )

        patched = [(key.clone(), value.clone()) for key, value in kv_states]
        if self.config.intervention == "zero_roi_kv":
            for key, value in patched:
                key[:, target_indices] = 0
                value[:, target_indices] = 0
            return patched

        reference_path = self.config.reference_dir / f"chunk_{chunk_index:05d}_roi_kv.pt"
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing paired K/V reference: {reference_path}")
        try:
            reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        except TypeError:
            reference = torch.load(reference_path, map_location="cpu")
        source_indices = reference["roi_indices"]
        if int(source_indices.numel()) != int(target_indices.numel()):
            raise RuntimeError(
                "Reference/target ROI token counts differ: "
                f"{int(source_indices.numel())} != {int(target_indices.numel())}. "
                "Use paired runs with identical prompts and ROI mapping."
            )
        if len(reference["layers"]) != len(patched):
            raise RuntimeError("Reference/target VLM layer counts differ")
        for (key, value), source in zip(patched, reference["layers"], strict=True):
            key[:, target_indices] = source["key"].to(device=key.device, dtype=key.dtype)
            value[:, target_indices] = source["value"].to(device=value.device, dtype=value.dtype)
        return patched

    def finish(self, chunk_index: int, actions: torch.Tensor) -> None:
        layer_ids = sorted(self.vlm_hidden)
        arrays: dict[str, np.ndarray] = {"actions": _tensor_to_numpy(actions)}
        for region in ("roi", "visual", "language"):
            if layer_ids:
                arrays[f"vlm_hidden_{region}"] = np.stack(
                    [self.vlm_hidden[idx][region] for idx in layer_ids]
                )
        action_layer_ids = sorted(self.action_hidden_samples)
        if action_layer_ids:
            arrays["action_hidden"] = np.stack(
                [np.mean(self.action_hidden_samples[idx], axis=0) for idx in action_layer_ids]
            )
        np.savez_compressed(self.config.output_dir / f"chunk_{chunk_index:05d}.npz", **arrays)

        def averaged(samples: dict[int, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
            result: dict[str, dict[str, float]] = {}
            for layer_idx, layer_samples in sorted(samples.items()):
                keys = sorted(layer_samples[0]) if layer_samples else []
                result[str(layer_idx)] = {
                    key: float(np.nanmean([sample[key] for sample in layer_samples])) for key in keys
                }
            return result

        record = {
            "format_version": 1,
            "variant": self.config.variant,
            "chunk_index": chunk_index,
            "intervention": self.config.intervention,
            "roi": self.roi_metadata,
            "vlm_attention": {str(key): value for key, value in sorted(self.vlm_attention.items())},
            "action_cross_attention": averaged(self.action_attention_samples),
            "npz": f"chunk_{chunk_index:05d}.npz",
            "roi_kv": f"chunk_{chunk_index:05d}_roi_kv.pt" if self.config.save_roi_kv else None,
        }
        with (self.config.output_dir / "manifest.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=True) + "\n")


def _prefill_kwargs(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "input_ids",
        "pixel_values",
        "image_token_pooling",
        "image_grids",
        "image_num_crops",
        "pixel_values_videos",
        "video_token_pooling",
        "video_grids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
    }
    return {key: value for key, value in call_kwargs.items() if key in allowed and value is not None}


def _set_eager_attention(backbone: torch.nn.Module) -> list[tuple[Any, str, Any]]:
    changed: list[tuple[Any, str, Any]] = []
    candidates = [
        getattr(backbone, "config", None),
        getattr(getattr(backbone, "config", None), "text_config", None),
        getattr(getattr(backbone, "transformer", None), "config", None),
    ]
    for candidate in candidates:
        if candidate is None or not hasattr(candidate, "_attn_implementation"):
            continue
        old = candidate._attn_implementation
        changed.append((candidate, "_attn_implementation", old))
        candidate._attn_implementation = "eager"
    return changed


def _restore_attributes(changed: list[tuple[Any, str, Any]]) -> None:
    for owner, name, value in reversed(changed):
        setattr(owner, name, value)


def _attach_recorder(policy: torch.nn.Module, config: DiagnosticConfig) -> None:
    recorder = _Recorder(policy, config)
    policy.config.enable_inference_cuda_graph = False
    set_cuda_graph = getattr(policy, "_set_inference_cuda_graph_enabled", None)
    if callable(set_cuda_graph):
        set_cuda_graph(False)

    backbone = policy._backbone()
    original_generate = backbone.generate_actions_from_inputs

    def wrapped_generate(_self, *args, **kwargs):
        if args:
            raise RuntimeError(
                "MolmoAct2 diagnostics expect keyword-only generate_actions_from_inputs arguments"
            )
        if not recorder.should_capture():
            recorder.call_index += 1
            return original_generate(**kwargs)

        chunk_index = recorder.begin(kwargs)
        supplied_kv = kwargs.get("encoder_kv_states")
        if supplied_kv is not None:
            kv_states = supplied_kv
        else:
            changed = _set_eager_attention(backbone)
            recorder.phase = "vlm"
            try:
                outputs = backbone(
                    **_prefill_kwargs(kwargs),
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                kv_states = backbone._extract_kv_states(outputs.past_key_values)
            finally:
                recorder.phase = "idle"
                _restore_attributes(changed)

        patched_kv = recorder.intervene(chunk_index, kv_states)
        generate_kwargs = dict(kwargs)
        generate_kwargs["encoder_kv_states"] = patched_kv
        recorder.phase = "action"
        try:
            actions = original_generate(**generate_kwargs)
        finally:
            recorder.phase = "idle"
        recorder.finish(chunk_index, actions)
        return actions

    backbone.generate_actions_from_inputs = types.MethodType(wrapped_generate, backbone)
    policy._molmo_diagnostic_recorder = recorder


def install_diagnostics() -> None:
    """Patch policy construction so every MolmoAct2 policy gets a recorder."""

    config = DiagnosticConfig.from_env()
    explicit_roi = _parse_roi(os.environ.get("MOLMO_TEXT_ROI"))
    if explicit_roi is not None:
        set_text_roi(explicit_roi, source="MOLMO_TEXT_ROI")

    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    if getattr(MolmoAct2Policy, "_molmo_diagnostics_installed", False):
        return
    original_init = MolmoAct2Policy.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _attach_recorder(self, config)

    MolmoAct2Policy.__init__ = patched_init
    MolmoAct2Policy._molmo_diagnostics_installed = True
    print(
        "[molmoact2_diagnostics] ACTIVE "
        f"(dir={config.output_dir}, variant={config.variant}, "
        f"intervention={config.intervention})"
    )
