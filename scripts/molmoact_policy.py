"""MolmoAct closed-loop policy with selectable 4-bit or 8-bit loading.

Matches ``simpler_env.policies.molmoact.molmoact_model.MolmoActInference`` for
image resize, prompting, action post-processing, and RT-1 unnormalization.
The GPU1 workflow defaults to 8-bit language layers with BF16 vision/head.
"""
from __future__ import annotations

import os

import cv2 as cv
import numpy as np
from PIL import Image


def molmo_prompt(instruction: str) -> str:
    return (
        f"The task is {instruction}. "
        "What is the action that the robot should take. "
        f"To figure out the action that the robot should take to {instruction}, "
        "let's think through it step by step. "
        "First, what is the depth map for this image? "
        "Second, what is the trajectory of the end effector? "
        "Based on the depth map of the image and the trajectory of the end effector, "
        "what is the action that the robot should take?"
    )


class MolmoAct4BitPolicy:
    """MolmoAct with the official SimplerEnv Google Robot action post-process."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        max_new_tokens: int = 320,
        image_size: tuple[int, int] = (256, 256),
        action_scale: float = 1.0,
        load_mode: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.device = device
        # Depth CoT alone is ~100+ tokens; full depth+traj+7 action tokens needs ≥208.
        # Eval previously used 192 → truncated action → parse_action([]) → zero motion.
        self.max_new_tokens = max_new_tokens
        self.image_size = list(image_size)
        self.action_scale = action_scale
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )

        mode = load_mode or os.environ.get("TRAP_MOLMOACT_LOAD_MODE", "4bit_skip_vision")
        if mode in {"8bit", "8bit_skip_vision", "4bit_skip_vision"}:
            skip = ["vision_backbone", "model.vision_backbone", "lm_head"]
            if mode.startswith("4bit"):
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=skip,
                )
            else:
                quant = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=skip,
                )
            load_kwargs = dict(quantization_config=quant, device_map={"": device})
            if mode.startswith("8bit"):
                # Match the optimizer's BF16 vision/head modules.
                load_kwargs["torch_dtype"] = torch.bfloat16
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                trust_remote_code=True,
                **load_kwargs,
            )
        elif mode in {"bf16", "full", "official"}:
            # Official SimplerEnv path — typically OOMs on 16GB with vision+LLM.
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            raise ValueError(f"Unknown TRAP_MOLMOACT_LOAD_MODE={mode!r}")

        self.model.eval()
        self.task_description: str | None = None
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.sticky_gripper_num_repeat = 15  # google_robot
        self.last_generated_text = ""
        self.last_parse_ok = False
        self.last_raw_action: np.ndarray | None = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.last_generated_text = ""
        self.last_parse_ok = False
        self.last_raw_action = None

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    def step(self, image: np.ndarray, task_description: str | None = None):
        from transforms3d.euler import euler2axangle

        torch = self.torch
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        assert image.dtype == np.uint8
        # Official MolmoActInference always resizes to 256×256 before the processor.
        image = self._resize_image(image)
        img = Image.fromarray(image)
        prompt = molmo_prompt(self.task_description or "")
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [dict(type="text", text=prompt)]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(images=[img], text=text, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        generated_tokens = generated_ids[:, inputs["input_ids"].size(1) :]
        n_new = int(generated_tokens.shape[1])
        generated_text = self.processor.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        self.last_generated_text = generated_text

        action_lists = None
        try:
            action_lists = self.model.parse_action(generated_text, unnorm_key="fractal20220817_data")
        except Exception as exc:
            print(f"[warn] parse_action failed: {exc}")

        self.last_parse_ok = bool(action_lists)
        if not action_lists:
            # Truncated CoT (too-low max_new_tokens) or malformed action tokens.
            print(
                f"[warn] no 7-D action parsed (new_tokens={n_new}/{self.max_new_tokens}); "
                f"tail={generated_text[-80:]!r}"
            )
            unnormalized = np.zeros(7, dtype=np.float64)
        else:
            unnormalized = action_lists[0]
            unnormalized = np.asarray(unnormalized, dtype=np.float64).reshape(-1)[:7]
            if unnormalized.shape[0] < 7:
                unnormalized = np.pad(unnormalized, (0, 7 - unnormalized.shape[0]))

        self.last_raw_action = unnormalized.copy()
        raw_action = {
            "world_vector": unnormalized[:3],
            "rotation_delta": unnormalized[3:6],
            "open_gripper": unnormalized[6:7],
        }
        out = {
            "world_vector": raw_action["world_vector"] * self.action_scale,
            "rot_axangle": None,
            "gripper": None,
            "terminate_episode": np.array([0.0]),
        }
        roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        ax, ang = euler2axangle(roll, pitch, yaw)
        out["rot_axangle"] = ax * ang * self.action_scale

        # google_robot sticky relative gripper (SimplerEnv MolmoActInference)
        current_gripper = raw_action["open_gripper"]
        if self.previous_gripper_action is None:
            relative = np.array([0.0])
        else:
            relative = self.previous_gripper_action - current_gripper
        self.previous_gripper_action = current_gripper
        if np.abs(relative) > 0.5 and not self.sticky_action_is_on:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative = self.sticky_gripper_action
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0
        out["gripper"] = relative
        return raw_action, out
