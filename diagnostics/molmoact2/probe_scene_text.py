"""Probe whether MolmoAct2 can transcribe text in captured rollout frames."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model", default="allenai/MolmoAct2-LIBERO")
    parser.add_argument(
        "--prompt",
        default=(
            "Transcribe every visible word in this robot-camera image exactly. "
            "Return only the text you can read, preserving word order."
        ),
    )
    parser.add_argument("--expected-text")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("ocr_probe.jsonl"))
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        extra_special_tokens={},
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
    ).to(args.device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as stream:
        for image_path in args.images:
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {"type": "image", "image": image},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {
                key: value.to(args.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            context_length = int(inputs["input_ids"].shape[1])
            autocast = (
                torch.autocast(device_type=args.device.split(":", 1)[0], dtype=dtype)
                if args.dtype == "bfloat16"
                else torch.autocast(device_type=args.device.split(":", 1)[0], enabled=False)
            )
            with torch.inference_mode(), autocast:
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generated_tokens = generated[:, context_length:]
            post_process = getattr(processor, "post_process_image_text_to_text", None)
            if callable(post_process):
                transcription = post_process(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            else:
                transcription = processor.tokenizer.decode(
                    generated_tokens[0], skip_special_tokens=True
                )
            record = {
                "image": str(image_path),
                "prompt": args.prompt,
                "transcription": transcription,
            }
            if args.expected_text is not None:
                record["expected_text"] = args.expected_text
                record["normalized_similarity"] = SequenceMatcher(
                    None,
                    _normalise(args.expected_text),
                    _normalise(transcription),
                ).ratio()
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{image_path}: {transcription}")


if __name__ == "__main__":
    main()
