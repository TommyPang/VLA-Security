#!/usr/bin/env python3
"""Measure one real MolmoAct patch-gradient step with synthetic, shape-matched inputs."""
import argparse
import json
import os
import time

import torch
from bitsandbytes.nn import Linear4bit, Linear8bitLt
from trap.config import load_config
from trap.data.molmoact import MolmoTextSegments, molmo_prompt
from trap.models.molmoact import MolmoActAdapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=f'{PROJECT_ROOT}/configs/molmoact_coke_to_orange_gpu1_8bit.yaml')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accum-steps', type=int, default=1)
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg.attack['batch_size'] = args.batch_size
    cfg.attack['grad_accum_steps'] = args.accum_steps
    device = torch.device(cfg.device)
    start = time.time()
    adapter = MolmoActAdapter(cfg, device)
    adapter.load()
    tok = adapter.processor.tokenizer
    message = [{'role': 'user', 'content': [dict(type='text', text=molmo_prompt('pick coke can'))]}]
    adapter.prompt_text = adapter.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    seed_ids = tok(' The orange and the can are on the table. Move the end effector toward the target. ', add_special_tokens=False).input_ids
    def tokens(n):
        return torch.tensor((seed_ids * (n // len(seed_ids) + 1))[:n], dtype=torch.long)
    segments = MolmoTextSegments(tokens(60), tokens(48), tokens(56), tokens(44))
    patch = torch.rand(tuple(cfg.attack['patch']['size']), device=device, requires_grad=True)
    int8_layers = sum(isinstance(m, Linear8bitLt) for m in adapter.model.modules())
    int4_layers = sum(isinstance(m, Linear4bit) for m in adapter.model.modules())
    loaded_gib = torch.cuda.memory_allocated(device) / 2**30
    torch.cuda.reset_peak_memory_stats(device)
    task_patch_grad = torch.zeros_like(patch)
    losses = []
    for _ in range(args.accum_steps):
        batch = {'images': [torch.rand(3, 512, 640) for _ in range(args.batch_size)],
                 'segments': [segments] * args.batch_size}
        metrics = adapter.training_step(batch, patch, include_regularizers=False)
        micro_grad = torch.autograd.grad(metrics['loss'] / args.accum_steps, patch)[0]
        task_patch_grad.add_(micro_grad.detach())
        losses.append(float(metrics['loss'].detach().cpu()))
        del batch, metrics, micro_grad
    reg_loss, _ = adapter.regularization_loss(patch)
    torch.autograd.backward((patch, reg_loss), (task_patch_grad, torch.ones_like(reg_loss)))
    torch.cuda.synchronize(device)
    print(json.dumps({'load_mode': os.environ.get('TRAP_MOLMOACT_LOAD_MODE'),
                      'batch_size': args.batch_size, 'accum_steps': args.accum_steps,
                      'effective_batch': args.batch_size * args.accum_steps,
                      'int8_layers': int8_layers, 'int4_layers': int4_layers,
                      'loaded_gib': round(loaded_gib, 2),
                      'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                      'peak_reserved_gib': round(torch.cuda.max_memory_reserved(device) / 2**30, 2),
                      'loss': round(sum(losses) / len(losses), 5),
                      'patch_grad_norm': round(float(patch.grad.norm().cpu()), 5),
                      'elapsed_sec': round(time.time() - start, 1)}, indent=2))


if __name__ == '__main__':
    main()
