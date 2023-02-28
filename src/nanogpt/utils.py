from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from nanogpt.config import GPTConfig, RunConfig


@torch.no_grad()
def estimate_loss(
    model: nn.Module, data, config: GPTConfig, device: str = "cpu"
) -> torch.Tensor:
    model.eval()

    ctx = get_context(config.run, device=device)

    losses = torch.zeros(config.training.eval_iters)

    for k in range(config.training.eval_iters):
        X, Y = get_batch(data, config, device)
        with ctx:
            logits, loss = model(X, Y)
        losses[k] = loss.item()

    model.train()

    return losses.mean()


def get_context(config: RunConfig, device: str = "cpu") -> Any:
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.d_type]
    return (
        nullcontext()
        if device == "cpu" or device == "mps"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )


def get_batch(
    data, config: GPTConfig, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(
        len(data) - config.model.block_size, (config.training.batch_size,)
    )

    x = torch.stack([data[i : i + config.model.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + config.model.block_size] for i in ix])

    if device == "cuda":
        # Pin arrays x, y, which allows us to move them to
        # GPU asynchronously (non_blocking = True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"
