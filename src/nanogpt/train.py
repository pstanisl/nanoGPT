import logging
import time
from pathlib import Path

import hydra
import torch
import torch.backends.cuda
import torch.backends.cudnn
from hydra.core.config_store import ConfigStore

from nanogpt.config import GPTConfig
from nanogpt.model import GPT, get_checkpoint, get_model
from nanogpt.utils import estimate_loss, get_batch, get_context, get_device, load_data

cs = ConfigStore.instance()
cs.store(name="gpt_config", node=GPTConfig)

logger = logging.getLogger(__name__)


def get_lr(it: int) -> float:
    return 3e-4


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    iter_num: int,
    best_val_loss: float,
) -> None:
    checkpoint_path = Path(config.run.output_dir) / "ckpt.pt"

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": config.model,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
    }
    logger.info(f"saving checkpoint to {checkpoint_path.parent}")
    torch.save(checkpoint, checkpoint_path)


def train(
    model,
    data_train,
    data_val,
    config: GPTConfig,
    iter_num: int = 0,
    best_val_loss: float = 1e9,
    device: str = "cpu",
) -> None:
    gradient_accumulation_steps = 5 * 8
    # Variable `iter_num` can be `> 0` because of resume, this variable represents
    # local interation number.
    local_iter_num = 0
    log_interval = 1
    running_mfu = -1.0

    ctx = get_context(config=config.run, device=device)
    # Optimzer
    optimizer = model.configure_optimizers(config.training.learning_rate)
    if config.run.init_from == "resume":
        checkpoint = get_checkpoint(config=config, device=device)
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(config.run.d_type == "float16"))

    # fetch the very first batch
    X, Y = get_batch(data_train, config=config, device=device)
    t0 = time.time()

    while True:
        lr = (
            get_lr(iter_num)
            if config.training.decau_lr
            else config.training.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate the loss on train/val sets and write checkpoints
        # if iter_num % config.training.eval_interval == 0 and master_process:
        if iter_num % config.training.eval_interval == 0:
            losses_train = estimate_loss(model, data_train, config, device=device)
            losses_val = estimate_loss(model, data_val, config, device=device)
            logger.info(
                f"step {iter_num}: train loss {losses_train:.4f}, "
                f"val loss {losses_val:.4f}"
            )

            if losses_val < best_val_loss or config.run.always_save_checkpoint:
                best_val_loss = losses_val.item()
                if iter_num > 0:
                    save_checkpoint(model, optimizer, config, iter_num, best_val_loss)

        # TODO: evaluation only run

        # Forward/backward update, with optional gradient accumulaton to simulate larger
        # batch size and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
            # Immediately async prefetch next batch while model is doing the forward
            # pass on the GPU
            X, Y = get_batch(data_train, config, device=device)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # Clip the gradient
        if config.training.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip
            )
        # Step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # Flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # if iter_num % log_interval == 0 and master_process:
        if iter_num % log_interval == 0:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = model.estimate_mfu(
                    config.training.batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            logger.info(
                f"{iter_num}/{config.training.max_iters}: loss {lossf:.4f}, "
                f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )

        iter_num += 1
        local_iter_num += 1
        # termination conditions
        if iter_num > config.training.max_iters:
            break


@hydra.main(version_base=None, config_path="../../conf", config_name="gpt")
def main(config: GPTConfig) -> None:
    # master_process = True
    seed_offset = 0

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # Get device where to run, possible vaules "cpu", "cuda", "mps"
    device = get_device()
    # Load data
    text = load_data(Path.cwd() / "data" / "tinyshakespeare.txt")
    chars = sorted(set(text))
    config.model.vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [
        stoi[c] for c in s
    ]  # encoder: take a string, output a list of integers
    decode = lambda l: "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model, iter_num, best_val_loss = get_model(config, device=device)
    # Crop down the model block size if desired, using model surgery
    # if config.model.block_size < model.config.block_size:
    #     model.crop_block_size(config.model.block_size)
    #     model_args["block_size"] = config.model.block_size # so that the checkpoint will have the right value
    model.to(device)

    if config.run.compile:
        logger.info("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    train(model, train_data, val_data, config, iter_num, best_val_loss, device=device)


if __name__ == "__main__":
    main()
