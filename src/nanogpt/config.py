from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float

    vocab_size: int
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    @staticmethod
    def get_keys() -> dict:
        return ModelConfig.__dict__.get("__annotations__").keys()


@dataclass
class TrainingConfig:
    learning_rate: float

    batch_size: int = 8
    decau_lr: bool = False
    eval_interval: int = 2000
    eval_iters: int = 200
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0
    max_iters: int = 3000


@dataclass
class RunConfig:
    output_dir: Path

    always_save_checkpoint: bool = False
    compile: bool = False
    d_type: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    init_from: str = "scratch"

    seed: int = 1337


@dataclass
class GPTConfig:
    model: ModelConfig
    run: RunConfig
    training: TrainingConfig
