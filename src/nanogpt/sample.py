import logging
from pathlib import Path

import hydra
import torch
import torch.backends.cuda
import torch.backends.cudnn
from hydra.core.config_store import ConfigStore

from nanogpt.config import GPTConfig
from nanogpt.model import get_model
from nanogpt.utils import get_context, get_device, load_data

cs = ConfigStore.instance()
cs.store(name="gpt_config", node=GPTConfig)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(config: GPTConfig) -> None:
    seed = 1337
    torch.manual_seed(config.run.seed)
    torch.cuda.manual_seed(config.run.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    device = get_device()
    ctx = get_context(config=config.run, device=device)
    # Always load trained model
    if config.run.init_from == "scratch":
        config.run.init_from = "resume"
    model, _, _ = get_model(config=config, device=device)
    model.eval()
    model.to(device)

    # Load data
    text = load_data(Path.cwd() / "data" / "tinyshakespeare.txt")
    chars = sorted(set(text))
    # config.model.vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [
        stoi[c] for c in s
    ]  # encoder: take a string, output a list of integers
    decode = lambda l: "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of int

    temperature = 0.8

    with torch.no_grad():
        with ctx:
            start_ids = encode("\n")
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            # print(x.size())
            y = model.generate(x, max_new_tokens=500, temperature=temperature, top_k=15)
            print(decode(y[0].tolist()))


if __name__ == "__main__":
    main()
