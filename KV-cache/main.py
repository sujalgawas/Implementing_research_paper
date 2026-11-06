from pathlib import Path

import torch
from transformers import GPT2Tokenizer

from inference.inference import generate
from model.gpt import GPT2


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "openai-community/gpt2"
WEIGHTS_PATH = ROOT / "gpt2_weights.pth"
CONFIG_PATH = ROOT / "gpt2_config"


def load_weights() -> None:
    from transformers import GPT2LMHeadModel

    hf_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    torch.save(hf_model.state_dict(), WEIGHTS_PATH)
    hf_model.config.save_pretrained(CONFIG_PATH)


def main() -> None:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if not WEIGHTS_PATH.is_file():
        load_weights()

    model = GPT2()
    weights = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    
    import time

    start_key = time.perf_counter()

    output = generate("once upon a time", model, tokenizer)

    end_key = time.perf_counter()
    execution_time = end_key - start_key
    print(f"----- with KV_cache {execution_time:.6f} seconds to complete. ------")
    print(output)


    start_key = time.perf_counter()

    output = generate("once upon a time", model, tokenizer,kv_cache=False)

    end_key = time.perf_counter()
    execution_time = end_key - start_key
    print(f"---- without KV_cache {execution_time:.6f} seconds to complete. -----")
    print(output)


if __name__ == "__main__":
    main()
