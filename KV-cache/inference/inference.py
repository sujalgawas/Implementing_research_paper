import torch

from cache.kv_cache import KVCache


@torch.no_grad()
def generate(
    prompt: str,
    model: torch.nn.Module,
    tokenizer,
    kv_cache: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = 50,
) -> str:

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
        next(model.parameters()).device
    )
    max_len = getattr(model, "max_len", 1024)

    if kv_cache:
        context = input_ids[:, -max_len:]
        cache = KVCache(
            num_layers=len(model.transformer.h),
            batch_size=context.size(0),
            num_heads=model.transformer.h[0].attn.n_head,
            head_dim=model.transformer.h[0].attn.head_dim,
            max_length=max_len,
            device=context.device,
            dtype=next(model.parameters()).dtype,
        )
        logits = model(context, kv_cache=cache)[:, -1, :] / temperature

    for step in range(max_new_tokens):
        if not kv_cache:
            context = input_ids[:, -max_len:]
            logits = model(context)[:, -1, :] / temperature
            
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if kv_cache and step + 1 < max_new_tokens:
            logits = model(next_token, kv_cache=cache)[:, -1, :] / temperature

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
