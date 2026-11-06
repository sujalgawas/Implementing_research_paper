import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cache.kv_cache import KVCache


class GPT2Conv1D(nn.Module):
    """GPT-2's Conv1D projection, whose weight is stored as [in, out]."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x.reshape(-1, x.size(-1)), self.weight).view(
            *x.shape[:-1], self.weight.size(1)
        )


class GPT2Attention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.c_attn = GPT2Conv1D(n_embd, 3 * n_embd)
        self.c_proj = GPT2Conv1D(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape
        query, key, value = self.c_attn(x).split(n_embd, dim=2)

        query = query.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        past_length = 0 if kv_cache is None else kv_cache.position
        if kv_cache is not None:
            key, value = kv_cache.update(key, value, layer_idx)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        key_length = key.size(2)
        query_positions = torch.arange(
            past_length, past_length + seq_len, device=x.device
        )
        key_positions = torch.arange(key_length, device=x.device)
        causal_mask = (
            key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        ).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        return self.resid_dropout(self.c_proj(context))


class GPT2MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.c_fc = GPT2Conv1D(n_embd, 4 * n_embd)
        self.c_proj = GPT2Conv1D(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.c_fc(x), approximate="tanh")
        return self.dropout(self.c_proj(x))


class GPT2Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.attn = GPT2Attention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.mlp = GPT2MLP(n_embd, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_idx: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), layer_idx, kv_cache, attention_mask)
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        num_head: int = 12,
        num_layers: int = 12,
        max_len: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab_size, d_model)
        self.transformer.wpe = nn.Embedding(max_len, d_model)
        self.transformer.h = nn.ModuleList(
            [GPT2Block(d_model, num_head, dropout) for _ in range(num_layers)]
        )
        self.transformer.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, seq_len = input_ids.shape
        past_length = 0 if kv_cache is None else kv_cache.position
        if past_length + seq_len > self.max_len:
            raise ValueError(
                f"Input sequence exceeds context length {self.max_len}"
            )

        position_ids = torch.arange(
            past_length, past_length + seq_len, device=input_ids.device
        ).unsqueeze(0)
        x = self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)

        for layer_idx, block in enumerate(self.transformer.h):
            x = block(
                x,
                attention_mask=attention_mask,
                layer_idx=layer_idx,
                kv_cache=kv_cache,
            )

        if kv_cache is not None:
            kv_cache.advance(seq_len)

        return self.lm_head(self.transformer.ln_f(x))
