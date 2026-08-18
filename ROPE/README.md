# PyTorch Implementation of GPT-2 with RoPE (Rotary Position Embedding) and Pre-LN

This repository contains a PyTorch implementation of a **GPT-2** style autoregressive language model upgraded with **Rotary Position Embeddings (RoPE)** and **Pre-Layer Normalization (Pre-LN)**. It combines the decoder-only architecture from OpenAI's paper [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019) with the rotary position encoding introduced in [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) by Su et al. (2021) — a key architectural foundation powering modern LLMs like LLaMA, Mistral, and Gemma.

The goal of this project was to implement 2D rotary position embeddings, explicit per-head Query/Key rotation, Pre-LN residual connections, byte-level BPE tokenization, and temperature-controlled autoregressive generation from scratch using the OpenWebText dataset.

---

## Project Philosophy: Why This Implementation?

Traditional GPT models (such as GPT-1 and GPT-2) rely on learned absolute positional embeddings added directly to token embeddings at the input layer. While simple, absolute positional embeddings do not explicitly capture relative token distances and struggle to generalize to sequence lengths beyond what was seen during training.

The primary goal of this project was gaining a **deep conceptual understanding** of modern transformer design patterns:

- **Rotary Position Embedding (RoPE)**: Incorporating relative positional information by rotating Query and Key vectors in the complex plane based on their token positions, rather than adding static positional vectors to input embeddings.
- **No Input Position Embeddings**: Simplifying token embeddings (`embeddingGPT`) to purely map token IDs to vectors, offloading all spatial and positional awareness to the attention mechanism.
- **Custom `RoPEMultiHeadAttention` with `F.scaled_dot_product_attention`**: Building an explicit multi-head projection module because RoPE must rotate $Q$ and $K$ **per-head** (`head_dim`) *after* linear projection, which standard `nn.MultiheadAttention` hides internally.
- **Pre-Layer Normalization (Pre-LN)**: Placing layer normalization before the self-attention and feed-forward sub-blocks, stabilizing gradient flow across deep transformer layers.
- **ByteLevel BPE Tokenization**: Using HuggingFace's ByteLevel BPE with a single special token (`<|endoftext|>`) to process raw UTF-8 text seamlessly.
- **Autoregressive Generation**: Temperature-controlled decoding with top-k sampling and early stopping on sequence boundary tokens.

To achieve this cleanly, I:
- Built **`RoPE`**, **`embeddingGPT`**, **`RoPEMultiHeadAttention`**, **`feedforward`**, **`Transformer` (Pre-LN Block)**, and **`GPT2`** top-level model wrapper.
- Implemented the 2D complex-plane rotation math using vector slicing (`even`/`odd` dimensions) and precomputed trigonometric caches (`cos` / `sin` buffers).
- Handled technical pipeline details: causal attention masking (`torch.triu`), target-shifted dataset preparation (`ids[:-1]` vs `ids[1:]`), padding loss masking (`-100`), and sampling generation loops.

---

## Rotary Position Embedding (RoPE): Core Mechanics

### Why RoPE?

RoPE encodes positional information by multiplying Query ($q$) and Key ($k$) representations by an orthogonal rotation matrix $R_{\Theta, m}$. Given token positions $m$ and $n$, the inner product between rotated queries and keys satisfies:

$$\langle R_{\Theta, m} q_m, R_{\Theta, n} k_n \rangle = g(q_m, k_n, m - n)$$

This property means the attention score between two tokens depends directly on their **relative distance** $(m - n)$ rather than their absolute coordinates, providing better length extrapolation and relative position awareness.

### 2D Rotation Implementation

Rather than multiplying by a large sparse matrix, RoPE pairs adjacent even and odd vector dimensions and rotates each 2D sub-vector:

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

where frequencies are calculated as:

$$\theta_i = 10000^{-2i / d}$$

```python
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, d_model, max_len=512, base=10000):
        super().__init__()

        # Frequencies for each pair of dimensions
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2).float() / d_model)
        )

        # Positions: [0, 1, 2, ..., max_len-1]
        positions = torch.arange(max_len).float()

        # [max_len, d_model / 2]
        freqs = torch.outer(positions, inv_freq)

        # Precompute cos and sin buffers
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x):
        """
        x shape: [..., seq_len, d_model]
        Works for [batch, seq_len, d_model] or [batch, num_heads, seq_len, head_dim]
        """
        seq_len = x.size(-2)

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        # Split even and odd dimensions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply rotation
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Put dimensions back together
        x_rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)

        return x_rotated
```

### Why a Custom `RoPEMultiHeadAttention` Module?

In PyTorch, standard `nn.MultiheadAttention` combines the linear projections ($W_q, W_k, W_v$) internally, preventing us from accessing $Q$ and $K$ before attention scoring.

Because RoPE must rotate $Q$ and $K$ **after** they have been projected and split into per-head vectors (`head_dim = d_model // num_heads`), I built `RoPEMultiHeadAttention` to explicitly project $Q, K, V$, apply `RoPE` to $Q$ and $K$, and compute attention using PyTorch's optimized `F.scaled_dot_product_attention`:

```python
import torch.nn as nn
import torch.nn.functional as F

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=512, base=10000, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout_p = dropout
        self.rope = RoPE(self.head_dim, max_len=max_len, base=base)

    def forward(self, x, attn_mask=None):
        batch, seq_len, d_model = x.shape

        # Linear projections & split into attention heads
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE on per-head vectors
        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attn_out)
```

---

## Tokenization & Dataset Pipeline

### Direct ByteLevel BPE Tokenizer

Following GPT-2, tokenization uses ByteLevel Byte-Pair Encoding from HuggingFace `tokenizers`:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=50257,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=ByteLevel.alphabet()
)

tokenizer.decoder = ByteLevelDecoder()
tokenizer.train(files=["data.txt"], trainer=trainer)
tokenizer.save("tokenizer.json")
```

### Dataset Preparation & Padding Masking

The `GPT2Dataset` formats input and target pairs for autoregressive next-token prediction, setting padding token positions to `-100` so `CrossEntropyLoss` ignores them during optimization:

```python
import torch
from torch.utils.data import Dataset

class GPT2Dataset(Dataset):
    def __init__(self, df, tokenizer, block_size=512):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.df = df
        self.pad_id = tokenizer.token_to_id('<|endoftext|>')

    def __pad(self, ids):
        ids = ids[:self.block_size]
        ids = ids + [self.pad_id] * (self.block_size - len(ids))
        return ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        encoding = self.tokenizer.encode(text)
        ids = self.__pad(encoding.ids)

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        
        # Ignore loss on padding tokens
        target_ids[target_ids == self.pad_id] = -100

        return input_ids, target_ids
```

---

## Architecture & Model Backbone

The model implements a 12-layer decoder with Pre-Layer Normalization:

- **`embeddingGPT`**: Token embedding lookup only (`nn.Embedding(vocab_size, d_model)`). Positional information is omitted here since it is injected into $Q$ and $K$ via RoPE.
- **`feedforward`**: Two-layer position-wise MLP expanding $d_{\text{model}} \to 4 \times d_{\text{model}} \to d_{\text{model}}$ with GELU activation and dropout:
  $$\text{FFN}(x) = \text{Dropout}(\text{Linear}_2(\text{GELU}(\text{Linear}_1(x))))$$
- **`Transformer` Block (Pre-LN)**: Pre-Layer Normalization wraps around self-attention and feed-forward networks:
  ```python
  class Transformer(nn.Module):
      def __init__(self, d_model, num_heads, max_len=512, dropout=0.1):
          super().__init__()
          self.layer_norm1 = nn.LayerNorm(d_model)
          self.masked_attn = RoPEMultiHeadAttention(
              d_model=d_model, num_heads=num_heads, max_len=max_len, dropout=dropout
          )
          self.layer_norm2 = nn.LayerNorm(d_model)
          self.feedforward = feedforward(d_model, dropout=dropout)
          self.dropout = nn.Dropout(dropout)

      def forward(self, x, attn_mask):
          # Pre-LN Self Attention
          normed = self.layer_norm1(x)
          attn_out = self.masked_attn(normed, attn_mask=attn_mask)
          x = x + self.dropout(attn_out)

          # Pre-LN Feedforward
          normed2 = self.layer_norm2(x)
          ff_out = self.feedforward(normed2)
          x = x + ff_out

          return x
  ```
- **`GPT2` Backbone**: Stacks 12 Transformer blocks (`d_model = 1024`, `num_heads = 16`), followed by a final Layer Normalization layer (`ln_f`) and linear output projection (`lm_head`, `bias=False`).

### Model Specifications

- **Model Dimension (`d_model`)**: 1024
- **Attention Heads (`num_heads`)**: 16
- **Head Dimension (`head_dim`)**: 64
- **Transformer Layers (`num_layers`)**: 12
- **Context Length (`max_len`)**: 512 tokens
- **Vocabulary Size**: 50,257
- **Base Frequency (`base`)**: 10,000

---

## Training and Temperature-Controlled Generation

### Training Setup

- **Dataset**: OpenWebText ([`Skylion007/openwebtext`](https://huggingface.co/datasets/Skylion007/openwebtext))
- **Optimizer**: `AdamW` ($\text{lr} = 3 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\text{eps} = 10^{-8}$, $\text{weight\_decay} = 0.1$)
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **Loss Function**: `nn.CrossEntropyLoss(ignore_index=-100)`
- **Batch Size**: 16
- **Epochs**: 3

```python
loss = loss_fn(
    logits.view(-1, logits.size(-1)),
    target_ids.view(-1)
)
```

### Autoregressive Text Generation with Temperature Scaling

During inference, token probabilities are sampled autoregressively using temperature scaling and top-k filtering:

$$\text{logits}_{\text{scaled}} = \frac{\text{logits}}{\text{temperature}}$$

- **Lower Temperature ($T < 0.7$)**: Makes distribution sharper, favoring confident and greedy token choices matching trained data distribution.
- **Higher Temperature ($T \ge 0.8$)**: Flattens distribution, introducing creativity and variety in text outputs.

```python
@torch.no_grad()
def generate(prompt, max_new_tokens=100, temperature=1.0, top_k=50, eos_token="<|endoftext|>"):
    model.eval()

    eos_id = tokenizer.token_to_id(eos_token)
    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.max_len:]

        logits = model(context)
        next_token_logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, _ = torch.topk(next_token_logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            next_token_logits = torch.where(
                next_token_logits < min_value,
                torch.full_like(next_token_logits, float('-inf')),
                next_token_logits
            )

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop when end-of-text special token is generated
        if next_token.item() == eos_id:
            break

    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)

# Generation example
output = generate("Once upon a time", max_new_tokens=100, temperature=0.8, top_k=50)
print(output)
```
