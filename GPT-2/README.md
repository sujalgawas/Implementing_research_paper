# PyTorch Implementation of GPT-2 for Autoregressive Language Modeling

This repository contains a PyTorch implementation of OpenAI's **GPT-2** model, built as a hands-on learning project to reconstruct the modified decoder-only Transformer architecture proposed in the seminal paper [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by Radford et al. (2019).

The goal of this project was to implement byte-level subword tokenization, Pre-Layer Normalization, residual connection scaling, BERT-inspired causal attention masking, and temperature-controlled generation from scratch using the OpenWebText dataset.

---

## Project Philosophy: Why This Implementation?

Building upon the foundation of GPT-1 introduced key architectural refinements and data representation shifts to enable zero-shot task transfer without requiring fine-tuning. The primary goal of this project was gaining a **deep conceptual understanding** of GPT-2's core innovations:

- **Pre-Layer Normalization (Pre-LN)**: Moving layer normalization to the input of each sub-block (attention and feed-forward), alongside an extra final layer norm (`ln_f`) after the final self-attention block.
- **Residual Connections from Research Paper**: Incorporating residual skip connections directly around the attention and feed-forward sub-blocks to stabilize gradient flow in deeper networks.
- **Direct ByteLevel BPE Tokenization**: Operating directly on raw UTF-8 byte streams to handle any text sequence without needing unknown (`<unk>`) out-of-vocabulary tokens.
- **Single Special Token Paradigm**: Streamlining sequence control down to a single special token (`<|endoftext|>`).
- **Controlled Decoding with Temperature**: Adding temperature scaling to logit sampling during autoregressive text generation to balance high accuracy matching original trained data distribution versus creative randomness.

To achieve this cleanly, I:
- Built **`embeddingGPT`**, **`feedforward`**, **`Transformer` (Decoder Block)**, and **`GPT2`** top-level model backbone.
- Leveraged PyTorch's native `nn.MultiheadAttention` and `nn.LayerNorm` components for numerical stability, clean code, and fast execution.
- Handled technical details manually: custom `ByteLevel` tokenizer pipeline, BERT-inspired `-inf` causal masking (`torch.triu`), target-shifted dataset preparation (`ids[:-1]` vs `ids[1:]`), and a temperature-scaled top-k text generation loop.

---

## Hardware Limitations and Scaling Challenges

Initially, the goal of this project was to construct a larger GPT-2 variant (such as GPT-2 Medium with 774M parameters or the full 1.5B parameter model).

However, during early training attempts with larger model dimensions and batch sizes, severe hardware constraints were encountered—resulting in constant system crashes due to **Memory Limit Errors / CUDA Out-of-Memory (OOM)** during forward and backward gradient passes.

### Hardware Trade-off & Final Model Specs:

To ensure training stability, eliminate memory limit crashes, and keep memory consumption within available GPU/TPU resources, the final architecture parameters were set to:

- **Model Dimension (`d_model`)**: 1024
- **Attention Heads (`num_heads`)**: 16
- **Transformer Layers (`num_layers`)**: 12
- **Parameter Count**: ~345 Million Parameters
- **Context Length (`max_len`)**: 512 tokens
- **Batch Size**: 16

---

## Tokenization: ByteLevel BPE & Single Special Token

### Direct ByteLevel Pre-Tokenizer Import

Taking inspiration directly from the GPT-2 research paper, the tokenizer pipeline bypasses traditional word or character splitters and directly imports `ByteLevel` from HuggingFace's `tokenizers` library (`from tokenizers.pre_tokenizers import ByteLevel` and `from tokenizers.decoders import ByteLevel as ByteLevelDecoder`).

This byte-level Byte-Pair Encoding (BPE) operates directly on raw UTF-8 bytes, allowing the vocabulary to encode any text sequence natively without ever failing on unknown tokens.

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Initialize BPE tokenizer with direct ByteLevel pre-tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=50257,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=ByteLevel.alphabet()
)

tokenizer.decoder = ByteLevelDecoder()

tokenizer.train(
    files=["data.txt"],
    trainer=trainer
)

tokenizer.save("tokenizer.json")
```

### Single Special Token Paradigm: `<|endoftext|>`

Unlike GPT-1 or BERT which used multiple special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`, `[CLS]`, `[SEP]`, `[MASK]`), this model only contains **one special token**: `<|endoftext|>`.

It functions as both:
1. **Document Boundary**: Separating independent text articles in pre-training data.
2. **Padding Token**: Aligning variable sentence lengths inside dataset batches.

### Example: Dataset Handling & Target Masking

In `GPT2Dataset`, `<|endoftext|>` is used for sequence padding. To ensure padded positions do not contribute to loss gradients, target token IDs equal to `<|endoftext|>` are mapped to `-100` (which PyTorch's `CrossEntropyLoss` ignores):

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
        
        # Mask padding tokens so loss is not computed on them
        target_ids[target_ids == self.pad_id] = -100

        return input_ids, target_ids
```

---

## Causal Attention Masking (BERT-Inspired `-inf` Strategy)

To ensure autoregressive language modeling properties (tokens cannot look ahead into the future), an upper-triangular causal attention mask is generated dynamically during the forward pass:

```python
attn_mask = torch.triu(
    torch.full((seq_len, seq_len), float('-inf'), device=x.device),
    diagonal=1
)
```

### Why `float('-inf')`?

We have used `float('-inf')` taking inspiration from the BERT model and original Transformer attention masking mechanics. 

While BERT uses masking bidirectionally to allow tokens to view left and right contexts simultaneously, GPT-2 adapts this masking matrix to enforce causality left-to-right. By setting future sequence positions (diagonal index 1 and above) to `float('-inf')`, the multi-head self-attention Softmax computation:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{attn\_mask}\right) V$$

converts all future positions to $e^{-\infty} = 0$ probability. This prevents information leakage while enabling full bidirectional-style attention access over all preceding tokens up to the current position.

---

## Architecture & Residual Connections

The model architecture faithfully implements the 12-layer decoder structure with Pre-Layer Normalization and residual connections:

- **`embeddingGPT`**: Combines token embeddings (`nn.Embedding(vocab_size, 1024)`) with trainable positional embeddings (`nn.Embedding(512, 1024)`).
- **`feedforward`**: A two-layer position-wise MLP expanding hidden dimensions from $1024 \to 4096 \to 1024$ with GELU activation and dropout:
  $$\text{FFN}(x) = \text{Dropout}(\text{Linear}_2(\text{GELU}(\text{Linear}_1(x))))$$
- **`Transformer` Block (With Residual Connections)**: As specified in the GPT-2 paper, Pre-LN residual connections are wrapped around self-attention and feed-forward sub-layers:
  
  ```python
  class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
      super().__init__()
      self.layer_norm1 = nn.LayerNorm(d_model)
      self.masked_attn = nn.MultiheadAttention(
          embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
      )
      self.layer_norm2 = nn.LayerNorm(d_model)
      self.feedforward = feedforward(d_model, dropout=dropout)
      self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
      # Residual Connection 1: Pre-LN Masked Self-Attention
      normed = self.layer_norm1(x)
      attn_out, _ = self.masked_attn(
          normed, normed, normed, attn_mask=attn_mask, need_weights=False
      )
      x = x + self.dropout(attn_out)

      # Residual Connection 2: Pre-LN Feedforward Network
      normed2 = self.layer_norm2(x)
      ff_out = self.feedforward(normed2)
      x = x + ff_out

      return x
  ```

- **`GPT2` Backbone**: Stacks 12 Transformer blocks (`d_model = 1024`, `num_heads = 16`), followed by a final Layer Normalization layer (`ln_f`) and Linear output projection (`lm_head`).

---

## Training and Temperature-Controlled Generation

### Training Setup & Model Checkpoints

- **Dataset**: OpenWebText ([`Skylion007/openwebtext`](https://huggingface.co/datasets/Skylion007/openwebtext))
- **Optimizer**: `AdamW` ($\text{lr} = 3 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\text{weight\_decay} = 0.1$)
- **Gradient Clipping**: `clip_grad_norm_(max_norm=1.0)`
- **Loss Function**: `CrossEntropyLoss(ignore_index=-100)`
- **Model Checkpoints**: Model weights and checkpoints are hosted on Hugging Face: [sujalgawas/gpt2-from-scratch-checkpoints](https://huggingface.co/sujalgawas/gpt2-from-scratch-checkpoints)

### Training Execution & Loss Progress

The model was trained on an NVIDIA T4 GPU (15 GB VRAM) for 45 hours completing 1 epoch. The training time and dataset scale were significantly reduced compared to the original GPT-2 paper due to memory limits (15 GB VRAM ceiling) and data constraints.

- **Initial Validation Loss**: 10.88
- **Final Epoch 1 Loss**:
  ```text
  === Epoch 1 done | Train loss 4.4881 | Val loss 4.3982 ===
  ```
During the single epoch training run, the validation loss dropped from **10.88** down to **4.3982**.

### Autoregressive Text Generation with Temperature Scaling

During generation, next-token predictions are adjusted via **temperature scaling**:

$$\text{logits}_{\text{scaled}} = \frac{\text{logits}}{\text{temperature}}$$

- **Low Temperature ($T < 0.7$)**: Sharpens probability distribution. The model prefers **high accuracy to the original trained data**, resulting in more deterministic, confident, and factual text.
- **High Temperature ($T \ge 0.8$)**: Flattens probability distribution, introducing **creativity, randomness, and diverse text generation**.

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
        # Apply temperature scaling to control creativity vs data fidelity
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

        # Early stopping when single special token <|endoftext|> is generated
        if next_token.item() == eos_id:
            break

    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)

# Generation example
output = generate("I was happy ", max_new_tokens=100, temperature=0.8, top_k=50)
print(output)
```

### Sample Generation Output

**Input Prompt:** `"I was happy "`  
**Generation Parameters:** `max_new_tokens=100`, `temperature=0.8`, `top_k=50`

**Output:**
> "I was happy iced by the idea of a project with one of the best design options. My favorite brand is the kind of product I’ve used to offer and is a great product and I’m not sure how to make it. I decided to be a friend in my head that I still have the same experience and the ability to build a product and some of my product. I wanted to make something that was perfect because I wanted to make something that was great with the product."