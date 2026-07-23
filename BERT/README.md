# PyTorch Implementation of BERT for Masked Language Modeling

This repository contains a PyTorch implementation of Google's **BERT** (Bidirectional Encoder Representations from Transformers) model, built as a hands-on learning project to reconstruct the encoder-only architecture proposed in the seminal paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Devlin et al. (2018).

The goal of this project was to implement bidirectional self-attention, token masking (MLM 80-10-10 rule), segment embeddings, and key padding mechanics from scratch for masked token filling on Wikipedia text.

---

## Project Philosophy: Why This Implementation?

Unlike autoregressive models (like GPT) that process text left-to-right with causal masks, BERT relies on **bidirectional context** — allowing each token to attend to tokens both to its left and to its right simultaneously.

The primary goal of this project was gaining a **deep conceptual understanding** of BERT's core pre-training mechanics:

- **Bidirectional Encoder Stack**: Utilizing an encoder-only Transformer architecture with unmasked self-attention across the full sequence.
- **Tri-Embedding Summation**: Combining Token Embeddings, Segment Embeddings (for sequence pairs), and Learned Positional Embeddings.
- **Masked Language Modeling (MLM)**: Implementing the paper's 80-10-10 masking strategy to prevent data leakage during training while learning rich contextual representations.

To achieve this cleanly, I:
- Implemented **`BertEmbeddings`**, **`FeedForward`**, **`Transformer` (Encoder Layer)**, and **`BERT`** backbone wrapper from scratch.
- Leveraged PyTorch's native `nn.MultiheadAttention` and `nn.LayerNorm` components for numerical stability and code readability.
- Handled key technical details manually: WordPiece tokenization, custom padding mask conversion (`key_padding_mask`), PyTorch loss target masking with `-100`, and a `fill_mask` inference pipeline.

---

## Dataset and Implementation Details Worth Mentioning

The model is trained on a 5,000-sample subset of English Wikipedia from HuggingFace [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.en`).

### 1. The 80-10-10 MLM Masking Strategy

During dataset item construction in `BERTDataset`, 15% of non-special tokens are selected for masking (excluding `[CLS]`, `[SEP]`, and `[PAD]`). Of the selected tokens:
- **80%** are replaced with the `[MASK]` token token ID.
- **10%** are replaced with a random token ID from the vocabulary.
- **10%** remain unchanged.

Unmasked token positions are assigned a label value of `-100` so PyTorch's `CrossEntropyLoss` automatically ignores them during gradient calculations:

```python
# Never mask CLS / SEP / PAD special tokens
special = (input_ids == CLS_TOKEN_ID) | (input_ids == SEP_TOKEN_ID) | (input_ids == PAD_TOKEN_ID)

prob = torch.full(input_ids.shape, self.mlm_prob)
prob.masked_fill_(special, 0.0)
masked_indices = torch.bernoulli(prob).bool()

labels[~masked_indices] = -100  # PyTorch ignores -100 in loss evaluation

# 80% -> [MASK]
replace_with_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
input_ids[replace_with_mask] = MASK_TOKEN_ID

# 10% -> random token, 10% -> original token
replace_with_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~replace_with_mask
random_tokens = torch.randint(VOCAB_SIZE, input_ids.shape, dtype=torch.long)
input_ids[replace_with_random] = random_tokens[replace_with_random]
```

### 2. Key Padding Mask Inversion

In PyTorch's `nn.MultiheadAttention`, `key_padding_mask` expects `True` for positions that should be **ignored/masked out** (`[PAD]`), and `False` for valid content tokens.

Since standard attention masks represent valid tokens as `True` and padding as `False`, I created a helper function to invert the boolean tensor before passing it to the Transformer encoder blocks:

```python
def get_padding_mask(attention_mask):
    # Converts True (real token) -> False (don't mask), False ([PAD]) -> True (mask out)
    return ~attention_mask
```

---

## Tokenization

For subword tokenization, I trained a custom **WordPiece** tokenizer using HuggingFace's `tokenizers` library (`WordPieceTrainer`) with `Whitespace` pre-tokenization on Wikipedia text.

The following special tokens were included in the vocabulary:
- `[UNK]` — Unknown token fallback for out-of-vocabulary words.
- `[CLS]` — Classification token prepended to the start of every sequence.
- `[SEP]` — Sentence separator / sequence termination token appended to sequences.
- `[PAD]` — Padding token used to align sequences in a batch to 128 length.
- `[MASK]` — Mask token used to obscure words during MLM pre-training.

Vocabulary size: **8,000**

---

## Architecture

The model architecture implements a 12-layer bidirectional Transformer encoder:

- **`BertEmbeddings`**: Computes the element-wise sum of three embedding layers:
  $$\text{Embedding} = \text{TokenEmbed}(x) + \text{SegmentEmbed}(\text{seg}) + \text{PositionalEmbed}(\text{pos})$$
- **`FeedForward`**: A two-layer position-wise MLP projecting $768 \to 4096 \to 768$ with **GELU** non-linearity:
  $$\text{FFN}(x) = \text{Linear}_2(\text{GELU}(\text{Linear}_1(x)))$$
- **`Transformer` Block**: Encoder layer featuring unmasked bidirectional multi-head self-attention (`nn.MultiheadAttention`, `batch_first=True`), residual connections, and layer normalization (`nn.LayerNorm`).
- **`BERT` Backbone**: Stacks 12 encoder blocks (`d_model = 768`, `num_head = 4`), followed by a Linear projection head (`mlm_head`) mapping representations to `vocab_size` for masked token prediction.

---

## Training and Inference

### Training Setup

- **Optimizer**: `AdamW` ($\text{lr} = 5 \times 10^{-5}$, $\text{weight\_decay} = 0.01$)
- **Loss Function**: `CrossEntropyLoss` over flattened prediction logits and target labels (ignoring `-100` positions).
- **Max Sequence Length**: 128 tokens
- **Batch Size**: 128

Training loss progression:

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1     | 7.8601     | 7.4743   |
| 2     | 7.2805     | 7.1285   |
| 3     | 7.0913     | 7.0571   |

### Fill-Mask Inference

For masked token prediction, I implemented `fill_mask()`. It encodes the prompt with `[CLS]` and `[SEP]`, constructs segment and padding masks, locates `[MASK]` token positions, evaluates output logits, and substitutes predicted token IDs:

```python
@torch.inference_mode()
def fill_mask(txt, model):
    model.eval()
    encoding = tokenizer.encode(txt)
    ids = [CLS_TOKEN_ID] + encoding.ids + [SEP_TOKEN_ID]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    segment_ids = torch.zeros_like(input_ids)
    key_padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    logits = model(input_ids, segment_ids, key_padding_mask)

    predicted_ids = input_ids.clone()
    mask_positions = (input_ids[0] == MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
    for pos in mask_positions:
        predicted_ids[0, pos] = logits[0, pos].argmax(dim=-1)

    return tokenizer.decode(predicted_ids[0].tolist())

# Example evaluation
print(fill_mask("the capital of france is [MASK] .", model))
```
