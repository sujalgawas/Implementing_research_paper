# PyTorch Implementation of GPT-1 for Autoregressive Language Modeling

This repository contains a PyTorch implementation of OpenAI's **GPT-1** model, built as a hands-on learning project to reconstruct the decoder-only Transformer architecture proposed in the seminal paper [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language_understanding_paper.pdf) by Radford et al. (2018).

The goal of this project was to implement the generative pre-training workflow, causal self-attention mechanics, and learned positional embeddings from scratch for next-token text generation on the BookCorpus dataset.

---

## Project Philosophy: Why This Implementation?

Rather than re-inventing low-level matrix multiplication or writing custom attention matrix math from memory, the primary goal was gaining a **deep conceptual understanding** of GPT-1's architectural innovations:

- **Decoder-Only Design**: Shifting away from the encoder-decoder paradigm of the original Transformer to an autoregressive, decoder-only stack.
- **Learned Positional Embeddings**: Replacing fixed sinusoidal functions with trainable 1D positional embeddings.
- **Causal Masking & GELU Activations**: Utilizing GELU non-linearities in feed-forward layers and upper-triangular masking to ensure tokens can only attend to previous positions.

To achieve this cleanly, I:
- Built the **`GPTEmbedding`**, **`FeedForward`**, **`Transfomer` (Decoder Block)**, and **`GPT1`** top-level model wrapper.
- Leveraged PyTorch's native `nn.MultiheadAttention` and `nn.LayerNorm` components for numerical stability, clean code, and fast execution.
- Handled critical implementation details manually: custom BPE tokenizer pipeline, causal masking (`torch.triu`), target-shifted dataset preparing (`x = ids[:-1]`, `y = ids[1:]`), and an autoregressive text generation loop.

---

## Dataset and Bugs Worth Mentioning

The model is trained on the [`bookcorpus/bookcorpus`](https://huggingface.co/datasets/bookcorpus/bookcorpus) dataset, a large collection of unpublished novels ideal for language modeling pre-training.

### Bug 1: Float Division in Dataset Splitting

When splitting the dataset into training and validation subsets using PyTorch's `random_split`, I calculated `train_len = 0.8 * length`. This resulted in a floating-point number (e.g., `8000.0`), causing `random_split` to throw a type error because it expects integer lengths.

**The Fix:**

Explicitly casting the result to `int`:

```python
length = len(book_corpus_dataset)
train_len = int(0.8 * length)
val_len = length - train_len

train_dataset, val_dataset = random_split(book_corpus_dataset, [train_len, val_len])
```

### Bug 2: Sequence Length Padding and Alignment in Dataset

During initial DataLoader execution, batching failed due to variable sentence lengths. Additionally, autoregressive training requires shifted input-target pairs (`input_ids` and `target_ids`) of fixed context window length.

**The Fix:**

I added a custom `_pad` helper method inside the `BookCorpus` PyTorch `Dataset` class to enforce a maximum sequence length of 512 tokens and split the tokens for next-token prediction:

```python
class BookCorpus(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset["train"]
        self.pad_len = tokenizer.token_to_id("<pad>")

    def _pad(self, ids, pad_id):
        ids = ids[:512]
        ids = ids + [pad_id] * (512 - len(ids))
        return ids

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        tokens = spacy_tokenizer(text)
        encoding = tokenizer.encode(" ".join(tokens))
        
        ids = self._pad(encoding.ids, self.pad_len)

        # Shifted sequences for autoregressive language modeling
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)

        return input_ids, target_ids
```

---

## Tokenization

For subword tokenization, I trained a custom **Byte-Pair Encoding (BPE)** model using HuggingFace's `tokenizers` library (`BpeTrainer`), combined with `spaCy` (`en_core_web_sm`) for initial text pre-tokenization.

The following special tokens were added to the vocabulary:
- `<pad>` — Padding token used to align sequences in a batch to 512 length.
- `<unk>` — Unknown token fallback for out-of-vocabulary words.
- `<bos>` — Beginning of Sentence token.
- `<eos>` — End of Sentence token.

Vocabulary size: **8,000**

---

## Architecture

The model architecture faithfully implements the 12-layer decoder-only structure described in the GPT-1 paper:

- **`GPTEmbedding`**: Combines token embeddings (`nn.Embedding(vocab_size, 768)`) with trainable positional embeddings (`nn.Embedding(512, 768)`).
- **`FeedForward`**: A two-layer position-wise network projecting dimensions from $768 \to 3072 \to 768$, utilizing the **GELU** activation function and $0.1$ dropout:
  $$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$
- **`Transfomer` Block**: A Transformer decoder block consisting of masked multi-head self-attention (`nn.MultiheadAttention`, `batch_first=True`), residual connections, and layer normalization (`nn.LayerNorm`).
- **`GPT1`**: The complete architecture stacking 12 decoder blocks with `d_model = 768` and `num_heads = 4`, followed by a final linear projection head (`lm_head`) mapping representations back to the vocabulary logits space.

---

## Training and Inference

### Training Setup

- **Optimizer**: `AdamW` ($\text{lr} = 2.5 \times 10^{-4}$, $\text{weight\_decay} = 0.01$)
- **Loss Function**: `CrossEntropyLoss` evaluated over flattened logits and target tokens:
  ```python
  loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
  ```
- **Context Length**: 512 tokens
- **Batch Size**: 4

### Autoregressive Inference

For text generation, I implemented a greedy decoding loop (`inferencing`). It tokenizes the prompt, constructs the upper-triangular causal attention mask, feeds the sequence into the model, predicts the next token via `argmax`, and appends it iteratively:

```python
@torch.inference_mode()
def inferencing(txt, max_tokens, model):
    model.eval()
    txt = spacy_tokenizer(txt)
    encoding = tokenizer.encode(" ".join(txt))
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        attn_mask = make_mask(input_ids)
        logits = model(input_ids, attn_mask)
        next_token = logits[:, -1].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

    return tokenizer.decode(input_ids[0].tolist())
```
