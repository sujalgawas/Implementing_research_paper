# English-to-Japanese Machine Translation with a Custom Transformer

This repository contains a PyTorch implementation of the Transformer model, built as a hands-on learning project to reconstruct the architecture proposed in the seminal research paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Some implementation patterns and structures are inspired by [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) by Harvard NLP.

The goal of this project was to translate English text into Japanese using the Kyoto Free Translation Task (KFTT) dataset.

---

## Project Philosophy: Why This Implementation?

While many implementations build every single layer from scratch — including attention heads and layer normalization — the purpose of this project was different. The goal was **deep conceptual understanding** of the Transformer's global flow: token masking, seq2seq interaction, and positional mechanics, rather than writing mathematical formulas from memory without any reference.

To achieve this, I:
- Implemented the overall **Encoder**, **Decoder**, and **Positional Encoding** wrapper blocks myself.
- Used PyTorch's native `nn.MultiheadAttention` and `nn.LayerNorm` components. This keeps the implementation cleaner, more numerically stable, and highly readable.
- Handled the critical implementation details myself: causal masking for self-attention, source/target padding masks, tokenization pipelines, and the custom learning rate warm-up scheduler.

---

## Dataset and a Bug Worth Mentioning

The model is trained on the Japanese-English translation dataset [`nntsuzu/KFTT`](https://huggingface.co/datasets/nntsuzu/KFTT) (Kyoto Free Translation Task).

**The Bug**

During initial training runs, I encountered an issue where the training loss would suddenly output `NaN` or null values mid-epoch. After investigating the dataset, I found that it contained a handful of lines consisting solely of whitespace characters. When tokenized, these empty inputs resulted in empty sequences, causing undefined behavior in the loss function.

**The Fix**

I filtered the dataset prior to tokenization, ensuring both the English and Japanese sides of each pair contained a non-empty, stripped string:

```python
def is_valid(example):
    t = example['translation']
    return len(t['en'].strip()) > 0 and len(t['ja'].strip()) > 0

dataset['train'] = dataset['train'].filter(is_valid)
```

---

## Tokenization

For subword tokenization, I used the **WordPiece** tokenizer, which matches the tokenizer family described in the original paper.

The paper trained on English and German, and I am not entirely sure whether they used a single shared vocabulary for both languages or kept them separate. In this project, since English and Japanese use completely different writing systems and scripts, I trained separate WordPiece tokenizers for each language.

The following special tokens were added to both vocabularies:

- `[PAD]` — Padding token, used to align sequences in a batch to equal length.
- `[SOS]` — Start of Sentence, passed to the decoder to kick off generation.
- `[EOS]` — End of Sentence, tells the model when to stop generating.
- `[UNK]` — Unknown token, fallback for words outside the vocabulary.

Resulting vocabulary sizes:
- English: **8,000**
- Japanese: **11,862**

---

## Architecture

The network follows the classic encoder-decoder structure from the original paper:

**Positional Encoding** — A custom module that adds sinusoidal patterns to raw token embeddings to inject information about the position of each token in the sequence.

**Encoder (`EngToJapEncoder`)** — Learns a context-rich representation of the English source sentence using self-attention, residual connections, layer normalization, and a position-wise feed-forward network.

**Decoder (`EngToJapDecoder`)** — Generates Japanese tokens autoregressively. It uses masked self-attention (so the model cannot look at future tokens during training), followed by cross-attention that queries the encoder's output using the current decoder state.

**Transformer** — A wrapper class that connects the encoder and decoder into a single end-to-end model.

---

## Training

The model was trained for **50 epochs** on a GPU.

Optimizer configuration, following the paper:
- Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$
- Dynamic learning rate with a warmup phase of 4,000 steps as defined in the paper:

$$d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5},\ \text{step} \cdot \text{warmup\_steps}^{-1.5})$$

- Loss: `CrossEntropyLoss` with the `[PAD]` index ignored.

Loss improved steadily across all 50 epochs:

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1     | 4.6839     | 3.4952   |
| 10    | 2.3267     | 2.3874   |
| 25    | 2.0629     | 2.2540   |
| 50    | 1.9099     | 2.1938   |

The model clearly has room to improve with more training. However, I stopped at 50 epochs intentionally — I had not set up model checkpointing properly at the time, and continuing to train without saving would have wasted compute for no lasting benefit.

**A note on the `##` tokens in the output**

When running the translation inference, you will notice `##` prefixes on many output tokens (e.g., `そ ##の ##は`). This is not a bug — it is how WordPiece represents subword fragments. The `##` prefix means that token is a continuation of the previous word rather than the start of a new one.

In a well-trained model, the decoder would handle these cleanly and a post-processing step would merge them back. In this case, the model is under-trained and the fragments surface as-is in the raw output. Given the training constraints, I left this as-is rather than invest more compute without proper model saving in place.
