# Attention Is All You Need (2017)

The original encoder–decoder transformer, built from the paper: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

## Files

- `model.py` — embeddings, sinusoidal positional encoding, layer norm, feed-forward, multi-head attention,
  residual connections, encoder, decoder, projection layer, and `build_transformer()`.

Training code, the dataset pipeline and config are not written yet.

## Paper hyperparameters (base model)

| | |
|---|---|
| `d_model` | 512 |
| `N` (layers) | 6 encoder, 6 decoder |
| `h` (heads) | 8 |
| `d_k` = `d_v` | 64 |
| `d_ff` | 2048 |
| dropout | 0.1 |
| FFN activation | ReLU |

## Deviations from the paper

- **Pre-norm residuals.** `ResidualConnection` computes `x + dropout(sublayer(norm(x)))`. The paper applies
  the norm after the residual add. Pre-norm trains more stably and is what most reimplementations use; it is
  also why `Encoder` and `Decoder` each need a final norm.
- **No weight sharing** between the two embedding layers and the output projection (paper, §3.4).
- **Scalar LayerNorm gain and bias** rather than one per feature.
