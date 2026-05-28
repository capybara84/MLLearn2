# Examples

このディレクトリには、本文で扱う数学を小さく実行して確認するためのコードを置いています。

```bash
python3 examples/01_softmax.py
python3 examples/02_cross_entropy.py
python3 examples/03_attention.py
python3 examples/04_transformer_block.py
```

各ファイルは、Transformer実装に進む前に確認しておきたい最小単位です。

- `01_softmax.py`: softmaxがスコアを重みに変えることを確認する
- `02_cross_entropy.py`: logits、targets、cross entropy lossの関係を確認する
- `03_attention.py`: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V` を確認する
- `04_transformer_block.py`: Self-Attention、LayerNorm、残差接続、Feed Forward Networkを組み合わせる
