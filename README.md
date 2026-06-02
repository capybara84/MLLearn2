# MLLearn2

『Attention Is All You Need』を読めるようになるために、Transformer に必要な数学を順番に学ぶための教材です。

最初から Attention の式へ進むのではなく、数、スカラー、ベクトル、行列、内積、線形変換、softmax、確率分布、損失関数、微分、勾配降下法、連鎖律、正規化を積み上げて、最後に Self-Attention と Transformer 実装へ接続します。

## まず読むもの

- [リンク付き目次](Table%20of%20Contents.md): 全章と節へ移動するための目次
- [用語集](Glossary.md): 章を読む途中で確認したい基本語の一覧
- [サンプルコード](examples/README.md): 本文で扱う計算を手元で確認するための PyTorch コード

## 読み方

1. [第1章 なぜTransformerに数学が必要なのか](Chapter%201%20-%20Why%20Does%20a%20Transformer%20Need%20Mathematics.md) から順番に読む
2. 各章の Mermaid 図で概念の流れを確認する
3. PyTorch サンプルコードを手元で少し変えて動かす
4. 各章のまとめや確認コードで理解を確認する
5. [第14章 Attentionの数式を読む](Chapter%2014%20-%20Reading%20the%20Math%20Behind%20Attention.md) で `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V` に戻る
6. [第15章 実装で確認する数学](Chapter%2015%20-%20Verifying%20the%20Math%20Through%20Implementation.md) と [第16章 Transformer実装に進む前の確認](Chapter%2016%20-%20Preparing%20to%20Implement%20Transformers.md) で実装へ進む準備をする

## 構成

- 第1章から第3章: Transformer に数学が必要な理由、数、スカラー、ベクトル
- 第4章から第6章: 内積、行列、線形変換
- 第7章から第9章: softmax、確率分布、損失関数とクロスエントロピー
- 第10章から第12章: 微分、勾配降下法、合成関数と連鎖律
- 第13章: 正規化と Layer Normalization
- 第14章: Self-Attention と `Attention Is All You Need` の中心式への接続
- 第15章から第16章: 実装での確認と Transformer 実装前の整理

## 実行環境

章内のサンプルコードは、Python と PyTorch で動かすことを想定しています。

```bash
python3 -c "import torch; print(torch.__version__)"
```

コードは小さな確認用なので、GPU は不要です。CPU だけで動くようにしています。

```bash
python3 examples/01_softmax.py
python3 examples/02_cross_entropy.py
python3 examples/03_attention.py
python3 examples/04_transformer_block.py
```

## この教材で重視していること

- 数式より先に、shape、入力、出力、計算の意味を見る
- ベクトル、行列、内積、softmax が Attention のどこで使われるかを何度も確認する
- 予測、損失、勾配、パラメータ更新の流れを実装と結びつけて理解する
- `QK^T`、`sqrt(d_k)`、mask、LayerNorm、残差接続を Transformer への橋として理解する
- 章ごとに図、コード、まとめで同じ概念を別の角度から見る
