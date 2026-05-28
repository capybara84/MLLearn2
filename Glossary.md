# 用語集

この用語集では、本文で何度も出てくる言葉を短く整理します。

## Attention score

QueryとKeyの内積から作るスコアです。

```text
scores = QK^T / sqrt(d_k)
```

まだ重みではなく、softmaxに入れる前の値です。

## Attention weight

Attention scoreにsoftmaxをかけた重みです。

```text
weights = softmax(scores)
```

各Queryが各Keyをどれくらい見るかを表します。

## backpropagation

lossから前の層へ向かって勾配を伝える仕組みです。

PyTorchでは、多くの場合 `loss.backward()` がこれを実行します。

## batch_size

一度にまとめて処理するデータの個数です。

Transformerでは、よく次のshapeの先頭に出てきます。

```text
[batch_size, seq_len, d_model]
```

## causal mask

未来のトークンを見ないようにするmaskです。

Decoder-only Transformerや次トークン予測で使います。

```text
scores = scores.masked_fill(mask == 0, -inf)
```

## cross entropy

正解に割り当てた確率を見て、予測の悪さを数値化するlossです。

正解トークンの確率が高いほど、lossは小さくなります。

## d_model

各トークンを表すベクトルの次元数です。

`seq_len` がトークンの個数で、`d_model` が各トークンのベクトルの長さです。

## embedding

トークンIDをベクトルに変換する仕組みです。

```text
token_ids: [batch_size, seq_len]
↓
x: [batch_size, seq_len, d_model]
```

## Feed Forward Network

Transformer blockの中で、各位置のベクトルをさらに変換する小さなニューラルネットワークです。

Attentionの後に使われます。

## gradient

パラメータを少し変えたとき、lossがどちら向きに増えやすいかを表す量です。

学習では、基本的に勾配の逆方向へパラメータを動かします。

## Key

Attentionで使うベクトルの1つです。

QueryとKeyの内積によって、トークン同士の相性を計算します。

## LayerNorm

各トークンのベクトルごとに、値のスケールを整える正規化です。

Transformerでは、通常 `d_model` 方向に平均と分散を計算します。

## logits

softmaxをかける前のスコアです。

言語モデルでは、各位置について語彙全体へのlogitsを出します。

```text
logits: [batch_size, seq_len, vocab_size]
```

## loss

予測がどれくらい悪いかを表す値です。

学習では、このlossが小さくなるようにパラメータを更新します。

## mask

見てはいけない位置や無視したい位置を隠すための仕組みです。

Attentionでは、softmax前のscoreに対して使うことが多いです。

## Multi-Head Attention

Attentionを複数のheadで並列に行う仕組みです。

複数の見方でトークン同士の関係を捉えるために使います。

## optimizer

勾配を使ってパラメータを更新する部品です。

PyTorchでは、たとえば `torch.optim.AdamW` などを使います。

## positional encoding

トークンの位置情報を表すベクトルです。

TransformerはRNNのように順番に読むわけではないので、位置情報を別途足します。

## Query

Attentionで使うベクトルの1つです。

直感的には「自分が探しているもの」を表します。

## residual connection

入力をサブレイヤーの出力に足す仕組みです。

```text
x + Sublayer(x)
```

深いネットワークでも情報や勾配を流しやすくします。

## seq_len

1つの文や系列に含まれるトークン数です。

```text
[batch_size, seq_len, d_model]
```

の真ん中の次元です。

## shape

テンソルの形です。

Transformer実装では、shapeを追えることが非常に重要です。

## softmax

スコアの列を、合計1の重みに変換する関数です。

Attentionでは、scoreをAttention weightに変換するために使います。

## tensor

スカラー、ベクトル、行列を一般化した多次元配列です。

PyTorchでは、多くのデータが `torch.Tensor` として扱われます。

## Value

Attentionで使うベクトルの1つです。

Attention weightに応じて混ぜられる「実際に渡す中身」です。

## vocab_size

語彙に含まれるトークンの個数です。

言語モデルの出力層では、各位置ごとに `vocab_size` 個のlogitsを出します。
