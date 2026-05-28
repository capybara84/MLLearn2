# 第16章 Transformer実装に進む前の確認

## 16.1 この章の目的

この章では、ここまで学んだ内容を整理します。

この教科書の目的は、数学そのものを深く極めることではありません。

目的は、次の式を読めるようになることでした。

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

そして、この式をPyTorchで実装できるようになることでした。

この式には、次のような要素が含まれています。

```text
ベクトル
行列
内積
転置
行列積
softmax
shape
線形変換
確率分布
勾配
正規化
```

ここまでの章で、それぞれを一通り見てきました。

この章では、それらをもう一度、Transformer実装に必要な形で確認します。

特に重要なのは、次の3つです。

```text
shapeを追えること
Attentionの式をコードにできること
学習ループの意味がわかること
```

Transformerを実装するとき、細かい数式をすべて暗記している必要はありません。

しかし、テンソルのshapeがどう変化するかを追えないと、実装はすぐに詰まります。

たとえば、Self-Attentionでは次のようなshapeが出てきます。

```text
x:       [batch_size, seq_len, d_model]
q:       [batch_size, seq_len, d_k]
k:       [batch_size, seq_len, d_k]
v:       [batch_size, seq_len, d_v]

scores:  [batch_size, seq_len, seq_len]
weights: [batch_size, seq_len, seq_len]
out:     [batch_size, seq_len, d_v]
```

この流れを読めることが、Transformer実装の土台です。

また、Transformerは単にforward計算を書けば終わりではありません。

言語モデルとして学習させるには、次の流れも必要です。

```text
token_ids
↓
inputs, targetsを作る
↓
model(inputs)
↓
logits
↓
cross entropy loss
↓
loss.backward()
↓
optimizer.step()
```

この章では、ここまでの内容を確認し、次に進む準備を整えます。

---

## 16.2 スカラー・ベクトル・行列・テンソルの確認

まず、データの形を確認します。

スカラーは1つの数です。

```text
3.14
```

ベクトルは、数を一列に並べたものです。

```text
[1.0, 2.0, 3.0]
```

行列は、数を縦横に並べたものです。

```text
[
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0]
]
```

テンソルは、それらを一般化した多次元配列です。

Transformerでは、主に次のshapeをよく使います。

```text
[batch_size, seq_len, d_model]
```

これは、次の意味です。

```text
batch_size:
まとめて処理する文の数

seq_len:
各文のトークン数

d_model:
各トークンを表すベクトルの次元数
```

たとえば、次のshapeを考えます。

```text
[2, 5, 8]
```

これは、次の意味です。

```text
2個の文
各文は5トークン
各トークンは8次元ベクトル
```

PyTorchでは、次のように作れます。

```python
import torch

batch_size = 2
seq_len = 5
d_model = 8

x = torch.randn(batch_size, seq_len, d_model)

print(x.shape)
```

出力は次のようになります。

```text
torch.Size([2, 5, 8])
```

Transformerを実装するときは、常にこのshapeを意識します。

```text
今のテンソルは何を表しているのか
各次元は何を意味しているのか
次の処理でshapeはどう変わるのか
```

これを追えることが重要です。

---

## 16.3 embeddingの確認

Transformerは、文字列をそのまま処理するわけではありません。

まず、文章をトークン列にします。

```text
"I love dogs"
↓
["I", "love", "dogs"]
```

次に、各トークンを整数IDに変換します。

```text
"I"    → 12
"love" → 45
"dogs" → 98
```

すると、文は整数列になります。

```text
[12, 45, 98]
```

しかし、この整数IDをそのまま意味のある数値として扱うわけではありません。

`98` が `45` より大きいからといって、意味的に大きいわけではありません。

そこで、embeddingを使って、各トークンIDをベクトルに変換します。

```text
token_id
↓
embedding
↓
embedding vector
```

PyTorchでは、`nn.Embedding` を使います。

```python
import torch
import torch.nn as nn

vocab_size = 100
d_model = 8

embedding = nn.Embedding(vocab_size, d_model)

token_ids = torch.tensor([
    [12, 45, 98],
    [8, 21, 4],
])

x = embedding(token_ids)

print("token_ids:", token_ids.shape)
print("x:", x.shape)
```

出力は次のようになります。

```text
token_ids: torch.Size([2, 3])
x: torch.Size([2, 3, 8])
```

shapeの変化は次の通りです。

```text
token_ids:
[batch_size, seq_len]

embedding後:
x: [batch_size, seq_len, d_model]
```

この `x` がTransformer blockに入る基本の入力です。

```text
token_ids
↓
embedding
↓
x
↓
Transformer blocks
```

つまり、Transformer実装の入口は次の形です。

```text
[batch_size, seq_len]
↓
[batch_size, seq_len, d_model]
```

この変換を理解しておくことが重要です。

---

## 16.4 線形変換の確認

Transformerでは、線形変換が何度も出てきます。

PyTorchでは、線形変換は `nn.Linear` で書きます。

```python
linear = nn.Linear(in_features, out_features)
```

これは、最後の次元を `in_features` から `out_features` に変換します。

たとえば、入力が次のshapeだとします。

```text
x: [batch_size, seq_len, d_model]
```

ここに、

```python
nn.Linear(d_model, d_k)
```

を適用すると、出力は次のshapeになります。

```text
[batch_size, seq_len, d_k]
```

PyTorchで確認します。

```python
import torch
import torch.nn as nn

batch_size = 2
seq_len = 5
d_model = 8
d_k = 4

x = torch.randn(batch_size, seq_len, d_model)

linear = nn.Linear(d_model, d_k)

y = linear(x)

print("x:", x.shape)
print("y:", y.shape)
```

出力は次のようになります。

```text
x: torch.Size([2, 5, 8])
y: torch.Size([2, 5, 4])
```

Transformerでは、この線形変換によってQ/K/Vを作ります。

```python
w_q = nn.Linear(d_model, d_k)
w_k = nn.Linear(d_model, d_k)
w_v = nn.Linear(d_model, d_v)

q = w_q(x)
k = w_k(x)
v = w_v(x)
```

shapeは次のようになります。

```text
x: [batch_size, seq_len, d_model]

q: [batch_size, seq_len, d_k]
k: [batch_size, seq_len, d_k]
v: [batch_size, seq_len, d_v]
```

QとKは内積を取るので、最後の次元が同じである必要があります。

```text
q: [batch_size, seq_len, d_k]
k: [batch_size, seq_len, d_k]
```

Vは最後に混ぜられる中身です。

```text
v: [batch_size, seq_len, d_v]
```

このように、線形変換は、入力ベクトルを別の役割を持つベクトルに変換するために使われます。

```text
x
↓
q, k, v
```

---

## 16.5 内積と `QK^T` の確認

Attentionでは、QueryとKeyの内積で相性スコアを作ります。

1つのQueryと1つのKeyの内積は、次のように計算します。

```text
q・k = q1*k1 + q2*k2 + ... + qn*kn
```

内積は、2つのベクトルから1つのスカラーを作る計算です。

```text
[d_k] と [d_k]
↓
スカラー
```

Attentionでは、これをすべてのQueryとすべてのKeyについてまとめて計算します。

そのために使うのが次の行列積です。

```text
QK^T
```

1つの文だけなら、shapeは次の通りです。

```text
Q: [seq_len, d_k]
K: [seq_len, d_k]
```

Kを転置します。

```text
K^T: [d_k, seq_len]
```

すると、次の行列積ができます。

```text
QK^T: [seq_len, d_k] @ [d_k, seq_len]
```

結果は次のshapeになります。

```text
[seq_len, seq_len]
```

これは、トークン同士の相性スコア表です。

バッチ付きでは、次のようになります。

```text
q: [batch_size, seq_len, d_k]
k: [batch_size, seq_len, d_k]
```

PyTorchでは、最後の2次元を転置します。

```python
scores = q @ k.transpose(-2, -1)
```

shapeは次のように変わります。

```text
q:                   [batch_size, seq_len, d_k]
k.transpose(-2, -1): [batch_size, d_k, seq_len]

scores:              [batch_size, seq_len, seq_len]
```

PyTorchで確認します。

```python
import torch

batch_size = 2
seq_len = 4
d_k = 8

q = torch.randn(batch_size, seq_len, d_k)
k = torch.randn(batch_size, seq_len, d_k)

scores = q @ k.transpose(-2, -1)

print("q:", q.shape)
print("k:", k.shape)
print("scores:", scores.shape)
```

出力は次のようになります。

```text
q: torch.Size([2, 4, 8])
k: torch.Size([2, 4, 8])
scores: torch.Size([2, 4, 4])
```

この `scores` は、まだ重みではありません。

これは、softmax前の相性スコアです。

```text
Attention score
```

---

## 16.6 softmaxとAttention weightの確認

`QK^T` によってAttention scoreができました。

次に、これをsoftmaxで重みに変換します。

Attentionの式では、次の部分です。

```text
softmax(QK^T / sqrt(d_k))
```

まず、`sqrt(d_k)` で割ります。

```python
scores = scores / math.sqrt(d_k)
```

これは、内積スコアが大きくなりすぎるのを防ぐためです。

次に、softmaxをかけます。

```python
weights = torch.softmax(scores, dim=-1)
```

ここで、`dim=-1` は最後の次元にsoftmaxをかけるという意味です。

Attention scoreのshapeは次の通りでした。

```text
scores: [batch_size, seq_len, seq_len]
```

最後の `seq_len` は、

```text
各QueryがどのKeyを見るか
```

を表しています。

そのため、最後の次元にsoftmaxをかけます。

softmax後のshapeは変わりません。

```text
weights: [batch_size, seq_len, seq_len]
```

ただし、最後の次元の合計が1になります。

```python
import torch
import math

batch_size = 2
seq_len = 4
d_k = 8

q = torch.randn(batch_size, seq_len, d_k)
k = torch.randn(batch_size, seq_len, d_k)

scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d_k)

weights = torch.softmax(scores, dim=-1)

print("weights:", weights.shape)
print(weights.sum(dim=-1))
```

出力は次のようになります。

```text
weights: torch.Size([2, 4, 4])
tensor([[1.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000, 1.0000]])
```

これは、各Queryについて、全Keyへの重みの合計が1であることを意味します。

```text
weights[b, i, :]
```

は、

```text
b番目の文の
i番目のトークンが
各トークンをどれくらい見るか
```

を表します。

この `weights` がAttention weightです。

---

## 16.7 `weights @ V` の確認

Attention weightができたら、次にValueを混ぜます。

式では次の部分です。

```text
softmax(QK^T / sqrt(d_k))V
```

つまり、

```text
out = weights @ V
```

です。

shapeを確認します。

```text
weights: [batch_size, seq_len, seq_len]
v:       [batch_size, seq_len, d_v]
```

行列積をすると、次のshapeになります。

```text
out:     [batch_size, seq_len, d_v]
```

PyTorchで確認します。

```python
import torch

batch_size = 2
seq_len = 4
d_v = 8

weights = torch.randn(batch_size, seq_len, seq_len)
weights = torch.softmax(weights, dim=-1)

v = torch.randn(batch_size, seq_len, d_v)

out = weights @ v

print("weights:", weights.shape)
print("v:", v.shape)
print("out:", out.shape)
```

出力は次のようになります。

```text
weights: torch.Size([2, 4, 4])
v: torch.Size([2, 4, 8])
out: torch.Size([2, 4, 8])
```

意味としては、各トークンが、他のトークンのValueを重みに応じて混ぜています。

たとえば、あるトークンの重みが次のようだったとします。

```text
[0.6, 0.3, 0.1]
```

Valueが次のようだったとします。

```text
v1
v2
v3
```

出力は次のようになります。

```text
0.6*v1 + 0.3*v2 + 0.1*v3
```

つまり、AttentionはValueの重み付き和を作っています。

```text
Attention weight
↓
Valueをどれくらい混ぜるか
```

ここまでをまとめると、Self-Attentionの中心は次の4行です。

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
out = weights @ v
```

これが、次の式に対応しています。

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

---

## 16.8 causal maskの確認

Decoder-only Transformer、つまりGPT系の言語モデルでは、未来のトークンを見てはいけません。

そのため、causal maskを使います。

たとえば、`seq_len = 4` の場合、見てよい位置は次のようになります。

```text
[
  [1, 0, 0, 0],
  [1, 1, 0, 0],
  [1, 1, 1, 0],
  [1, 1, 1, 1]
]
```

これは、

```text
1番目のトークンは1番目だけを見る
2番目のトークンは1番目と2番目を見る
3番目のトークンは1番目から3番目を見る
4番目のトークンは1番目から4番目を見る
```

という意味です。

PyTorchでは、次のように作ります。

```python
import torch

seq_len = 4

mask = torch.tril(torch.ones(seq_len, seq_len))

print(mask)
```

出力は次のようになります。

```text
tensor([[1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]])
```

Attention scoreにmaskを適用するには、見てはいけない位置を `-inf` にします。

```python
scores = scores.masked_fill(mask == 0, float("-inf"))
```

こうすると、softmax後にその位置の重みは0になります。

```python
import torch

scores = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
])

mask = torch.tril(torch.ones(4, 4))

masked_scores = scores.masked_fill(mask == 0, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)

print("masked_scores:")
print(masked_scores)

print("weights:")
print(weights)
```

出力は次のようになります。

```text
masked_scores:
tensor([[1., -inf, -inf, -inf],
        [1., 2., -inf, -inf],
        [1., 2., 3., -inf],
        [1., 2., 3., 4.]])

weights:
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.2689, 0.7311, 0.0000, 0.0000],
        [0.0900, 0.2447, 0.6652, 0.0000],
        [0.0321, 0.0871, 0.2369, 0.6439]])
```

未来の位置の重みが0になっていることがわかります。

Decoder-only Transformerを実装する場合、このmaskは必須です。

```text
未来を見ない
↓
次トークン予測として正しく学習できる
```

---

## 16.9 LayerNormと残差接続の確認

Transformer blockでは、Attentionだけでなく、LayerNormと残差接続も使います。

LayerNormは、値のスケールを整えるために使います。

残差接続は、元の入力を次の層へ流しやすくするために使います。

元論文の形は、次のように書かれます。

```text
LayerNorm(x + Sublayer(x))
```

最近の実装では、次のPre-LNの形もよく使われます。

```text
x + Sublayer(LayerNorm(x))
```

ここでは、Pre-LNの形を確認します。

```python
import torch
import torch.nn as nn

batch_size = 2
seq_len = 4
d_model = 8

x = torch.randn(batch_size, seq_len, d_model)

layer_norm = nn.LayerNorm(d_model)
sublayer = nn.Linear(d_model, d_model)

y = x + sublayer(layer_norm(x))

print("x:", x.shape)
print("y:", y.shape)
```

出力は次のようになります。

```text
x: torch.Size([2, 4, 8])
y: torch.Size([2, 4, 8])
```

入力と出力のshapeは同じです。

```text
[batch_size, seq_len, d_model]
```

これが重要です。

Transformer blockは、入力と出力のshapeが同じなので、何層も重ねられます。

```text
Block 1: [B, T, C] → [B, T, C]
Block 2: [B, T, C] → [B, T, C]
Block 3: [B, T, C] → [B, T, C]
```

ここで、

```text
B = batch_size
T = seq_len
C = d_model
```

です。

LayerNormはshapeを変えません。

残差接続も、足し合わせるテンソルのshapeが同じであればshapeを変えません。

```text
x + sublayer(...)
```

そのため、Transformer blockは同じshapeを保ちながら、内部表現を更新していきます。

```text
同じshapeのまま、トークン表現だけが変化する
```

---

## 16.10 cross entropyと次トークン予測の確認

言語モデルでは、各位置で次のトークンを予測します。

たとえば、トークンID列が次のようだったとします。

```text
[12, 45, 98, 3, 7]
```

入力と正解は1つずらして作ります。

```text
inputs:
[12, 45, 98, 3]

targets:
[45, 98, 3, 7]
```

PyTorchでは、次のように書きます。

```python
import torch

token_ids = torch.tensor([
    [12, 45, 98, 3, 7],
    [8, 21, 21, 56, 4],
])

inputs = token_ids[:, :-1]
targets = token_ids[:, 1:]

print("inputs:")
print(inputs)

print("targets:")
print(targets)
```

出力は次のようになります。

```text
inputs:
tensor([[12, 45, 98,  3],
        [ 8, 21, 21, 56]])

targets:
tensor([[45, 98,  3,  7],
        [21, 21, 56,  4]])
```

モデルは `inputs` を受け取り、各位置で語彙全体へのlogitsを出します。

```text
logits: [batch_size, seq_len, vocab_size]
```

targetsは、各位置の正解トークンIDです。

```text
targets: [batch_size, seq_len]
```

lossを計算するときは、batchとseq_lenをまとめます。

```python
import torch
import torch.nn.functional as F

batch_size = 2
seq_len = 4
vocab_size = 100

logits = torch.randn(batch_size, seq_len, vocab_size)

targets = torch.tensor([
    [45, 98, 3, 7],
    [21, 21, 56, 4],
])

B, T, V = logits.shape

loss = F.cross_entropy(
    logits.reshape(B * T, V),
    targets.reshape(B * T)
)

print(loss)
```

shapeの変換は次の通りです。

```text
logits:
[batch_size, seq_len, vocab_size]
↓
[batch_size * seq_len, vocab_size]

targets:
[batch_size, seq_len]
↓
[batch_size * seq_len]
```

これにより、各位置の次トークン予測をまとめて分類問題として扱います。

重要なのは、`F.cross_entropy` にはsoftmax後の確率ではなく、softmax前のlogitsを渡すことです。

```text
正しい:
F.cross_entropy(logits, targets)

避ける:
F.cross_entropy(torch.softmax(logits), targets)
```

---

## 16.11 学習ループの確認

Transformerを学習させるときの基本ループは、PyTorchでは次のようになります。

```python
logits = model(inputs)

loss = F.cross_entropy(
    logits.reshape(B * T, V),
    targets.reshape(B * T)
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

この流れを言葉で書くと、次のようになります。

```text
1. モデルに入力を入れる
2. logitsを出す
3. logitsとtargetsからlossを計算する
4. 前回の勾配をリセットする
5. backwardで勾配を計算する
6. optimizerでパラメータを更新する
```

小さな言語モデル風の例で確認します。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

token_ids = torch.tensor([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
])

inputs = token_ids[:, :-1]
targets = token_ids[:, 1:]

vocab_size = 10
d_model = 8

embedding = nn.Embedding(vocab_size, d_model)
output_layer = nn.Linear(d_model, vocab_size)

params = list(embedding.parameters()) + list(output_layer.parameters())
optimizer = torch.optim.AdamW(params, lr=0.01)

for step in range(100):
    hidden = embedding(inputs)
    logits = output_layer(hidden)

    B, T, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(step, float(loss))
```

これは本物のTransformerではありません。

しかし、学習ループの構造は同じです。

本物のTransformerでは、embeddingとoutput_layerの間にTransformer blocksが入ります。

```text
inputs
↓
embedding
↓
Transformer blocks
↓
output_layer
↓
logits
↓
cross entropy loss
```

学習の基本は変わりません。

```text
forward
loss
backward
update
```

この流れが理解できていれば、Transformer実装に進む準備ができています。

---

## 16.12 最小限のSelf-Attention確認コード

ここで、Self-Attentionの最小確認コードをまとめておきます。

このコードは、Multi-Headではありません。

単一headのSelf-Attentionです。

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        d_k = q.size(-1)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        out = weights @ v

        return out, weights


batch_size = 2
seq_len = 4
d_model = 8

x = torch.randn(batch_size, seq_len, d_model)

attention = SelfAttention(d_model)

out, weights = attention(x)

print("x:", x.shape)
print("weights:", weights.shape)
print("out:", out.shape)
```

出力は次のようになります。

```text
x: torch.Size([2, 4, 8])
weights: torch.Size([2, 4, 4])
out: torch.Size([2, 4, 8])
```

causal maskを使う場合は、次のようにします。

```python
mask = torch.tril(torch.ones(seq_len, seq_len))

out, weights = attention(x, mask)

print("mask:", mask.shape)
print("weights:", weights.shape)
print("out:", out.shape)
```

このSelfAttentionクラスの中でやっていることは、次の式そのものです。

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

ただし、実際のTransformerでは、これをMulti-Headに拡張します。

また、Attentionの後に出力線形層を入れます。

```text
Single-Head Attention
↓
Multi-Head Attention
↓
output projection
```

その次に、残差接続、LayerNorm、Feed Forward Networkを組み合わせます。

```text
Transformer block
```

このコードは、その手前の最小確認用です。

---

## 16.13 ここまでで理解しておきたいチェックリスト

ここまでの数学編を終えた時点で、次のことが説明できれば十分です。

まず、shapeについてです。

```text
[batch_size, seq_len, d_model]
```

を見て、

```text
batch_sizeは文の数
seq_lenはトークン数
d_modelは各トークンのベクトル次元
```

と説明できること。

次に、embeddingについてです。

```text
token_ids: [batch_size, seq_len]
↓
embedding
↓
x: [batch_size, seq_len, d_model]
```

と説明できること。

次に、Q/K/Vについてです。

```text
xから線形変換でq, k, vを作る
```

と説明できること。

```text
q = w_q(x)
k = w_k(x)
v = w_v(x)
```

次に、Attention scoreについてです。

```text
scores = q @ k.transpose(-2, -1)
```

が、

```text
QueryとKeyの内積をまとめて計算している
```

と説明できること。

次に、softmaxについてです。

```text
weights = torch.softmax(scores, dim=-1)
```

が、

```text
各Queryが各Keyを見る重みを作っている
```

と説明できること。

次に、Valueを混ぜる部分です。

```text
out = weights @ v
```

が、

```text
Attention weightに応じてValueを重み付き和している
```

と説明できること。

次に、causal maskについてです。

```text
Decoder-only Transformerでは未来のトークンを見ないようにmaskする
```

と説明できること。

次に、cross entropyについてです。

```text
logitsとtargetsからlossを計算する
正解トークンの確率が高いほどlossは小さい
```

と説明できること。

次に、学習ループについてです。

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

の意味を説明できること。

最後に、LayerNormと残差接続についてです。

```text
LayerNormは値のスケールを整える
残差接続は x + Sublayer(x)
Transformer blockでは両方が重要
```

と説明できること。

このチェックリストを完全に暗記する必要はありません。

しかし、見たときに意味が追える状態になっていれば、Transformer実装に進めます。

---

## 16.14 次に学ぶべきこと

この数学編の次に学ぶべきことは、**ニューラルネットワークの基本**です。

すでに機械学習の基本を学び、この数学編でTransformerに必要な最小限の数学を見ました。

次に必要なのは、ニューラルネットワークを部品として理解することです。

特に、次の内容が重要です。

```text
ニューラルネットワークとは何か
線形層
活性化関数
多層パーセプトロン
forwardとbackward
パラメータ
optimizer
ミニバッチ学習
過学習
正則化
Dropout
LayerNorm
残差接続
```

このうち、線形層、LayerNorm、残差接続は、この数学編でもすでに少し出てきました。

次の教科書では、それらをニューラルネットワーク全体の文脈で整理するとよいです。

その後、自然言語処理の基本に進みます。

```text
tokenization
vocab
embedding
language model
next token prediction
seq2seq
RNN
LSTM
Attention
```

そして最後に、Transformer本体に進みます。

```text
Self-Attention
Multi-Head Attention
Positional Encoding
Feed Forward Network
Residual Connection
LayerNorm
Encoder
Decoder
Decoder-only Transformer
```

つまり、学習の流れは次のようになります。

```text
機械学習の基本
↓
数学の最低限
↓
ニューラルネットワークの基本
↓
自然言語処理の基本
↓
Attention以前の流れ
↓
Transformer
↓
小さなTransformer実装
```

この数学編は、その中の2番目にあたります。

ここまで理解できれば、Transformerの数式で使われる数学的な道具はかなり揃っています。

---

## 16.15 まとめ

この章では、Transformer実装に進む前の確認をしました。

まず、Transformerで最も重要なshapeを確認しました。

```text
[batch_size, seq_len, d_model]
```

これは、

```text
文の数
トークン数
各トークンのベクトル次元
```

を表します。

次に、embeddingを確認しました。

```text
token_ids: [batch_size, seq_len]
↓
embedding
↓
x: [batch_size, seq_len, d_model]
```

次に、線形変換でQ/K/Vを作ることを確認しました。

```text
q = w_q(x)
k = w_k(x)
v = w_v(x)
```

次に、Attentionの中心計算を確認しました。

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
out = weights @ v
```

これは、次の式に対応しています。

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

次に、causal maskを確認しました。

```python
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float("-inf"))
```

これは、Decoder-only Transformerで未来のトークンを見ないようにするために使います。

次に、言語モデルのloss計算を確認しました。

```text
logits:  [batch_size, seq_len, vocab_size]
targets: [batch_size, seq_len]
```

cross entropyを計算するときは、次のようにshapeを変えます。

```python
B, T, V = logits.shape

loss = F.cross_entropy(
    logits.reshape(B * T, V),
    targets.reshape(B * T)
)
```

次に、PyTorchの学習ループを確認しました。

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

この3行は、ニューラルネットワーク学習の基本です。

最後に、次へ進むための順番を確認しました。

```text
機械学習の基本
↓
数学の最低限
↓
ニューラルネットワークの基本
↓
自然言語処理の基本
↓
Attention以前の流れ
↓
Transformer
↓
小さなTransformer実装
```

この数学編で学んだ内容は、Transformerの式を読むための道具です。

特に重要なのは、次の一文です。

```text
Attentionは、QueryとKeyの内積でトークン同士の相性スコアを作り、softmaxで重みに変換し、その重みに応じてValueを混ぜる仕組みである。
```

この説明ができて、対応するPyTorchコードを読めるなら、この数学編の目的は達成できています。

次は、ニューラルネットワークの基本に進むとよいです。