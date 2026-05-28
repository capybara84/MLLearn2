# 第6章 線形変換

## 6.1 線形変換とは何か

前章では、行列を「ベクトルを変換するもの」として見る考え方を学びました。

この章では、その考え方をもう少し詳しく見ていきます。

**線形変換**とは、ざっくり言えば、ベクトルに行列を掛けて、別のベクトルに変換することです。

たとえば、次のようなベクトルがあるとします。

```text
x = [1.0, 2.0]
```

このベクトルに行列を掛けます。

```text
W = [
  [1.0, 0.0],
  [0.0, 2.0]
]
```

すると、ベクトルが別のベクトルに変換されます。

```text
xW = [1.0, 4.0]
```

この例では、2番目の成分が2倍になりました。

つまり、この行列 `W` は、ベクトルを次のように変換する働きを持っています。

```text
[x1, x2] → [x1, 2*x2]
```

これが線形変換の基本的なイメージです。

```text
入力ベクトル
↓
行列を掛ける
↓
出力ベクトル
```

ニューラルネットワークでは、入力を少しずつ別の表現へ変換していきます。

Transformerでも同じです。

入力されたトークンのベクトルは、そのまま使われるのではなく、何度も別のベクトルに変換されます。

たとえば、Self-Attentionでは、入力ベクトルからQuery、Key、Valueを作ります。

```text
入力ベクトル
↓
Queryベクトル
Keyベクトル
Valueベクトル
```

この変換に使われるのが線形変換です。

---

## 6.2 行列はベクトルを別のベクトルに変換する

行列を掛けると、ベクトルの値が変わります。

それだけでなく、次元数も変えることができます。

たとえば、3次元ベクトルを2次元ベクトルに変換することができます。

```text
x = [1.0, 2.0, 3.0]
```

この `x` は3次元ベクトルです。

ここに、次のような行列を掛けます。

```text
W = [
  [1.0, 0.0],
  [0.0, 1.0],
  [1.0, 1.0]
]
```

`W` のshapeは次の通りです。

```text
[3, 2]
```

`x` のshapeは次の通りです。

```text
[3]
```

`x @ W` を計算すると、shapeは次のようになります。

```text
[3] @ [3, 2] → [2]
```

つまり、3次元ベクトルが2次元ベクトルに変換されます。

計算してみます。

```text
x @ W
= [1.0, 2.0, 3.0] @ [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
  ]
```

出力の1つ目の成分は、次のようになります。

```text
1.0*1.0 + 2.0*0.0 + 3.0*1.0 = 4.0
```

出力の2つ目の成分は、次のようになります。

```text
1.0*0.0 + 2.0*1.0 + 3.0*1.0 = 5.0
```

したがって、

```text
x @ W = [4.0, 5.0]
```

です。

このように、行列を使うと、ベクトルの次元数を変えることができます。

```text
3次元ベクトル → 2次元ベクトル
```

逆に、2次元ベクトルを4次元ベクトルに変換することもできます。

```text
[2] @ [2, 4] → [4]
```

Transformerでは、このような次元変換が多く出てきます。

たとえば、入力の次元数を `d_model`、Queryの次元数を `d_k` とすると、次のようになります。

```text
x:   [d_model]
W_Q: [d_model, d_k]

q:   [d_k]
```

つまり、

```text
入力ベクトル x
↓
W_Q で変換
↓
Queryベクトル q
```

です。

---

## 6.3 `Wx + b` の意味

ニューラルネットワークの説明では、よく次の式が出てきます。

```text
y = Wx + b
```

または、実装や説明の流儀によっては次のように書かれることもあります。

```text
y = xW + b
```

この教科書では、Transformer実装でshapeを追いやすいように、基本的には次の形で考えます。

```text
y = xW + b
```

ここで、それぞれの意味は次の通りです。

```text
x: 入力ベクトル
W: 重み行列
b: バイアスベクトル
y: 出力ベクトル
```

たとえば、次のような入力ベクトルがあるとします。

```text
x = [1.0, 2.0, 3.0]
```

これは3次元ベクトルです。

これを2次元ベクトルに変換したいとします。

その場合、重み行列 `W` のshapeは次のようになります。

```text
W: [3, 2]
```

すると、

```text
xW: [3] @ [3, 2] → [2]
```

となり、2次元ベクトルが得られます。

ここにバイアス `b` を足します。

```text
b = [0.5, -0.5]
```

もし、

```text
xW = [4.0, 5.0]
```

だった場合、

```text
y = xW + b
```

は次のようになります。

```text
y = [4.0, 5.0] + [0.5, -0.5]
y = [4.5, 4.5]
```

つまり、`xW` でベクトルを変換し、最後に `b` で値を少しずらしています。

```text
x
↓
Wで変換
↓
xW
↓
bを足す
↓
y
```

ニューラルネットワークでは、`W` と `b` は学習されるパラメータです。

最初はランダムな値から始まり、学習によって少しずつ調整されます。

```text
最初のW, b: ランダム
↓
損失を計算
↓
勾配で更新
↓
よりよいW, bになる
```

Transformerの中でも、この `xW + b` に相当する処理が何度も出てきます。

PyTorchでは、これは `nn.Linear` で実装します。

```python
import torch
import torch.nn as nn

x = torch.tensor([1.0, 2.0, 3.0])

linear = nn.Linear(3, 2)

y = linear(x)

print(y)
print(y.shape)
```

出力のshapeは次のようになります。

```text
torch.Size([2])
```

これは、

```text
3次元ベクトル
↓
nn.Linear(3, 2)
↓
2次元ベクトル
```

という変換です。

---

## 6.4 ニューラルネットワークにおける重み行列

ニューラルネットワークでは、重み行列が非常に重要です。

重み行列は、入力をどのように変換するかを決めるパラメータです。

たとえば、次のような線形層を考えます。

```python
import torch.nn as nn

linear = nn.Linear(3, 2)
```

これは、3次元の入力を2次元の出力に変換する層です。

```text
入力: [3]
出力: [2]
```

この層の内部には、重みとバイアスがあります。

```python
print(linear.weight.shape)
print(linear.bias.shape)
```

出力は次のようになります。

```text
torch.Size([2, 3])
torch.Size([2])
```

ここで注意が必要です。

PyTorchの `nn.Linear(in_features, out_features)` では、重みのshapeは次のようになります。

```text
[out_features, in_features]
```

つまり、

```text
nn.Linear(3, 2)
```

なら、重みは次のshapeです。

```text
[2, 3]
```

数学的な説明では `xW + b` と書き、`W` を `[3, 2]` と考えることがあります。

一方、PyTorch内部では重みが `[2, 3]` で持たれていて、実際にはその転置を使う形で計算されます。

ただし、最初はこの違いに深入りしなくて大丈夫です。

実装上、重要なのは次のことです。

```text
nn.Linear(in_features, out_features)
=
最後の次元を in_features から out_features に変換する
```

たとえば、次の入力があるとします。

```python
import torch
import torch.nn as nn

x = torch.randn(5, 3)

linear = nn.Linear(3, 2)

y = linear(x)

print("x:", x.shape)
print("y:", y.shape)
```

出力は次のようになります。

```text
x: torch.Size([5, 3])
y: torch.Size([5, 2])
```

これは、

```text
[5, 3]
↓ nn.Linear(3, 2)
[5, 2]
```

です。

先頭の `5` はそのまま残り、最後の次元だけが `3` から `2` に変わっています。

Transformerでも同じです。

入力が次のshapeだとします。

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

つまり、`nn.Linear` は各トークンのベクトルに同じ変換を適用していると考えられます。

```text
各トークンの d_model 次元ベクトル
↓
同じLinear層
↓
各トークンの d_k 次元ベクトル
```

---

## 6.5 入力ベクトルを別の表現に変換する

線形変換の役割は、入力ベクトルを別の表現に変換することです。

ここでいう「別の表現」とは、同じ情報を別の見方で表すものです。

たとえば、あるトークンのembedding vectorがあるとします。

```text
x = [0.10, 0.20, -0.30, 0.40]
```

これは、そのトークンの基本的なベクトル表現です。

しかし、Self-Attentionでは、この `x` をそのまま使うのではなく、Query、Key、Valueという3つの役割に分けます。

```text
x
↓
Queryとしての表現 q
Keyとしての表現 k
Valueとしての表現 v
```

同じ入力ベクトル `x` から、異なる線形変換によって3種類のベクトルを作ります。

```text
q = xW_Q + b_Q
k = xW_K + b_K
v = xW_V + b_V
```

ここで、

```text
W_Q: Query用の重み行列
W_K: Key用の重み行列
W_V: Value用の重み行列
```

です。

同じ `x` を入力しても、使う重み行列が違うので、出力されるベクトルも違います。

```text
同じx
↓
W_Qで変換 → q
W_Kで変換 → k
W_Vで変換 → v
```

これは、人間の言葉で言えば、同じ対象を別の観点から見るようなものです。

たとえば、ある人を表す情報があるとします。

```text
名前、年齢、職業、住所、趣味
```

この人を、

```text
採用候補者として見る
顧客として見る
友人として見る
```

では、注目する情報が違います。

同じ元情報でも、目的によって「取り出したい表現」が変わります。

Self-Attentionでも同じように、同じトークンベクトルから、目的の違う3種類の表現を作っています。

```text
Query: 自分が探しているもの
Key: 自分が持っているラベル
Value: 実際に渡す中身
```

このように、線形変換は「同じベクトルを、別の役割を持つベクトルに変換する」ために使われます。

---

## 6.6 embeddingからQ/K/Vを作る

ここでは、Transformerで非常に重要な処理である、embeddingからQ/K/Vを作る流れを見ます。

まず、入力はトークンIDです。

```text
token_ids: [batch_size, seq_len]
```

これをembedding層に通すと、各トークンがベクトルになります。

```text
x: [batch_size, seq_len, d_model]
```

ここで、`x` は各トークンのembedding vectorです。

次に、この `x` からQuery、Key、Valueを作ります。

```text
q = W_Q(x)
k = W_K(x)
v = W_V(x)
```

PyTorchでは、次のように書けます。

```python
import torch
import torch.nn as nn

batch_size = 2
seq_len = 5
vocab_size = 100
d_model = 8

token_ids = torch.tensor([
    [12, 45, 98, 3, 7],
    [8, 21, 21, 56, 4],
])

embedding = nn.Embedding(vocab_size, d_model)

x = embedding(token_ids)

w_q = nn.Linear(d_model, d_model)
w_k = nn.Linear(d_model, d_model)
w_v = nn.Linear(d_model, d_model)

q = w_q(x)
k = w_k(x)
v = w_v(x)

print("token_ids:", token_ids.shape)
print("x:", x.shape)
print("q:", q.shape)
print("k:", k.shape)
print("v:", v.shape)
```

出力は次のようになります。

```text
token_ids: torch.Size([2, 5])
x: torch.Size([2, 5, 8])
q: torch.Size([2, 5, 8])
k: torch.Size([2, 5, 8])
v: torch.Size([2, 5, 8])
```

shapeを追うと、次のようになります。

```text
token_ids:
[batch_size, seq_len]

embedding後:
x: [batch_size, seq_len, d_model]

Linear後:
q: [batch_size, seq_len, d_model]
k: [batch_size, seq_len, d_model]
v: [batch_size, seq_len, d_model]
```

この例では、簡単のためにQuery、Key、Valueの次元数をすべて `d_model` と同じにしています。

つまり、

```text
d_k = d_model
d_v = d_model
```

です。

より一般的には、次のように考えます。

```text
x: [batch_size, seq_len, d_model]

w_q: nn.Linear(d_model, d_k)
w_k: nn.Linear(d_model, d_k)
w_v: nn.Linear(d_model, d_v)

q: [batch_size, seq_len, d_k]
k: [batch_size, seq_len, d_k]
v: [batch_size, seq_len, d_v]
```

重要なのは、`nn.Linear` が最後の次元だけを変換することです。

```text
[batch_size, seq_len, d_model]
↓
[batch_size, seq_len, d_k]
```

この処理によって、各トークンのembedding vectorは、Attentionで使うためのQuery、Key、Valueに変換されます。

---

## 6.7 Transformerの中の線形層

Transformerでは、線形層が多くの場所で使われます。

代表的なのは次の場所です。

```text
Q/K/Vを作る線形層
Multi-Head Attention後の出力線形層
Feed Forward Networkの中の線形層
最終的な語彙への出力層
```

まず、Q/K/Vを作る線形層です。

```text
q = W_Q(x)
k = W_K(x)
v = W_V(x)
```

これは、入力ベクトルをAttention用の3種類の表現に変換します。

次に、Multi-Head Attention後の出力線形層です。

Multi-Head Attentionでは、複数のheadの出力を結合します。

その後、もう一度線形層を通して、`d_model` 次元の表現に戻します。

```text
複数headの出力
↓
結合
↓
出力線形層
↓
d_model次元のベクトル
```

次に、Feed Forward Networkの中の線形層です。

Transformerの各層には、Attentionだけでなく、Feed Forward Networkも含まれています。

典型的には、次のような構造です。

```text
Linear
↓
活性化関数
↓
Linear
```

たとえば、元論文のTransformerでは、`d_model = 512`、中間次元 `d_ff = 2048` のように、いったん大きな次元に広げてから戻します。

```text
[batch_size, seq_len, d_model]
↓ Linear(d_model, d_ff)
[batch_size, seq_len, d_ff]
↓ 活性化関数
[batch_size, seq_len, d_ff]
↓ Linear(d_ff, d_model)
[batch_size, seq_len, d_model]
```

このように、Feed Forward Networkでも線形変換が使われます。

最後に、言語モデルの出力層です。

言語モデルでは、最終的に「次のトークンが何か」を予測します。

そのためには、各位置のベクトルを語彙サイズ分のスコアに変換する必要があります。

```text
hidden state: [batch_size, seq_len, d_model]
↓ Linear(d_model, vocab_size)
logits: [batch_size, seq_len, vocab_size]
```

ここで、`vocab_size` は語彙数です。

たとえば、語彙が50,000個あれば、各位置について50,000個のスコアを出します。

```text
次のトークン候補1のスコア
次のトークン候補2のスコア
...
次のトークン候補50000のスコア
```

このように、Transformerでは、線形層がいたるところに出てきます。

線形層は、Transformerの内部で表現を変換する基本部品です。

---

## 6.8 PyTorchで線形変換を確認する

ここでは、PyTorchで線形変換を確認します。

まず、1つのベクトルを変換してみます。

```python
import torch
import torch.nn as nn

x = torch.tensor([1.0, 2.0, 3.0])

linear = nn.Linear(3, 2)

y = linear(x)

print("x:", x)
print("x.shape:", x.shape)
print("y:", y)
print("y.shape:", y.shape)
```

出力のshapeは次のようになります。

```text
x.shape: torch.Size([3])
y.shape: torch.Size([2])
```

これは、次の変換です。

```text
[3]
↓ Linear(3, 2)
[2]
```

次に、複数のベクトルをまとめて変換します。

```python
import torch
import torch.nn as nn

x = torch.randn(4, 3)

linear = nn.Linear(3, 2)

y = linear(x)

print("x.shape:", x.shape)
print("y.shape:", y.shape)
```

出力は次のようになります。

```text
x.shape: torch.Size([4, 3])
y.shape: torch.Size([4, 2])
```

これは、4本の3次元ベクトルを、まとめて2次元ベクトルに変換していると考えられます。

```text
[4, 3]
↓ Linear(3, 2)
[4, 2]
```

次に、Transformerでよく出る3次元テンソルを変換します。

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

print("x.shape:", x.shape)
print("y.shape:", y.shape)
```

出力は次のようになります。

```text
x.shape: torch.Size([2, 5, 8])
y.shape: torch.Size([2, 5, 4])
```

これは、次の変換です。

```text
[batch_size, seq_len, d_model]
↓ Linear(d_model, d_k)
[batch_size, seq_len, d_k]
```

具体的には、

```text
[2, 5, 8]
↓ Linear(8, 4)
[2, 5, 4]
```

です。

先頭の2つの次元はそのまま残ります。

```text
batch_size = 2
seq_len = 5
```

最後の次元だけが変わります。

```text
d_model = 8
↓
d_k = 4
```

この「最後の次元だけを変換する」という性質は、Transformer実装で非常によく使います。

---

## 6.9 PyTorchでQ/K/Vを作る

ここでは、実際にPyTorchでQ/K/Vを作ります。

まず、入力テンソル `x` を用意します。

```python
import torch
import torch.nn as nn

batch_size = 2
seq_len = 5
d_model = 8
d_k = 4
d_v = 6

x = torch.randn(batch_size, seq_len, d_model)

print("x:", x.shape)
```

出力は次のようになります。

```text
x: torch.Size([2, 5, 8])
```

これは、

```text
2個の文
各文は5トークン
各トークンは8次元ベクトル
```

という意味です。

次に、Q/K/V用の線形層を作ります。

```python
w_q = nn.Linear(d_model, d_k)
w_k = nn.Linear(d_model, d_k)
w_v = nn.Linear(d_model, d_v)
```

それぞれの意味は次の通りです。

```text
w_q: d_model次元 → d_k次元
w_k: d_model次元 → d_k次元
w_v: d_model次元 → d_v次元
```

次に、入力 `x` をそれぞれの線形層に通します。

```python
q = w_q(x)
k = w_k(x)
v = w_v(x)

print("q:", q.shape)
print("k:", k.shape)
print("v:", v.shape)
```

出力は次のようになります。

```text
q: torch.Size([2, 5, 4])
k: torch.Size([2, 5, 4])
v: torch.Size([2, 5, 6])
```

shapeを追うと、次のようになります。

```text
x: [2, 5, 8]

q: [2, 5, 4]
k: [2, 5, 4]
v: [2, 5, 6]
```

これは、

```text
d_model = 8
d_k = 4
d_v = 6
```

なので、

```text
xの最後の次元 8 が
q, k では 4 に変換され
v では 6 に変換された
```

ということです。

ここで重要なのは、`q` と `k` の最後の次元が同じであることです。

```text
q: [batch_size, seq_len, d_k]
k: [batch_size, seq_len, d_k]
```

なぜなら、AttentionではQueryとKeyの内積を計算するからです。

```text
q @ k.transpose(-2, -1)
```

内積を計算するためには、QueryとKeyの次元が一致している必要があります。

一方、`v` の次元 `d_v` は、理屈の上では `d_k` と違っていても構いません。

Attention scoreを作るときに使うのはQueryとKeyです。

Valueは、最後に重みに応じて混ぜられる中身です。

次に、Attention scoreまで計算してみます。

```python
import math

scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d_k)

print("scores:", scores.shape)
```

出力は次のようになります。

```text
scores: torch.Size([2, 5, 5])
```

shapeを追うと、次のようになります。

```text
q:                   [2, 5, 4]
k.transpose(-2, -1): [2, 4, 5]

scores:              [2, 5, 5]
```

最後に、softmaxをかけ、Valueを混ぜます。

```python
weights = torch.softmax(scores, dim=-1)
out = weights @ v

print("weights:", weights.shape)
print("out:", out.shape)
```

出力は次のようになります。

```text
weights: torch.Size([2, 5, 5])
out: torch.Size([2, 5, 6])
```

shapeを追うと、次の通りです。

```text
weights: [2, 5, 5]
v:       [2, 5, 6]

out:     [2, 5, 6]
```

このサンプル全体をまとめると、次のようになります。

```python
import torch
import torch.nn as nn
import math

batch_size = 2
seq_len = 5
d_model = 8
d_k = 4
d_v = 6

x = torch.randn(batch_size, seq_len, d_model)

w_q = nn.Linear(d_model, d_k)
w_k = nn.Linear(d_model, d_k)
w_v = nn.Linear(d_model, d_v)

q = w_q(x)
k = w_k(x)
v = w_v(x)

scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d_k)

weights = torch.softmax(scores, dim=-1)

out = weights @ v

print("x:", x.shape)
print("q:", q.shape)
print("k:", k.shape)
print("v:", v.shape)
print("scores:", scores.shape)
print("weights:", weights.shape)
print("out:", out.shape)
```

出力は次のようになります。

```text
x: torch.Size([2, 5, 8])
q: torch.Size([2, 5, 4])
k: torch.Size([2, 5, 4])
v: torch.Size([2, 5, 6])
scores: torch.Size([2, 5, 5])
weights: torch.Size([2, 5, 5])
out: torch.Size([2, 5, 6])
```

このコードは、Self-Attentionの中心部分にかなり近いです。

ただし、この章の目的はAttentionの完全な理解ではありません。

ここで重要なのは、線形層によって、入力 `x` から `q`, `k`, `v` を作っていることです。

```text
x
↓ Linear
q, k, v
```

---

## 6.10 線形変換だけでは足りない理由

ここまで、線形変換について学びました。

線形変換は、ニューラルネットワークの非常に重要な部品です。

しかし、線形変換だけを何層も重ねても、表現力はあまり増えません。

なぜなら、線形変換を何度重ねても、全体としてはまた1つの線形変換にまとめられるからです。

たとえば、次のように2回線形変換をしたとします。

```text
h = xW_1
y = hW_2
```

これをまとめると、

```text
y = xW_1W_2
```

です。

`W_1W_2` もまた1つの行列なので、結局、

```text
y = xW
```

という1回の線形変換と同じ形になります。

つまり、線形変換だけを何層も重ねても、本質的には1回の線形変換と変わらないのです。

そこで、ニューラルネットワークでは、線形変換の間に **非線形な処理** を入れます。

代表的なのが活性化関数です。

```text
Linear
↓
活性化関数
↓
Linear
```

活性化関数には、ReLU、GELU、tanhなどがあります。

TransformerのFeed Forward Networkでは、線形層の間に活性化関数が入ります。

```text
Linear(d_model, d_ff)
↓
活性化関数
↓
Linear(d_ff, d_model)
```

この非線形性があることで、ニューラルネットワークは複雑な関数を表現できるようになります。

ただし、Self-Attentionの中のQ/K/Vを作る部分では、まず線形変換が基本になります。

```text
x → q
x → k
x → v
```

その後、内積、softmax、重み付き和によって、トークン同士の情報を混ぜます。

つまり、Transformer全体では、線形変換だけでなく、さまざまな操作が組み合わさっています。

```text
線形変換
内積
softmax
重み付き和
活性化関数
正規化
残差接続
```

この章では、その中でも特に基本となる線形変換を扱いました。

---

## 6.11 まとめ

この章では、線形変換について学びました。

線形変換とは、ベクトルに行列を掛けて、別のベクトルに変換することです。

```text
入力ベクトル
↓
行列を掛ける
↓
出力ベクトル
```

ニューラルネットワークでは、線形層としてよく使われます。

基本的な式は次の通りです。

```text
y = xW + b
```

ここで、

```text
x: 入力ベクトル
W: 重み行列
b: バイアスベクトル
y: 出力ベクトル
```

です。

`W` と `b` は学習されるパラメータです。

最初はランダムに近い値から始まり、損失が小さくなるように更新されます。

PyTorchでは、線形変換は `nn.Linear` で実装します。

```python
linear = nn.Linear(in_features, out_features)
```

これは、最後の次元を `in_features` から `out_features` に変換します。

たとえば、

```text
[batch_size, seq_len, d_model]
↓ Linear(d_model, d_k)
[batch_size, seq_len, d_k]
```

です。

Transformerでは、線形変換がさまざまな場所で使われます。

```text
Q/K/Vを作る線形層
Multi-Head Attention後の出力線形層
Feed Forward Networkの中の線形層
語彙サイズへの出力層
```

特にSelf-Attentionでは、入力ベクトルからQuery、Key、Valueを作るために線形変換を使います。

```text
q = W_Q(x)
k = W_K(x)
v = W_V(x)
```

または、実装上は次のように書きます。

```python
q = w_q(x)
k = w_k(x)
v = w_v(x)
```

この章で特に重要なのは、次の理解です。

```text
行列はベクトルを別のベクトルに変換する
線形層は最後の次元を変換する
Transformerでは線形層でQ/K/Vを作る
QとKは内積を取るので同じ次元にする必要がある
線形変換はTransformerの基本部品である
```

次章では、softmaxについて学びます。

softmaxは、Attention scoreを「どのトークンをどれくらい見るか」という重みに変換するために使われます。

Transformerの中心式では、次の部分です。

```text
softmax(QK^T / sqrt(d_k))
```

ここを理解すると、Attentionの式がさらに読みやすくなります。