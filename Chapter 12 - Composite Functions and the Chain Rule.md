# 第12章 合成関数と連鎖律

**この章のゴール**

ニューラルネットワークを関数の合成として見て、連鎖律とbackpropagationが勾配を前の層へ伝える仕組みだと理解すること。

## 12.1 ニューラルネットワークは関数の合成である

前章では、勾配降下法について学びました。

勾配降下法では、lossを小さくするために、パラメータを勾配の逆方向へ更新します。

```text
parameter = parameter - learning_rate * gradient
```

ここで必要になるのが、各パラメータに対する勾配です。

```text
∂loss / ∂parameter
```

つまり、

```text
このパラメータを少し変えたら、lossがどう変わるか
```

を知る必要があります。

しかし、ニューラルネットワークでは、パラメータからlossまでの距離はかなり遠いです。

たとえば、Transformerでは、入力からlossまでに多くの処理があります。

```text
token_ids
↓
embedding
↓
positional encoding
↓
Self-Attention
↓
Feed Forward Network
↓
LayerNorm
↓
何層も繰り返す
↓
output layer
↓
logits
↓
cross entropy
↓
loss
```

このように、lossは1回の計算で直接出てくるわけではありません。

たくさんの関数を順番に通った結果として出てきます。

このように、関数を順番につなげたものを **合成関数** と呼びます。

たとえば、次のような流れを考えます。

```text
x
↓ f
h
↓ g
y
```

これは、まず `x` に関数 `f` を適用して `h` を作り、その `h` に関数 `g` を適用して `y` を作る、という意味です。

式で書くと、次のようになります。

```text
h = f(x)
y = g(h)
```

まとめると、

```text
y = g(f(x))
```

です。

これが合成関数です。

ニューラルネットワークも、基本的にはこの合成関数です。

```text
入力
↓
関数1
↓
関数2
↓
関数3
↓
出力
↓
loss
```

Transformerも同じです。

各層は、入力テンソルを別のテンソルへ変換する関数だと見なせます。

```text
embedding: token_idをベクトルに変換する関数
Self-Attention: ベクトル列を文脈を反映したベクトル列に変換する関数
Feed Forward Network: 各位置のベクトルをさらに変換する関数
output layer: hidden stateを語彙スコアに変換する関数
cross entropy: logitsとtargetsからlossを計算する関数
```

このように、ニューラルネットワークは多くの関数を合成したものです。

そのため、勾配を計算するには、合成関数の微分を理解する必要があります。

そこで出てくるのが **連鎖律** です。

---

## 12.2 合成関数とは何か

合成関数とは、ある関数の出力を、次の関数の入力にするものです。

たとえば、次の2つの関数を考えます。

```text
f(x) = 2x
g(h) = h + 3
```

まず、`x` に `f` を適用します。

```text
h = f(x)
h = 2x
```

次に、その `h` に `g` を適用します。

```text
y = g(h)
y = h + 3
```

これをまとめると、

```text
y = g(f(x))
```

です。

実際に `x = 4` を入れてみます。

まず、`f` を計算します。

```text
h = f(4)
h = 2 * 4
h = 8
```

次に、`g` を計算します。

```text
y = g(8)
y = 8 + 3
y = 11
```

つまり、

```text
x = 4
↓ f
h = 8
↓ g
y = 11
```

です。

ニューラルネットワークでも同じことが起きています。

たとえば、非常に小さなネットワークを考えます。

```text
h = Linear(x)
a = ReLU(h)
logits = Linear(a)
loss = CrossEntropy(logits, target)
```

これは、関数を順番につなげたものです。

```text
x
↓ Linear
h
↓ ReLU
a
↓ Linear
logits
↓ CrossEntropy
loss
```

Transformerでも同じです。

```text
x
↓ Self-Attention
x'
↓ Feed Forward Network
x''
↓ Output Layer
logits
↓ Cross Entropy
loss
```

このように、合成関数では、途中の値が次の計算に使われます。

そのため、最初の方のパラメータがlossにどう影響するかを考えるには、途中の計算をたどる必要があります。

たとえば、`W_Q` というQuery用の重みを考えます。

```text
x
↓ W_Q
Q
↓ QK^T
scores
↓ softmax
weights
↓ weights @ V
attention output
↓ さらに次の層
logits
↓ cross entropy
loss
```

`W_Q` は直接lossを作っているわけではありません。

しかし、`W_Q` が変わると `Q` が変わります。

`Q` が変わると `scores` が変わります。

`scores` が変わると `weights` が変わります。

`weights` が変わるとAttentionの出力が変わります。

そして最終的にlossが変わります。

```text
W_Qが変わる
↓
Qが変わる
↓
scoresが変わる
↓
weightsが変わる
↓
出力が変わる
↓
lossが変わる
```

このつながりを通して勾配を計算するために、連鎖律が必要になります。

---

## 12.3 連鎖律とは何か

**連鎖律**とは、合成関数の微分を計算するためのルールです。

まず、次のような合成関数を考えます。

```text
h = f(x)
y = g(h)
```

つまり、

```text
y = g(f(x))
```

です。

ここで知りたいのは、

```text
xを少し変えたら、yがどう変わるか
```

です。

つまり、

```text
dy/dx
```

を求めたい。

しかし、`x` は直接 `y` に変わるわけではありません。

途中に `h` があります。

```text
x → h → y
```

そこで、変化を2段階に分けて考えます。

```text
xを少し変えると、hがどう変わるか
hを少し変えると、yがどう変わるか
```

それぞれは次の微分で表せます。

```text
dh/dx
dy/dh
```

このとき、`x` が `y` に与える影響は、この2つを掛け合わせたものになります。

```text
dy/dx = dy/dh * dh/dx
```

これが連鎖律です。

日本語で読むと、次のようになります。

```text
xがyに与える影響
=
xがhに与える影響
×
hがyに与える影響
```

もう少し直感的に言えば、

```text
途中の変化を掛け合わせる
```

ということです。

たとえば、次のような関係を考えます。

```text
h = 2x
y = 3h
```

`x` が1増えると、`h` は2増えます。

```text
dh/dx = 2
```

`h` が1増えると、`y` は3増えます。

```text
dy/dh = 3
```

では、`x` が1増えると、`y` はどれくらい増えるでしょうか。

```text
dy/dx = dy/dh * dh/dx
dy/dx = 3 * 2
dy/dx = 6
```

実際に、式をまとめると、

```text
y = 3h
h = 2x
```

なので、

```text
y = 3 * 2x
y = 6x
```

です。

つまり、`x` が1増えると、`y` は6増えます。

このように、連鎖律は、関数が何段階にもつながっているときに、変化の影響をたどるためのルールです。

---

## 12.4 出力の誤差を前の層へ伝える

ニューラルネットワークの学習では、まず最後にlossが計算されます。

```text
入力
↓
ニューラルネットワーク
↓
予測
↓
loss
```

しかし、更新したいパラメータはネットワークの中のあちこちにあります。

たとえば、Transformerでは次のようなパラメータがあります。

```text
embeddingの重み
Q/K/Vの重み
Attention出力の重み
Feed Forward Networkの重み
LayerNormのパラメータ
出力層の重み
```

lossは最後に出ます。

しかし、最初の方にあるembeddingやQ/K/Vの重みも更新しなければなりません。

そのためには、lossから前の層へ向かって影響をたどる必要があります。

```text
loss
↑
logits
↑
output layer
↑
Transformer blocks
↑
embedding
```

これがバックプロパゲーションの考え方です。

日本語では「誤差逆伝播」と呼ばれます。

名前の通り、最後に出たlossの情報を、ネットワークの後ろから前へ伝えていきます。

ここで使われている数学的なルールが連鎖律です。

たとえば、次のような流れを考えます。

```text
x
↓
h
↓
logits
↓
loss
```

`x` に関するlossの勾配を知りたい場合、次のようにたどります。

```text
∂loss / ∂x
=
∂loss / ∂logits
×
∂logits / ∂h
×
∂h / ∂x
```

これは、連鎖律そのものです。

ニューラルネットワークでは、このような計算を大量に行います。

しかし、人間がすべて手で計算する必要はありません。

PyTorchが自動微分で計算してくれます。

ただし、PyTorchが内部でやっていることは、基本的には次の流れです。

```text
forwardで計算のつながりを記録する
loss.backward()で後ろから前へ勾配を伝える
各パラメータの.gradに勾配を入れる
```

つまり、

```text
loss.backward()
```

は、連鎖律を使って、lossから各パラメータへの勾配を計算していると考えればよいです。

---

## 12.5 backpropagationの数学的な正体

**backpropagation**、つまりバックプロパゲーションの数学的な正体は、連鎖律です。

ニューラルネットワークは、たくさんの関数をつなげた合成関数です。

```text
y = f_n(f_{n-1}(...f_2(f_1(x))))
```

ここで、最終的にlossを計算します。

```text
loss = L(y, target)
```

学習で知りたいのは、各パラメータがlossにどう影響するかです。

たとえば、ある中間層の重み `W` について、

```text
∂loss / ∂W
```

を知りたい。

しかし、`W` は直接lossにつながっているわけではありません。

```text
W
↓
中間出力
↓
次の層
↓
さらに次の層
↓
logits
↓
loss
```

このように、何段階もの計算を通ってlossに影響します。

そこで、連鎖律を使います。

非常に単純な例で考えます。

```text
h = xW
y = hU
loss = L(y)
```

ここで、`W` に対するlossの勾配を知りたいとします。

`W` はまず `h` に影響します。

```text
W → h
```

`h` は `y` に影響します。

```text
h → y
```

`y` は `loss` に影響します。

```text
y → loss
```

したがって、`W` がlossに与える影響は、次のつながりをたどります。

```text
W → h → y → loss
```

連鎖律のイメージで書くと、

```text
∂loss / ∂W
=
∂loss / ∂y
×
∂y / ∂h
×
∂h / ∂W
```

です。

厳密なテンソルの形は少し複雑ですが、直感としてはこれで十分です。

つまり、バックプロパゲーションは、

```text
最後のlossから出発して、
各計算を逆向きにたどりながら、
連鎖律で勾配を掛け合わせていく
```

処理です。

forwardでは、入力から出力へ計算します。

```text
入力
↓
中間層
↓
出力
↓
loss
```

backwardでは、lossから入力側へ勾配を伝えます。

```text
loss
↓
出力層の勾配
↓
中間層の勾配
↓
前の層の勾配
```

この「逆向きにたどる」ことから、backpropagationと呼ばれます。

---

## 12.6 実装では自動微分がやってくれる

連鎖律は重要ですが、実際にTransformerを実装するとき、人間が手で全ての微分を書くことはほとんどありません。

PyTorchが自動微分で計算してくれます。

たとえば、次のようなコードを考えます。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

h = x * 3
y = h ** 2

y.backward()

print(x.grad)
```

この計算の流れは次の通りです。

```text
x
↓ h = 3x
h
↓ y = h^2
y
```

式としてまとめると、

```text
y = (3x)^2
```

です。

このとき、`x = 2` なら、

```text
h = 3 * 2 = 6
y = 6^2 = 36
```

微分を考えると、

```text
dy/dh = 2h
dh/dx = 3
```

連鎖律より、

```text
dy/dx = dy/dh * dh/dx
```

です。

`h = 6` なので、

```text
dy/dh = 2 * 6 = 12
```

したがって、

```text
dy/dx = 12 * 3 = 36
```

PyTorchの出力も、`36` になります。

```text
tensor(36.)
```

このように、PyTorchは計算グラフをたどって、自動的に連鎖律を適用します。

ニューラルネットワークでも同じです。

```python
logits = model(inputs)
loss = loss_fn(logits, targets)

loss.backward()
```

この `loss.backward()` によって、PyTorchはモデルの中のすべての学習可能なパラメータについて勾配を計算します。

```text
embedding.weight.grad
linear.weight.grad
linear.bias.grad
...
```

つまり、私たちが手でやる必要があるのは、基本的には次のことです。

```text
forward計算を書く
lossを計算する
loss.backward()を呼ぶ
optimizer.step()で更新する
```

連鎖律そのものを毎回手で実装する必要はありません。

ただし、何が起きているかを理解するためには、連鎖律の直感は必要です。

なぜなら、勾配がどこからどこへ伝わるのかを理解していないと、モデルがうまく学習しないときに原因を考えにくいからです。

---

## 12.7 それでも連鎖律を知るべき理由

PyTorchが自動微分してくれるなら、連鎖律を知らなくてもよいのではないか、と思うかもしれません。

実際、簡単なモデルを動かすだけなら、連鎖律の式を手で書けなくても動きます。

しかし、Transformerを理解して実装できるようになりたいなら、連鎖律の直感は重要です。

理由は大きく3つあります。

1つ目は、勾配がどこを通って伝わるかを理解するためです。

Transformerでは、lossから非常に多くの経路を通って勾配が流れます。

```text
loss
↓
output layer
↓
Transformer blocks
↓
Attention
↓
Q/K/V
↓
embedding
```

たとえば、Q/K/Vの重みは、Attentionの結果を通じてlossに影響します。

```text
W_Q
↓
Q
↓
QK^T
↓
softmax
↓
weights @ V
↓
loss
```

この流れを理解するには、連鎖律の考え方が必要です。

2つ目は、勾配が消えたり爆発したりする問題を理解するためです。

合成関数では、微分が何段階にも掛け合わされます。

```text
∂loss / ∂x
=
多くの微分の積
```

途中で小さい値が何度も掛けられると、勾配が非常に小さくなることがあります。

```text
0.1 × 0.1 × 0.1 × 0.1 = 0.0001
```

これが勾配消失の直感です。

逆に、大きい値が何度も掛けられると、勾配が非常に大きくなることがあります。

```text
10 × 10 × 10 × 10 = 10000
```

これが勾配爆発の直感です。

Transformerで残差接続やLayer Normalizationが重要なのは、深いネットワークでも勾配を流しやすくし、学習を安定させるためです。

3つ目は、モデルの構造を設計するときに役立つからです。

新しい層や処理を実装するとき、その処理が微分可能かどうかを考える必要があります。

ニューラルネットワークは、基本的に勾配によって学習します。

そのため、モデルの中に勾配が通らない処理を入れると、その部分はうまく学習できなくなることがあります。

たとえば、argmaxは通常、そのままでは勾配を流しにくい操作です。

一方、softmaxは微分可能なので、Attentionの中で学習に使えます。

この違いは重要です。

```text
softmax:
連続的で微分可能
↓
勾配が流れる

argmax:
離散的に1つを選ぶ
↓
通常は勾配が流れにくい
```

Attentionでargmaxではなくsoftmaxを使う理由の1つは、重みを滑らかに作れて、学習しやすいからです。

このように、連鎖律を理解していると、ニューラルネットワークの設計や実装の意味が見えやすくなります。

---

## 12.8 PyTorchで連鎖律を確認する

ここでは、PyTorchで連鎖律を確認します。

まず、単純な合成関数を考えます。

```text
h = 3x
y = h^2
```

つまり、

```text
y = (3x)^2
```

です。

`x = 2` のとき、

```text
h = 6
y = 36
```

微分を手で考えると、

```text
dy/dh = 2h
dh/dx = 3
```

`h = 6` なので、

```text
dy/dh = 12
```

したがって、

```text
dy/dx = dy/dh * dh/dx
dy/dx = 12 * 3
dy/dx = 36
```

PyTorchで確認します。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

h = 3 * x
y = h ** 2

y.backward()

print("x:", x)
print("h:", h)
print("y:", y)
print("x.grad:", x.grad)
```

出力は次のようになります。

```text
x: tensor(2., requires_grad=True)
h: tensor(6., grad_fn=<MulBackward0>)
y: tensor(36., grad_fn=<PowBackward0>)
x.grad: tensor(36.)
```

`x.grad` が `36` になりました。

これは、連鎖律で計算した値と一致しています。

次に、もう少しニューラルネットワークらしい例を見ます。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

h = w * x + b
loss = h ** 2

loss.backward()

print("h:", h)
print("loss:", loss)
print("x.grad:", x.grad)
print("w.grad:", w.grad)
print("b.grad:", b.grad)
```

この計算は次の通りです。

```text
h = wx + b
loss = h^2
```

`x = 2`, `w = 3`, `b = 1` なので、

```text
h = 3 * 2 + 1
h = 7
```

```text
loss = 7^2
loss = 49
```

それぞれの勾配を考えます。

まず、

```text
d loss / d h = 2h = 14
```

です。

次に、

```text
d h / d w = x = 2
```

なので、

```text
d loss / d w = d loss / d h * d h / d w
d loss / d w = 14 * 2
d loss / d w = 28
```

また、

```text
d h / d b = 1
```

なので、

```text
d loss / d b = 14 * 1
d loss / d b = 14
```

さらに、

```text
d h / d x = w = 3
```

なので、

```text
d loss / d x = 14 * 3
d loss / d x = 42
```

PyTorchの出力も、次のようになります。

```text
x.grad: tensor(42.)
w.grad: tensor(28.)
b.grad: tensor(14.)
```

このように、PyTorchは内部で連鎖律を使って、各変数への勾配を計算しています。

---

## 12.9 PyTorchの計算グラフ

PyTorchの自動微分は、計算グラフを使って行われます。

計算グラフとは、どの値がどの計算から作られたかを表すつながりです。

たとえば、次のコードを考えます。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)

h = x * w
y = h + 1
loss = y ** 2
```

この計算は、次のような流れです。

```text
x     w
 \   /
  \ /
  h = x * w
  ↓
  y = h + 1
  ↓
  loss = y^2
```

PyTorchは、`requires_grad=True` のテンソルが関わる計算を記録します。

この記録があるから、あとで `loss.backward()` を呼ぶと、勾配を計算できます。

```python
loss.backward()
```

すると、PyTorchは計算グラフを逆向きにたどります。

```text
loss
↓
y
↓
h
↓
x, w
```

そして、連鎖律を使って、`x` と `w` に対する勾配を計算します。

```python
print(x.grad)
print(w.grad)
```

ここで大事なのは、PyTorchがforward計算のときに、必要な情報を記録しているということです。

そのため、通常の学習ループでは、次の順番になります。

```text
1. forward計算をする
2. lossを計算する
3. backwardで勾配を計算する
4. optimizerで更新する
```

コードでは次のようになります。

```python
logits = model(inputs)
loss = loss_fn(logits, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

もし、`requires_grad=True` が付いていないテンソルだけで計算した場合、PyTorchは勾配を計算しません。

また、`with torch.no_grad():` の中で行った計算も、通常は計算グラフに記録されません。

たとえば、パラメータ更新では次のように書きます。

```python
with torch.no_grad():
    parameter -= learning_rate * parameter.grad
```

これは、更新操作そのものを計算グラフに含めないためです。

学習時のforward計算では勾配が必要です。

一方、パラメータ更新や推論だけの処理では勾配が不要なことがあります。

この違いを意識しておくと、PyTorchの挙動が理解しやすくなります。

---

## 12.10 小さなネットワークでbackpropagationを見る

ここでは、小さなニューラルネットワークでbackpropagationを確認します。

1つの線形層を持つ分類モデルを考えます。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.tensor([1])

model = nn.Linear(3, 2)

logits = model(x)
loss = F.cross_entropy(logits, target)

print("logits:", logits)
print("loss:", loss)
```

この時点では、まだ勾配は計算されていません。

勾配を計算するには、`backward()` を呼びます。

```python
loss.backward()
```

すると、モデルのパラメータに勾配が入ります。

```python
print("weight grad:")
print(model.weight.grad)

print("bias grad:")
print(model.bias.grad)
```

全体のコードは次の通りです。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.tensor([1])

model = nn.Linear(3, 2)

logits = model(x)
loss = F.cross_entropy(logits, target)

loss.backward()

print("logits:", logits)
print("loss:", loss)
print("weight:")
print(model.weight)
print("weight grad:")
print(model.weight.grad)
print("bias:")
print(model.bias)
print("bias grad:")
print(model.bias.grad)
```

ここで、`model.weight.grad` は、重みに対するlossの勾配です。

```text
∂loss / ∂weight
```

`model.bias.grad` は、バイアスに対するlossの勾配です。

```text
∂loss / ∂bias
```

これらの勾配は、次の流れを逆向きにたどって計算されています。

```text
x
↓ Linear
logits
↓ CrossEntropy
loss
```

backwardでは逆向きです。

```text
loss
↓ CrossEntropyの勾配
↓ logitsへの勾配
↓ Linearへの勾配
↓ weight, biasへの勾配
```

このように、backpropagationは、lossから各パラメータへ勾配を伝える処理です。

---

## 12.11 Transformerにおけるbackpropagationの流れ

Transformerでも、backpropagationの考え方は同じです。

ただし、計算の経路がかなり複雑になります。

たとえば、Self-Attentionだけを見ても、次のような流れがあります。

```text
x
↓
q = W_Q(x)
k = W_K(x)
v = W_V(x)
↓
scores = q @ k^T / sqrt(d_k)
↓
weights = softmax(scores)
↓
out = weights @ v
```

この `out` はさらに次の層へ進み、最終的にlossになります。

```text
out
↓
次の層
↓
logits
↓
loss
```

backpropagationでは、この流れを逆にたどります。

```text
loss
↓
logits
↓
次の層
↓
out
↓
weights @ v
↓
weights, v
↓
softmax
↓
scores
↓
q, k
↓
W_Q, W_K, W_V
```

つまり、`W_Q`, `W_K`, `W_V` にも勾配が届きます。

これは非常に重要です。

Self-Attentionの中のQ/K/Vは、手で決めているわけではありません。

学習によって、どのようなQuery、Key、Valueを作ればよいかが調整されます。

```text
W_Qが更新される
↓
Queryの作り方が変わる

W_Kが更新される
↓
Keyの作り方が変わる

W_Vが更新される
↓
Valueの作り方が変わる
```

これにより、モデルはタスクに役立つAttentionのパターンを学習していきます。

たとえば、言語モデルなら、次トークン予測に役立つように、各トークンがどのトークンを参照すべきかを学びます。

Transformerの各部品は微分可能な操作でできています。

```text
Linear
行列積
softmax
加算
LayerNorm
活性化関数
cross entropy
```

これらは基本的に自動微分で扱えます。

だから、Transformer全体を1つの大きな合成関数として扱い、lossからすべてのパラメータへ勾配を流すことができます。

```text
Transformer全体
=
巨大な合成関数

学習
=
lossから各パラメータへの勾配を計算して更新すること
```

---

## 12.12 detachとno_gradの直感

PyTorchを使っていると、`detach()` や `torch.no_grad()` が出てくることがあります。

この章の内容と関係するので、直感だけ説明しておきます。

まず、`torch.no_grad()` です。

これは、その中の計算を勾配計算の対象にしないためのものです。

たとえば、推論だけをしたい場合には勾配は不要です。

```python
with torch.no_grad():
    logits = model(inputs)
```

このようにすると、PyTorchは計算グラフを作りません。

そのため、メモリを節約できます。

また、パラメータ更新を手で行うときにも使います。

```python
with torch.no_grad():
    w -= learning_rate * w.grad
```

この更新操作自体は、学習対象の計算グラフに含めたくないので、`no_grad()` の中で行います。

次に、`detach()` です。

`detach()` は、あるテンソルを計算グラフから切り離します。

たとえば、

```python
y = x.detach()
```

とすると、`y` は `x` と同じ値を持ちますが、そこから先の計算では `x` への勾配が流れません。

直感的には、

```text
ここで勾配の流れを止める
```

という操作です。

通常のTransformer実装では、最初から頻繁に使う必要はありません。

むしろ、意味を理解せずに `detach()` を使うと、勾配が必要なところで止まってしまい、学習できなくなることがあります。

この段階では、次の理解で十分です。

```text
torch.no_grad():
その範囲の計算を勾配計算の対象にしない

detach():
そのテンソルから過去への勾配の流れを切る
```

どちらも、計算グラフと勾配の流れを制御するための道具です。

---

## 12.13 まとめ

この章では、合成関数と連鎖律について学びました。

ニューラルネットワークは、多くの関数を順番につなげた合成関数です。

```text
入力
↓
関数1
↓
関数2
↓
関数3
↓
出力
↓
loss
```

Transformerも同じです。

```text
token_ids
↓
embedding
↓
Self-Attention
↓
Feed Forward Network
↓
output layer
↓
logits
↓
cross entropy
↓
loss
```

合成関数の微分を計算するためのルールが、連鎖律です。

単純な形では、次のように書けます。

```text
h = f(x)
y = g(h)

dy/dx = dy/dh * dh/dx
```

これは、

```text
xがhに与える影響
×
hがyに与える影響
=
xがyに与える影響
```

という意味です。

ニューラルネットワークでは、lossから各パラメータへの影響を、連鎖律によって逆向きにたどります。

この処理がbackpropagationです。

```text
forward:
入力からlossへ計算する

backward:
lossから各パラメータへ勾配を伝える
```

PyTorchでは、自動微分によってbackpropagationを行います。

```python
loss.backward()
```

これにより、各パラメータの `.grad` に勾配が入ります。

```python
parameter.grad
```

その後、optimizerがパラメータを更新します。

```python
optimizer.step()
```

Transformerでも、同じ仕組みで学習します。

```text
loss
↓
output layer
↓
Transformer blocks
↓
Attention
↓
Q/K/Vの重み
↓
embedding
```

この章で特に重要なのは、次の理解です。

```text
ニューラルネットワークは合成関数である
連鎖律は合成関数の微分を計算するルールである
backpropagationは連鎖律を使って勾配を逆向きに伝える処理である
PyTorchの自動微分は、このbackpropagationを自動で行ってくれる
```

次章では、正規化について学びます。

Transformerでは、Layer Normalizationが非常に重要です。

深いネットワークでは、値のスケールが大きくなりすぎたり小さくなりすぎたりすると、学習が不安定になります。

正規化は、値のスケールを整え、学習を安定させるための重要な道具です。
