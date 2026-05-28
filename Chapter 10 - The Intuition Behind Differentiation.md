# 第10章 微分の直感

## 10.1 微分とは何か

この章では、**微分**について学びます。

微分は、ニューラルネットワークの学習を理解するために必要です。

なぜなら、ニューラルネットワークでは、損失を小さくするために、パラメータをどちら向きにどれくらい動かせばよいかを知る必要があるからです。

たとえば、モデルにある重み `w` があるとします。

この `w` を少し変えると、損失 `loss` も変わります。

```text
w を少し増やす
↓
loss が増えるのか、減るのか

w を少し減らす
↓
loss が増えるのか、減るのか
```

この関係を知るための道具が微分です。

微分は、ざっくり言うと、

```text
ある値を少し変えたとき、結果がどれくらい変わるかを見るもの
```

です。

たとえば、次のような関数を考えます。

```text
y = x^2
```

`x` を変えると、`y` も変わります。

```text
x = 1 → y = 1
x = 2 → y = 4
x = 3 → y = 9
```

このとき、`x` を少し増やしたら `y` がどれくらい増えるかを知りたい。

それが微分の考え方です。

ニューラルネットワークでは、次のように考えます。

```text
重みを少し変えたら、lossはどう変わるか
```

もし、ある重みを増やすとlossが増えるなら、その重みは減らした方がよさそうです。

```text
wを増やす → lossが増える
↓
wは減らした方がよい
```

逆に、ある重みを増やすとlossが減るなら、その重みは増やした方がよさそうです。

```text
wを増やす → lossが減る
↓
wは増やした方がよい
```

このように、微分は「lossを小さくするために、パラメータをどう動かすか」を知るために使われます。

---

## 10.2 変化量を見るという考え方

微分を理解するには、まず「変化量」を考えるとよいです。

たとえば、次の関数を考えます。

```text
y = 2x
```

`x` を変えると、`y` も変わります。

```text
x = 1 → y = 2
x = 2 → y = 4
x = 3 → y = 6
```

`x` が1増えると、`y` は2増えています。

```text
x: 1 → 2
y: 2 → 4
yの変化量: 2
```

```text
x: 2 → 3
y: 4 → 6
yの変化量: 2
```

この関数では、`x` を1増やすと、いつでも `y` は2増えます。

つまり、

```text
xの変化量に対して、yは2倍変化する
```

と言えます。

この「どれくらい変化するか」が微分の基本です。

次に、少し違う関数を考えます。

```text
y = x^2
```

この場合、値は次のようになります。

```text
x = 1 → y = 1
x = 2 → y = 4
x = 3 → y = 9
x = 4 → y = 16
```

`x` が1増えたときの `y` の変化量を見ます。

```text
x: 1 → 2
y: 1 → 4
yの変化量: 3
```

```text
x: 2 → 3
y: 4 → 9
yの変化量: 5
```

```text
x: 3 → 4
y: 9 → 16
yの変化量: 7
```

今度は、変化量が一定ではありません。

`x` が大きくなるほど、`y` の増え方も大きくなっています。

つまり、`y = x^2` では、場所によって変化の勢いが違います。

微分は、この「その場所での変化の勢い」を見るための道具です。

ニューラルネットワークでも同じです。

あるパラメータを少し変えたとき、lossがどれくらい変わるかは、今のパラメータの値によって変わります。

```text
今のwでは、少し増やすとlossが大きく減る
今のwでは、少し増やしてもlossはほとんど変わらない
今のwでは、少し増やすとlossが増えてしまう
```

このような変化を知るために微分を使います。

---

## 10.3 傾きとは何か

微分は、よく「傾き」として説明されます。

たとえば、次の関数を考えます。

```text
y = 2x
```

これは直線です。

`x` が1増えると、`y` は2増えます。

このとき、直線の傾きは2です。

```text
傾き = yの変化量 / xの変化量
```

具体的には、

```text
x: 1 → 2
y: 2 → 4

xの変化量 = 1
yの変化量 = 2

傾き = 2 / 1 = 2
```

です。

このように、傾きは「入力を少し変えたとき、出力がどれくらい変わるか」を表します。

次に、`y = x^2` を考えます。

これは直線ではなく曲線です。

曲線の場合、場所によって傾きが変わります。

たとえば、`x = 1` の近くでは、傾きは比較的小さいです。

`x = 5` の近くでは、傾きは大きくなります。

```text
x が小さい場所 → ゆるやか
x が大きい場所 → 急になる
```

微分は、この「ある点での傾き」を求めるものです。

たとえば、

```text
y = x^2
```

の微分は、

```text
dy/dx = 2x
```

です。

これは、`x` の場所によって傾きが変わることを表しています。

たとえば、

```text
x = 1 のとき、傾き = 2
x = 2 のとき、傾き = 4
x = 3 のとき、傾き = 6
```

です。

ニューラルネットワークでは、`y` の代わりに `loss` を考えます。

```text
loss = f(w)
```

ここで、`w` は重みです。

微分は、

```text
wを少し変えたとき、lossがどれくらい変わるか
```

を表します。

つまり、

```text
d loss / d w
```

は、重み `w` に対するlossの傾きです。

この傾きがわかると、lossを減らす方向がわかります。

---

## 10.4 損失を小さくする方向を知る

機械学習では、損失を小さくしたいです。

```text
lossを小さくする
=
モデルの予測をよくする
```

そのためには、パラメータをどう動かせばlossが小さくなるかを知る必要があります。

ここで微分が役に立ちます。

たとえば、ある重み `w` に対するlossの傾きがプラスだったとします。

```text
d loss / d w > 0
```

これは、

```text
wを増やすとlossが増えやすい
```

という意味です。

lossを小さくしたいなら、`w` は増やすのではなく、減らすべきです。

```text
傾きがプラス
↓
wを減らす
```

逆に、傾きがマイナスだったとします。

```text
d loss / d w < 0
```

これは、

```text
wを増やすとlossが減りやすい
```

という意味です。

この場合、lossを小さくしたいなら、`w` を増やすべきです。

```text
傾きがマイナス
↓
wを増やす
```

まとめると、lossを小さくするには、傾きと逆向きに動かします。

```text
傾きがプラス → パラメータを減らす
傾きがマイナス → パラメータを増やす
```

これが勾配降下法の基本です。

勾配降下法は次章で詳しく扱いますが、直感は非常に単純です。

```text
lossが増える方向がわかる
↓
その逆に動く
↓
lossが下がる
```

微分は、この「lossが増える方向」を教えてくれます。

だから、ニューラルネットワークの学習には微分が必要なのです。

---

## 10.5 パラメータを少し変えると損失はどう変わるか

ニューラルネットワークには大量のパラメータがあります。

たとえば、Transformerでは、線形層の重み行列、バイアス、embedding、LayerNormのパラメータなど、多くの学習可能な値があります。

```text
embeddingの重み
Q/K/Vの重み
Feed Forward Networkの重み
出力層の重み
LayerNormのパラメータ
```

学習では、これらのパラメータを少しずつ更新します。

そのためには、各パラメータについて、次のことを知りたいです。

```text
このパラメータを少し増やすと、lossは増えるのか減るのか
このパラメータを少し減らすと、lossは増えるのか減るのか
```

たとえば、ある重み `w1` について、

```text
d loss / d w1 = 0.8
```

だったとします。

これは、`w1` を増やすとlossが増えやすいということです。

だから、`w1` は減らした方がよいです。

別の重み `w2` について、

```text
d loss / d w2 = -0.3
```

だったとします。

これは、`w2` を増やすとlossが減りやすいということです。

だから、`w2` は増やした方がよいです。

さらに別の重み `w3` について、

```text
d loss / d w3 = 0.0
```

だったとします。

これは、`w3` を少し変えてもlossがほとんど変わらないということです。

このように、各パラメータごとにlossへの影響を調べます。

```text
w1の影響
w2の影響
w3の影響
...
```

ニューラルネットワークでは、パラメータの数が非常に多いので、人間が手で計算することはできません。

そこで、PyTorchなどのフレームワークが自動微分によって計算してくれます。

ただし、PyTorchが自動でやってくれるとしても、何を計算しているのかを理解しておくことは重要です。

PyTorchが計算しているのは、基本的には次の情報です。

```text
各パラメータを少し変えたら、lossがどう変わるか
```

この情報を使って、optimizerがパラメータを更新します。

---

## 10.6 偏微分とは何か

ここまでは、入力が1つだけの関数を考えてきました。

たとえば、

```text
y = x^2
```

です。

しかし、ニューラルネットワークでは、パラメータがたくさんあります。

たとえば、lossが3つの重みによって決まるとします。

```text
loss = f(w1, w2, w3)
```

このとき、知りたいのは次のような情報です。

```text
w1を少し変えるとlossはどう変わるか
w2を少し変えるとlossはどう変わるか
w3を少し変えるとlossはどう変わるか
```

このように、複数の変数があるときに、1つの変数だけに注目して微分することを **偏微分** と呼びます。

たとえば、

```text
∂loss / ∂w1
```

は、`w1` に関する偏微分です。

これは、

```text
w2やw3は固定したまま、w1だけを少し変えたらlossはどう変わるか
```

を表します。

同じように、

```text
∂loss / ∂w2
```

は、

```text
w1やw3は固定したまま、w2だけを少し変えたらlossはどう変わるか
```

を表します。

ニューラルネットワークでは、各パラメータについて偏微分を計算します。

```text
∂loss / ∂w1
∂loss / ∂w2
∂loss / ∂w3
...
```

これらが、各パラメータの更新方向を決める材料になります。

ここで、記号が少し変わっています。

1変数の微分では、よく次のように書きます。

```text
d y / d x
```

複数変数のときは、偏微分を表すために次の記号を使います。

```text
∂
```

つまり、

```text
∂loss / ∂w
```

のように書きます。

読み方は「lossをwで偏微分したもの」です。

この記号を見たら、まずは次のように読めば十分です。

```text
wを少し変えたらlossがどう変わるか
```

---

## 10.7 勾配とは何か

**勾配**とは、すべての偏微分をまとめたものです。

たとえば、lossが3つの重みで決まるとします。

```text
loss = f(w1, w2, w3)
```

それぞれの偏微分が次のようだったとします。

```text
∂loss / ∂w1 = 0.8
∂loss / ∂w2 = -0.3
∂loss / ∂w3 = 0.0
```

これらをまとめたベクトルが勾配です。

```text
gradient = [0.8, -0.3, 0.0]
```

勾配は、lossが最も増えやすい方向を表します。

つまり、勾配の方向にパラメータを動かすと、lossは増えやすくなります。

しかし、学習でやりたいのはlossを減らすことです。

そのため、勾配とは逆方向に動かします。

```text
パラメータを勾配の逆方向に動かす
↓
lossが下がりやすい
```

これが勾配降下法です。

```text
w = w - learning_rate * gradient
```

ここで、`learning_rate` は学習率です。

どれくらいの大きさで動かすかを決める値です。

たとえば、

```text
w = [1.0, 2.0, 3.0]
gradient = [0.8, -0.3, 0.0]
learning_rate = 0.1
```

だったとします。

更新は次のようになります。

```text
w_new = w - 0.1 * gradient
```

計算すると、

```text
w_new = [1.0, 2.0, 3.0] - 0.1 * [0.8, -0.3, 0.0]
```

```text
w_new = [1.0, 2.0, 3.0] - [0.08, -0.03, 0.0]
```

```text
w_new = [0.92, 2.03, 3.0]
```

ここで、`w1` は減りました。

なぜなら、`∂loss / ∂w1` がプラスだったからです。

```text
w1: 1.0 → 0.92
```

`w2` は増えました。

なぜなら、`∂loss / ∂w2` がマイナスだったからです。

```text
w2: 2.0 → 2.03
```

`w3` は変わりませんでした。

なぜなら、勾配が0だったからです。

```text
w3: 3.0 → 3.0
```

このように、勾配は各パラメータをどちら向きに動かすべきかを教えてくれます。

---

## 10.8 PyTorchで微分を確認する

ここでは、PyTorchで微分を確認します。

まず、非常に単純な関数を考えます。

```text
y = x^2
```

`x = 3` のとき、`y` は次のようになります。

```text
y = 3^2 = 9
```

この関数の微分は、

```text
dy/dx = 2x
```

です。

したがって、`x = 3` のときの傾きは、

```text
dy/dx = 2 * 3 = 6
```

になります。

PyTorchで確認してみます。

```python
import torch

x = torch.tensor(3.0, requires_grad=True)

y = x ** 2

y.backward()

print("x:", x)
print("y:", y)
print("x.grad:", x.grad)
```

出力は次のようになります。

```text
x: tensor(3., requires_grad=True)
y: tensor(9., grad_fn=<PowBackward0>)
x.grad: tensor(6.)
```

`x.grad` が `6` になっています。

これは、`x = 3` における `y = x^2` の微分です。

ここで重要なのは、`requires_grad=True` です。

```python
x = torch.tensor(3.0, requires_grad=True)
```

これは、PyTorchに対して、

```text
この値について勾配を計算したい
```

と伝える指定です。

そして、

```python
y.backward()
```

を呼ぶと、PyTorchが自動微分によって勾配を計算します。

計算結果は、`x.grad` に入ります。

```python
print(x.grad)
```

このように、PyTorchでは微分を自動で計算できます。

---

## 10.9 PyTorchでlossの勾配を確認する

次に、機械学習らしい例を見ます。

非常に単純なモデルを考えます。

```text
y_pred = w * x
```

ここで、`w` は学習したいパラメータです。

入力 `x` と正解 `y_true` が次のように与えられたとします。

```text
x = 2.0
y_true = 10.0
```

もし `w = 3.0` なら、予測は次のようになります。

```text
y_pred = 3.0 * 2.0 = 6.0
```

正解は10なので、予測は小さすぎます。

損失を二乗誤差で定義します。

```text
loss = (y_pred - y_true)^2
```

このとき、`w` を増やせば、`y_pred` も増えます。

今は予測が小さすぎるので、`w` を増やした方がlossは下がりそうです。

PyTorchで確認します。

```python
import torch

x = torch.tensor(2.0)
y_true = torch.tensor(10.0)

w = torch.tensor(3.0, requires_grad=True)

y_pred = w * x

loss = (y_pred - y_true) ** 2

loss.backward()

print("y_pred:", y_pred)
print("loss:", loss)
print("w.grad:", w.grad)
```

出力は次のようになります。

```text
y_pred: tensor(6., grad_fn=<MulBackward0>)
loss: tensor(16., grad_fn=<PowBackward0>)
w.grad: tensor(-16.)
```

`w.grad` が `-16` になりました。

これは、

```text
wを増やすとlossが減る
```

ということを意味します。

なぜなら、勾配がマイナスだからです。

勾配降下法では、次のように更新します。

```text
w = w - learning_rate * gradient
```

もし `learning_rate = 0.1` なら、

```text
w_new = 3.0 - 0.1 * (-16)
w_new = 3.0 + 1.6
w_new = 4.6
```

`w` は増えます。

これは直感と合っています。

今の予測は6で、正解は10です。

予測を大きくするために、`w` を増やした方がよいからです。

実際にPyTorchで更新してみます。

```python
learning_rate = 0.1

with torch.no_grad():
    w -= learning_rate * w.grad

print("updated w:", w)
```

出力は次のようになります。

```text
updated w: tensor(4.6000, requires_grad=True)
```

このように、勾配を使うと、lossを小さくする方向にパラメータを更新できます。

---

## 10.10 勾配を使った1ステップの更新

ここでは、PyTorchで1ステップの学習をまとめて書いてみます。

やることは次の通りです。

```text
1. 予測する
2. lossを計算する
3. backwardで勾配を計算する
4. 勾配を使ってパラメータを更新する
5. 勾配をリセットする
```

コードで書くと、次のようになります。

```python
import torch

x = torch.tensor(2.0)
y_true = torch.tensor(10.0)

w = torch.tensor(3.0, requires_grad=True)

learning_rate = 0.1

y_pred = w * x

loss = (y_pred - y_true) ** 2

loss.backward()

with torch.no_grad():
    w -= learning_rate * w.grad

w.grad.zero_()

print("updated w:", w)
```

ここで、重要な部分を順に見ます。

まず、予測です。

```python
y_pred = w * x
```

次に、lossを計算します。

```python
loss = (y_pred - y_true) ** 2
```

次に、勾配を計算します。

```python
loss.backward()
```

この時点で、`w.grad` に勾配が入ります。

次に、勾配を使って更新します。

```python
with torch.no_grad():
    w -= learning_rate * w.grad
```

`with torch.no_grad():` は、この更新操作自体を勾配計算の対象にしないためのものです。

パラメータを更新するときには、通常このように書きます。

最後に、勾配をリセットします。

```python
w.grad.zero_()
```

PyTorchでは、勾配は自動的に上書きされるのではなく、加算されます。

そのため、次のステップに進む前に勾配をゼロにする必要があります。

ニューラルネットワークの学習では、この流れを何度も繰り返します。

```text
予測
↓
loss
↓
backward
↓
update
↓
gradをzero
↓
次のデータへ
```

実際のPyTorchでは、手で `w -= ...` と書く代わりに、optimizerを使うことが多いです。

```python
optimizer.step()
optimizer.zero_grad()
```

ただし、最初は手で更新してみると、微分と勾配降下法の意味がよくわかります。

---

## 10.11 Transformerで微分はどこに関係するのか

Transformerでは、多くの計算が組み合わさっています。

```text
embedding
Q/K/Vの線形変換
QK^T
softmax
weights @ V
Feed Forward Network
LayerNorm
出力層
cross entropy loss
```

これらの計算を通して、最終的にlossが出ます。

```text
入力トークン
↓
Transformer
↓
logits
↓
cross entropy
↓
loss
```

学習では、このlossを小さくしたいです。

そのため、Transformerの中にあるすべての学習可能なパラメータについて、勾配を計算します。

たとえば、次のようなパラメータです。

```text
embedding table
W_Q
W_K
W_V
Attention output projection
Feed Forward Networkの重み
LayerNormのパラメータ
出力層の重み
```

それぞれについて、

```text
このパラメータを少し変えるとlossはどう変わるか
```

を計算します。

これが勾配です。

```text
∂loss / ∂parameter
```

Transformerの中の計算は多段階です。

たとえば、`W_Q` は直接lossを出しているわけではありません。

`W_Q` はまずQueryを作ります。

```text
x
↓ W_Q
Q
```

そのQueryがAttention scoreに使われます。

```text
QK^T
```

それがsoftmaxに入り、Valueを混ぜ、次の層に渡り、最終的にlogitsになり、lossになります。

```text
W_Q
↓
Q
↓
Attention
↓
次の層
↓
logits
↓
loss
```

このように、遠く離れたところにあるパラメータが、最終的なlossに影響します。

この影響を連鎖的に計算する仕組みがバックプロパゲーションです。

バックプロパゲーションについては後の章で扱います。

今は、次のことを理解しておけば十分です。

```text
Transformerの学習では、lossから各パラメータへの勾配を計算する
PyTorchは自動微分でそれを計算してくれる
勾配を使って、lossが小さくなる方向にパラメータを更新する
```

---

## 10.12 PyTorchの自動微分とニューラルネットワーク

最後に、小さなニューラルネットワークで自動微分を確認します。

ここでは、入力ベクトルを線形層に通し、lossを計算します。

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
print("weight grad:")
print(model.weight.grad)
print("bias grad:")
print(model.bias.grad)
```

このコードでは、次の流れになっています。

```text
x
↓ Linear(3, 2)
logits
↓ cross entropy
loss
↓ backward
勾配
```

`model.weight.grad` には、重みに対する勾配が入ります。

`model.bias.grad` には、バイアスに対する勾配が入ります。

つまり、

```text
重みを少し変えたらlossがどう変わるか
バイアスを少し変えたらlossがどう変わるか
```

が計算されています。

次に、optimizerを使った更新も見てみます。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.tensor([1])

model = nn.Linear(3, 2)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

logits = model(x)
loss = F.cross_entropy(logits, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("loss:", loss)
```

このコードでは、次の3つが重要です。

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

意味は次の通りです。

```text
optimizer.zero_grad():
前回の勾配をリセットする

loss.backward():
今のlossに対する勾配を計算する

optimizer.step():
勾配を使ってパラメータを更新する
```

ニューラルネットワークの学習ループでは、この3つが基本になります。

実際には、これをデータごと、またはミニバッチごとに繰り返します。

```text
for batch in data:
    logits = model(inputs)
    loss = cross_entropy(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Transformerの学習でも、基本構造は同じです。

モデルが大きくなっても、やっていることは変わりません。

```text
予測する
lossを計算する
勾配を計算する
パラメータを更新する
```

---

## 10.13 まとめ

この章では、微分の直感について学びました。

微分は、ある値を少し変えたときに、結果がどれくらい変わるかを見るための道具です。

```text
xを少し変える
↓
yがどれくらい変わるか
```

ニューラルネットワークでは、次のように使います。

```text
パラメータを少し変える
↓
lossがどれくらい変わるか
```

この情報がわかると、lossを小さくする方向がわかります。

```text
パラメータを増やすとlossが増える
↓
パラメータを減らす

パラメータを増やすとlossが減る
↓
パラメータを増やす
```

複数のパラメータがある場合、各パラメータについて微分を考えます。

これを偏微分と呼びます。

```text
∂loss / ∂w1
∂loss / ∂w2
∂loss / ∂w3
```

すべての偏微分をまとめたものが勾配です。

```text
gradient = [∂loss/∂w1, ∂loss/∂w2, ∂loss/∂w3, ...]
```

勾配は、lossが最も増えやすい方向を表します。

lossを小さくするには、その逆向きに動きます。

```text
w = w - learning_rate * gradient
```

PyTorchでは、自動微分によって勾配を計算できます。

```python
loss.backward()
```

勾配は、各パラメータの `.grad` に入ります。

```python
parameter.grad
```

実際の学習では、optimizerを使って更新します。

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

この章で特に重要なのは、次の理解です。

```text
微分は「少し変えたときの変化」を見る
勾配は各パラメータに対するlossの変化をまとめたもの
勾配の逆方向に動くとlossが下がりやすい
PyTorchは自動微分で勾配を計算してくれる
```

次章では、勾配降下法について学びます。

この章では「勾配とは何か」を見ました。

次章では、その勾配を使って実際にどのようにパラメータを更新し、モデルを学習させるのかを詳しく見ていきます。