import torch


def stable_softmax(x):
    shifted = x - x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(shifted)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


scores = torch.tensor([2.0, 1.0, 0.0])

manual = stable_softmax(scores)
pytorch = torch.softmax(scores, dim=-1)

print("scores:", scores)
print("manual softmax:", manual)
print("torch softmax: ", pytorch)
print("sum:", pytorch.sum())

attention_scores = torch.tensor(
    [
        [2.0, 1.0, 0.0],
        [0.5, 1.5, 0.0],
        [1.0, 1.0, 2.0],
    ]
)

weights = torch.softmax(attention_scores, dim=-1)

print()
print("attention_scores:")
print(attention_scores)
print("weights:")
print(weights)
print("row sums:", weights.sum(dim=-1))
