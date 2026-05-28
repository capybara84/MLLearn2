import torch
import torch.nn.functional as F


torch.manual_seed(0)

token_ids = torch.tensor([1, 2, 3, 4])

inputs = token_ids[:-1]
targets = token_ids[1:]

logits = torch.tensor(
    [
        [0.1, 0.2, 2.0, -0.5, 0.0],
        [0.0, 0.2, 0.5, 1.5, -0.4],
        [0.3, -0.2, 0.0, 0.4, 1.8],
    ]
)

loss = F.cross_entropy(logits, targets)

probs = torch.softmax(logits, dim=-1)
target_probs = probs[torch.arange(targets.numel()), targets]
manual_loss = -torch.log(target_probs).mean()

print("token_ids:", token_ids)
print("inputs:", inputs)
print("targets:", targets)
print()
print("logits shape:", logits.shape)
print("probs:")
print(probs)
print("target_probs:", target_probs)
print("manual loss:", manual_loss.item())
print("torch loss: ", loss.item())
