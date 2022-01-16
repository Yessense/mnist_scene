import torch

a = torch.tensor(range(4)).unsqueeze(-1)
print(a)
print(a.expand(-1, 5))