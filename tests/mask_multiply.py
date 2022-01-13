
import torch
batch_size = 6
n_labels = 4
latent_dim = 12
a = torch.randint(2, (batch_size,n_labels))
b = torch.randn((batch_size, n_labels, latent_dim))

c = a.unsqueeze(-1).expand(b.size())
d = b * c
h = b * a
e = d.sum(axis = 1)
print("done")

f = list([torch.tensor(i) for i in range(10)])
g = torch.stack(f)
print("done")