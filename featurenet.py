import torch
ht=4
wd=6
batch=8
coords1 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
coords2 = torch.stack(coords1[::-1], dim=0).float()
x=coords2[None].repeat(batch, 1, 1, 1)
coords = x[:, [0]]
print(coords)
#print(coords1)
#print(coords2)
#print(x)
print(coords.shape)
print(x.shape)