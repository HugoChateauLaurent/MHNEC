import torch

x = torch.tensor([0,5,7,89,25.98,5,5], )
print(torch.nn.functional.softmax(x))
print(torch.nn.functional.softmax(x/x.sum()))
print(x/x.sum())