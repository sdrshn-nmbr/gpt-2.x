import torch

x = torch.randint(low=0, high=10, size=(4, 4))  # generates a 4x4 tensor of random integers between 0 and 100
print(f"x: \n{x}\n")

print(f"x.size(): {x.size()}\n")

y = x.view(16)
print(f"y: \n{y}\n")

print(f"y.size(): {y.size()}\n")

z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(f"z: \n{z}\n")

print(f"z.size(): {z.size()}\n")