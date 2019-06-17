import torch

class_num = 10
batch_size = 4
label = torch.tensor([1, 2, 3, 4]).reshape(1, -1)
one_hot = torch.zeros(class_num, batch_size).scatter_(0, label, 1)

print(label)
print(one_hot)
