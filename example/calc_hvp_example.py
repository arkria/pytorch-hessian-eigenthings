import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hessian_eigenthings.hvp_operator import HVPOperator


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.head1 = nn.Linear(128, 41)

    def forward(self, x):
        return self.head1(x)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def generate_data():
    x = np.random.random((128, 128)).astype(np.float32)
    y = np.random.randint(41, size=128).astype(np.int64)
    return x, y


if __name__ == '__main__':
    ds = Dataset(*generate_data())
    dl = DataLoader(dataset=ds, batch_size=128)
    model = Model()
    criterion = nn.CrossEntropyLoss()
    hvp_operator = HVPOperator(model, dl, criterion, use_gpu=False)
    hvp = hvp_operator.apply(torch.cat([g.contiguous().view(-1) for g in model.parameters()]))
    print(hvp)