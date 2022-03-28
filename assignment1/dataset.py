import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

class Weather(Dataset):
    def __init__(self, X, Y):
        self.x, self.y = X, Y.values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]