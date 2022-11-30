import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

class Resources(Dataset):
    def __init__(self, X, Y):
        self.keys = Y.index.date
        self.x, self.y = X, Y

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        return self.x[key].values, self.y.loc[str(key)].values.squeeze()