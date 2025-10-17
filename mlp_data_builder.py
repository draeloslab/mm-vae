from utils import*
from torch.utils.data import Dataset
import torch

class MLPDataBuilder(Dataset):
    def __init__(self, data, feature_cols, label_cols):
        # self.features = torch.tensor(data[feature_cols].values, dtype=torch.float32)
        # self.labels = torch.tensor(data[label_cols].values, dtype=torch.float32)
        self.features = data[feature_cols].values.astype('float32')
        self.labels = data[label_cols].values.astype('float32')
        #self.id_labels = data[id_cols].values

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return len(self.features)