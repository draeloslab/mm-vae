from utils import*
from torch.utils.data import Dataset
import torch

class MLPDataBuilder(Dataset):
    def __init__(self, data, feature_cols):
        self.features = data[feature_cols].values.astype('float32')
        self.mouse_ids = data['mouse_ids']

    def __getitem__(self, index):
        return self.features[index], self.mouse_ids[index]
    
    def __len__(self):
        return len(self.features)