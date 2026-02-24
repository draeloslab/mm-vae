from torch.utils.data import  Dataset

class MLPDataBuilder(Dataset):
    def __init__(self, data, feature_cols, label_cols):
        self.features = data[feature_cols].values.astype('float32')
        self.labels = data[label_cols].values.astype('float32')

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)