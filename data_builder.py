from utils import*
from torch.utils.data import Dataset

class DataBuilder(Dataset):
    def __init__(self, data):
        self.x, self.fly_id, self.genotype, self.AD_label = load_data(data)
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.fly_id[index], self.genotype[index], self.AD_label[index]
    def __len__(self):
        return self.len