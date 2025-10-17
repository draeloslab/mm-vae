from utils import*
from torch.utils.data import Dataset

class DataBuilder(Dataset):
    def __init__(self, data):
        self.x, self.data_label, self.layer1, self.mouse_ids = load_data_pseudo(data)
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.data_label[index], self.layer1[index], self.mouse_ids[index]
    def __len__(self):
        return self.len