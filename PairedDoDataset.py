import torch
from torch.utils.data import Dataset

class PairedDODataset(Dataset):
    def __init__(self, paired_df, genetic_dims, diet_col, id_col, label_col): 
        self.paired_df = paired_df
        self.genetic_dims = genetic_dims
        self.diet_col = diet_col 
        self.id_col = id_col
        self.label_col = label_col

    def __len__(self):
        return len(self.paired_df)

    def __getitem__(self, idx):
        sample = self.paired_df.iloc[idx]

        # for the genoprobs data 
        genetic_df = torch.tensor(sample[1:self.genetic_dims].values.astype('float32'))
        genetic_label = sample[self.label_col]
        genetic_diet = sample[self.diet_col]
        genetic_id = sample[self.id_col]

        # for the physiological data 
        physio_df = torch.tensor(sample[self.genetic_dims:-1].values.astype('float32'))
        physio_label = sample[self.label_col]
        physio_diet = sample[self.diet_col]
        physio_id = sample[self.id_col]

        return (genetic_df, genetic_label, genetic_diet, genetic_id), (physio_df, physio_label, physio_diet, physio_id)