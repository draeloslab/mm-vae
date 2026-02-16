import torch
from torch.utils.data import Dataset

class PairedRegionDataset(Dataset):
    def __init__(self, paired_df, data_cols_hc, data_cols_pfc, meta_cols_hc, meta_cols_pfc, label_col, join_col='strain'): 
        self.paired_df = paired_df
        self.join_col = join_col
        self.label_col = label_col

        self.data_cols_hc = data_cols_hc
        self.data_cols_pfc = data_cols_pfc
        self.meta_cols_hc = meta_cols_hc
        self.meta_cols_pfc = meta_cols_pfc

    def __len__(self):
        return len(self.paired_df)

    def __getitem__(self, idx):
        sample = self.paired_df.iloc[idx]

        # for HC 
        data_hc = torch.tensor(sample[self.data_cols_hc].values.astype('float32'))
        label_hc = sample[self.label_col]
        meta_hc = sample[self.meta_cols_hc].to_dict()

        # for PFC 
        data_pfc = torch.tensor(sample[self.data_cols_pfc].values.astype('float32'))
        label_pfc = sample[self.label_col]
        meta_pfc = sample[self.meta_cols_pfc].to_dict()

        return (data_hc, label_hc, meta_hc), (data_pfc, label_pfc, meta_pfc)