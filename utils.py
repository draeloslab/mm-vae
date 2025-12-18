import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate

def load_data(df):
    x = df.iloc[:, 2:].values.astype('float32')
    fly_id = df['FlyID'].tolist()
    genotype = df['Genotype_SpecSheet'].tolist()
    df['AD_label'] = np.where(
        (df['Genotype_SpecSheet'].str.contains('attp2', case=False, na=False)) | (df['Genotype_SpecSheet'].str.contains('attp4', case=False, na=False)), 
        0, 
        1
    )
    AD_label = df['AD_label'].values
    return x, fly_id, genotype, AD_label

def custom_collate(batch):
    data_list, fly_id_list, genotype_list = zip(*batch)
    collated_data_tensor = default_collate(data_list)
    return collated_data_tensor, list(fly_id_list), list(genotype_list)
