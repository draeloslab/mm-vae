import torch
from torch.utils.data._utils.collate import default_collate

def load_data(df):
    x = df.iloc[:, 2:].values.astype('float32')
    fly_id = df['FlyID'].tolist()
    genotype = df['Genotype_SpecSheet'].tolist()
    return x, fly_id, genotype

def custom_collate(batch):
    data_list, fly_id_list, genotype_list = zip(*batch)
    collated_data_tensor = default_collate(data_list)
    return collated_data_tensor, list(fly_id_list), list(genotype_list)
