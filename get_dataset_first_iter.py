from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np



#My goal is to create a VAE that inherits from the original VAE that implements slowed learning rates from https://arxiv.org/pdf/1612.00796
#generated images from https://arxiv.org/pdf/1906.03288, and a way to test two tasks A and B in 4 different ways
#A = 0-4, B = 5-9
#A=even, B=odd
#A = 0-4, B = 0-5
#A = 0-1, B = 2-3, C = 4-5...
# I also want to create a way to traverse the latent space by taking the means of two numbers and going in a straight line, 
#then also by implementing stepwise density based trajectory
#using accuracy to judge

# def get_train_test_loaders(batch_size, split, **kwargs):
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=True, download=True,
#                     transform=transforms.ToTensor()),
#         batch_size=batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#         batch_size=batch_size, shuffle=False, **kwargs)

#     return tr_loader_list, te_loader_list




def _get_targets(ds): #gpt function to help get the labels
    # MNIST uses `targets`; older torchvision used `train_labels` / `test_labels`
    for name in ("targets", "train_labels", "test_labels"):
        if hasattr(ds, name):
            t = getattr(ds, name)
            return t if isinstance(t, torch.Tensor) else torch.tensor(t)
    raise AttributeError("Could not find labels/targets on the dataset.")

def _indices_for_labels(ds, labels): #gpt function to get indices of targets. Prining out the labels it adds up
    targets = _get_targets(ds)
    labels = torch.tensor(list(labels))
    if hasattr(torch, "isin"):
        mask = torch.isin(targets, labels)
    else:
        # fallback for very old PyTorch
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for l in labels:
            mask |= (targets == int(l))
    return mask.nonzero(as_tuple=False).squeeze().tolist()



def get_train_test_loaders(batch_size, split, root="../data", download=True, **kwargs):
    gen = torch.Generator()
    gen.manual_seed(0)
    #need the dataset first not the dataloader to create subset then dataloader
    full_train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
    full_test_dataset =datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())
    print("size full train: ", len(full_train_dataset))
    print("size full test: ", len(full_test_dataset))
    train_loader_list, test_loader_list = [], []

    for task in split:
        tr_idx = _indices_for_labels(full_train_dataset, task)
        te_idx = _indices_for_labels(full_test_dataset,  task)

        tr_subset = torch.utils.data.Subset(full_train_dataset, tr_idx)
        te_subset = torch.utils.data.Subset(full_test_dataset,  te_idx)
        print("len train subset: ", task, " is: ", len(tr_subset))
        print("len test subset: ", task, " is: ", len(te_subset))

        tr_loader = torch.utils.data.DataLoader(tr_subset, batch_size=batch_size, shuffle=True,  **kwargs)
        te_loader = torch.utils.data.DataLoader(te_subset, batch_size=batch_size, shuffle=False, **kwargs)

        train_loader_list.append(tr_loader)
        test_loader_list.append(te_loader)

    return train_loader_list, test_loader_list
