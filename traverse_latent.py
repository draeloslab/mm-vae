from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# I also want to create a way to traverse the latent space by taking the means of two numbers and going in a straight line, 
#then also by implementing stepwise density based trajectory

#I want to make a map of each cluster as well, probably some code already exists out there

#using accuracy to judge

def traverse_latent_linear():
    pass

def traverse_latent_stepwise_density():
    pass

def visualize_clusters():
    pass