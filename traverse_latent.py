from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from get_dataset_first_iter import _indices_for_labels
from matplotlib.colors import ListedColormap, BoundaryNorm
import umap

# I also want to create a way to traverse the latent space by taking the means of two numbers and going in a straight line, 
#then also by implementing stepwise density based trajectory

#I want to make a map of each cluster as well, probably some code already exists out there

#using accuracy to judge

def traverse_latent_linear():
    pass

def traverse_latent_stepwise_density():
    pass

#this function just returns a numpy array usable with the umap function
#the numpy array is composed of the data from the current task and all previous tasks the model has trained on
#right now this uses train data, but maybe it should be used on test data or a combination
def create_umap_data(labels):
    print("labels: ", labels)
    #can only access dataset 1 at a time, easier to make a large dataloader. mayber theres a better way
    full_train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
    tr_idx = _indices_for_labels(full_train_dataset, labels)
    tr_subset = torch.utils.data.Subset(full_train_dataset, tr_idx)
    #make a big DataLoader so you dont have to loop
    tr_loader = torch.utils.data.DataLoader(tr_subset, batch_size=len(tr_subset.dataset), shuffle=False)
    full_test_dataset =datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())
    te_idx = _indices_for_labels(full_test_dataset, labels)
    te_subset = torch.utils.data.Subset(full_test_dataset, te_idx)
    #make a big DataLoader so you dont have to loop
    te_loader = torch.utils.data.DataLoader(te_subset, batch_size=len(te_subset.dataset), shuffle=False)

    for (data, label_data) in tr_loader:
        X_train = data
        y_train = label_data
        print("UMAP CHECK:")
        print(len(X_train))
        print(len(y_train))

    for (data, label_data) in te_loader:
        X_test = data
        y_test = label_data
        print("UMAP CHECK:")
        print(len(X_test))
        print(len(y_test))

    return X_train, y_train, X_test, y_test

def plot_umap(path, task_num, epoch, embedding, labels_np, data_type):

    plt.figure(figsize=(6,5), dpi=140)
    # use 10 distinct colors rather than a continuous colormap
    tab10 = plt.get_cmap('tab10')
    cmap  = ListedColormap([tab10(i) for i in range(10)])

    bounds = np.arange(-0.5, 10.5, 1)   # [-0.5, 0.5, 1.5, ..., 9.5, 10.5]
    norm   = BoundaryNorm(bounds, cmap.N)

    sc = plt.scatter(embedding[:, 0], embedding[:, 1],
                    c=labels_np, cmap=cmap, norm=norm, s=5)

    ax = plt.gca()
    ax.set_aspect('equal', 'datalim')

    cbar = plt.colorbar(sc, boundaries=bounds, ticks=np.arange(10))
    cbar.set_label('Digit')

    plt.title(data_type + " data UMAP of Latent space for task: " + str(task_num) + " epoch: " + str(epoch))
    plt.tight_layout()
    plt.savefig(path + "_" + data_type + "_data.png")
    plt.close()

def get_embedding(X, y, device, model):
    X = X.to(device).view(X.size(0), -1) #flatten to N, 784 for input
    mu, _ = model.encode(X)

    feats_np = mu.detach().cpu().numpy() #technically don't need detach because of no_grad()
    labels_np = y.detach().cpu().numpy()

    reducer = umap.UMAP(random_state=1) #probably gonna want to ask about hyperparams
    embedding = reducer.fit_transform(feats_np)
    print(embedding.shape)
    return embedding, labels_np
    

def umap_vis(model, epoch, task_num, X_train, y_train, X_test, y_test, device, path):  #https://umap-learn.readthedocs.io/en/latest/basic_usage.html
                #theres a thing where you can see the numbers on the umap, but not worht the time because rna-seq
    model.eval() #you need with no_grad, some things act differently unless in eval mode
    with torch.no_grad():
        #run the umap data through the encoder
        
        embedding, labels_np= get_embedding(X_train, y_train, device, model)
        plot_umap(path, task_num, epoch, embedding, labels_np, "Train")


        #TEST DATA
        #run the umap data through the encoder
        embedding, labels_np = get_embedding(X_test, y_test, device, model)
        plot_umap(path, task_num, epoch, embedding, labels_np, "Test")