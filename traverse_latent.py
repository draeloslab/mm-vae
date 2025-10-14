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
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture

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
#gives train and test data separately
def create_umap_data(labels, batch_size):
    #THERE HAS TO BE A BETTER WAY TO DO THIS
    print("labels: ", labels)
    #can only access dataset 1 at a time, easier to make a large dataloader. mayber theres a better way
    full_train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
    tr_idx = _indices_for_labels(full_train_dataset, labels)
    tr_subset = torch.utils.data.Subset(full_train_dataset, tr_idx)
    #make a big DataLoader so you dont have to loop
    tr_loader = torch.utils.data.DataLoader(tr_subset, batch_size=batch_size, shuffle=False)
    full_test_dataset =datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())
    te_idx = _indices_for_labels(full_test_dataset, labels)
    te_subset = torch.utils.data.Subset(full_test_dataset, te_idx)
    #make a big DataLoader so you dont have to loop
    te_loader = torch.utils.data.DataLoader(te_subset, batch_size=batch_size, shuffle=False)

    # for (data, label_data) in tr_loader:
    #     X_train = data
    #     y_train = label_data
    #     print("UMAP CHECK:")
    #     print(len(X_train))
    #     print(len(y_train))

    # for (data, label_data) in te_loader:
    #     X_test = data
    #     y_test = label_data
    #     print("UMAP CHECK:")
    #     print(len(X_test))
    #     print(len(y_test))

    return tr_loader, te_loader

#plots the UMAPS with means of multi-gaussina and variance if applicable. also can do projection
def plot_umap(path, task_num, epoch, embedding, labels_np, data_type, avg_recon, avg_dkl, ari, nmi,  projected, gmm_centers = None, gmm_std = None, reducer = None, std_samples = 360):#ari, nmi,  projected):

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

     # GMM prior centers + 1-std, this would be more complex if the stds were different or learnable obviously
    if (gmm_centers is not None) and (reducer is not None)and (gmm_std is not None):
        centers_np = gmm_centers.detach().cpu().numpy()
        k, latent_dim = centers_np.shape      

        #project centers
        centers_2d = reducer.transform(centers_np)
        #plot centers as X
        ax.scatter(centers_2d[:,0], centers_2d[:,1],
                   marker='x', s=80, linewidths=2, zorder=5, c='black')

        # std
        rng = np.random.default_rng(42)
        colors = plt.cm.Accent(np.linspace(0, 1, k))
        for i in range(k):
            points = rng.normal(size=(std_samples, latent_dim)).astype(np.float32) #draw random points in latent 
            points /= np.linalg.norm(points, axis=1, keepdims=True) + 1e-12 #make it unit length (add a little cuz 0)
            shell = centers_np[i][None, :] + gmm_std * points # make it a std away
            shell_2d = reducer.transform(shell) #put into umap space

            #find the middle of the random points and then order them in a circle so we can connect them gpt help
            #print(centers_2d)
            ctr = centers_2d[i]
            ang = np.arctan2(shell_2d[:,1] - ctr[1], shell_2d[:,0] - ctr[0])
            order = np.argsort(ang, kind='mergesort')
            ring = shell_2d[order]

            ring_closed = np.vstack([ring, ring[0:1]])   # repeat first point to close
            ax.plot(ring_closed[:,0], ring_closed[:,1],
                    color=colors[i], linewidth=1.6, alpha=0.9, zorder=4,
                    solid_joinstyle='round', solid_capstyle='round')


    if not projected:
        textstr = f"avg recon loss: {avg_recon:.4f}\navg dkl loss: {avg_dkl:.4f}\nARI score: {ari:.4f}\nNMI score: {nmi:.4f}"#\n dunn_score: {sil_score:.4f}" #\nARI score: {ari:.4f}\nNMI score: {nmi:.4f}"
        ax.text(
            0.98, 0.02, textstr,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.3')
        )

        plt.title(data_type + " data UMAP of Latent space for task: " + str(task_num) + " epoch: " + str(epoch))
        plt.tight_layout()
        plt.savefig(path + "_" + data_type + "_data.png")
        plt.close()
    else: 
        plt.title("Projected " + data_type + " data UMAP of Latent space for task: " + str(task_num)+ " epoch: " + str(epoch))
        plt.tight_layout()
        plt.savefig(path + "_" + data_type +  "_epoch_" + str(epoch)+"_data_projected.png")
        plt.close()

#gets the umap embedding and reducer used in plot_umap
def get_embedding(loader, device, model):
    feats, labels = [], []
    for (data, label_data) in loader:
        X = data.to(device).view(data.size(0), -1) #flatten to N, 784 for input
        mu, _ = model.encode(X)
        y = label_data
        feats.append(mu.detach().cpu().numpy()) #technically don't need detach because of no_grad()
        labels.append(y.detach().cpu().numpy())


    feats_np  = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    labels_np = np.concatenate(labels, axis=0)

    reducer = umap.UMAP(random_state=1) #probably gonna want to ask about hyperparams
    reducer = reducer.fit(feats_np)
    embedding = reducer.transform(feats_np)
    print(embedding.shape)
    return embedding, labels_np, feats_np, reducer
    
#gets everything and plots umap
def umap_vis(model, epoch, task_num, umap_tr_loader, umap_te_loader, device, path, avg_recon, avg_dkl, test_recon, test_dkl, projected = False, gmm_centers = None, gmm_std = None):  #https://umap-learn.readthedocs.io/en/latest/basic_usage.html
                #theres a thing where you can see the numbers on the umap, but not worht the time because rna-seq
    model.eval() #you need with no_grad, some things act differently unless in eval mode
    with torch.no_grad():
        #run the umap data through the encoder
        
        embedding, labels_np_train, mu_train, reducer= get_embedding(umap_tr_loader, device, model)
        ari, nmi = get_ARI_NMI(umap_tr_loader, model, device)
        sil_score = get_silhouette_score(umap_tr_loader, model, device)
        plot_umap(path, task_num, epoch, embedding, labels_np_train, "Train", avg_recon, avg_dkl, ari, nmi, projected, gmm_centers = gmm_centers, gmm_std = gmm_std, reducer = reducer) #ari, nmi, projected)


        #TEST DATA
        #run the umap data through the encoder
        embedding, labels_np_test, mu_test, reducer= get_embedding(umap_te_loader, device, model)
        ari, nmi = get_ARI_NMI(umap_te_loader, model, device)
        plot_umap(path, task_num, epoch, embedding, labels_np_test, "Test", test_recon, test_dkl, ari, nmi, projected, gmm_centers = gmm_centers, gmm_std = gmm_std, reducer = reducer) #ari, nmi, projected)

    return mu_train, mu_test, labels_np_train, labels_np_test

def project_umap_helper(projection_info, projection_labels, tasks, epochs, model_num, split_num, trte):
    reducer = umap.UMAP(random_state=1)
    final_key = str(tasks-1)+ "_" + str(epochs)+trte
    print("FINAL KEY: ", final_key)
    reducer = reducer.fit(projection_info[str(tasks-1)+ "_" + str(epochs)+"train"])

    for task in range(tasks):
        for epoch in range(1, epochs + 1):
            #basically train the last umap using the final data encoded on the final model
            #then run with the same umap, the previous data from each epoch (using the previous model)
            umap_path = 'results/' + model_num + '/split'+ split_num +'/UMAPs/projected/UMAP_' + 'task_' + str(task)
            embedding = reducer.transform(projection_info[str(task)+"_" + str(epoch)+trte])

            labels = projection_labels[str(task)+"_" + str(epoch)+trte]

            print("FEW PRINTS TO TEST PROJECTION: embedding/labels")
            print(len(embedding))
            print(len(labels))
            plot_umap(umap_path, task, epoch, embedding, labels, trte, -1, -1, True)

def project_umap(projection_info, projection_labels, tasks, epochs, model_num, split_num):
        #should make a funciton to split this into train and test
        project_umap_helper(projection_info, projection_labels, tasks, epochs, model_num, split_num, "train")
        project_umap_helper(projection_info, projection_labels, tasks, epochs, model_num, split_num, "test")


def get_encode_shape(loader, model, device):
    first_batch = next(iter(loader))
    inputs, labels = first_batch
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(inputs.to(device))
    out_shape = mu.shape[1:]
    return out_shape


def find_means(loader, model, device):
    #I could just append to the numpy array, but thats a pain cuz you copy over
    #Instead I run a quick encode to find the final size to initialize so Im not hardcoding

    #When you do iter(loader), you get a fresh iterator each time.
    #That means next(iter(loader)) just makes a temporary iterator, gives you the first batch, and then
    #I was worried this was gonna break stuff before or after

    #I'm assuming were not gonna want to hold all the embeddings for this, instead better to loop twice
    label_means = {}
    label_counts = {}

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            #print(inputs.size())
            mus, _ = model.encode(inputs.view(-1, 784).to(device))   # (batch_size, latent_dim) 784 is 28 by 28 image btw
            #print(mus.size())
            mus = mus.cpu().numpy()
            labels = labels.cpu().numpy()

            for mu, label in zip(mus, labels):
                if label not in label_means:
                    label_means[label] = mu.astype(np.float32, copy=False) #typing for dist_from_means
                    label_counts[label] = 1
                else:
                    label_counts[label] += 1
                    n = label_counts[label]
                    old_mean = label_means[label]
                    label_means[label] = old_mean + (mu - old_mean)/n #quick mafs

    return label_means

def dist_from_means(loader, model, device, means): #little help from gpt, but the algorithm definitely checks out
    intra_sum = {label: 0.0 for label in means}
    intra_cnt = {label: 0   for label in means}
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            mus, _ = model.encode(inputs.view(-1, 784).to(device))   # (batch_size, latent_dim)
            mus =  mus.cpu().numpy().astype(np.float32, copy=False)
            labels = labels.cpu().numpy()

            for mu, label in zip(mus, labels):
                if label in means:
                    dist = np.linalg.norm(mu - means[label])
                    intra_sum[label] += float(dist)
                    intra_cnt[label] += 1

    intra_dists = {
        lbl: (intra_sum[lbl] / intra_cnt[lbl]) if intra_cnt[lbl] > 0 else np.nan
        for lbl in means
    }

    # Inter: avg distance from each centroid to all other centroids
    inter_dists = {}
    key_list = list(means)
    for i, label in enumerate(key_list):
        mean = means[label]
        acc, k = 0.0, 0
        for lj in key_list:
            if label == lj: 
                continue #dont count dist to self
            acc += float(np.linalg.norm(mean - means[lj]))
            k += 1
        inter_dists[label] = (acc / k) if k > 0 else np.nan

    return inter_dists, intra_dists

def get_ARI_NMI(loader, model, device):
    print("for timing", flush = True)
    X = []
    labels_tot = []
    n = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            #print(inputs.size())
            mus, _ = model.encode(inputs.view(-1, 784).to(device))   # (batch_size, latent_dim) 784 is 28 by 28 image btw
            #print(mus.size())
            mus = mus.cpu().numpy()
            labels2 = labels.cpu().numpy()

            X.append(mus)
            labels_tot.append(labels2)
            n += labels.size(0)

        X = np.concatenate(X, axis=0)
        #X = StandardScaler().fit_transform(X) #to avoid a certain dimension from over powering
        y = np.concatenate(labels_tot, axis=0)

        unique_labels = np.unique(y)
        if unique_labels.size < 2:
            print("not enough labels", flush=True)
            return float("nan"), float("nan")
        
        k = unique_labels.size
        gmm = GaussianMixture(
        n_components=k,
        covariance_type='full',
        n_init=10,
        reg_covar=1e-6,
        random_state=42,
        init_params='kmeans'
        )
        yk = gmm.fit_predict(X)  # component labels

        ari = float(adjusted_rand_score(y, yk))
        nmi = float(normalized_mutual_info_score(y, yk))
       
        # k = unique_labels.size
        # km = KMeans(n_clusters=k, n_init=10, random_state=42)
        # yk = km.fit_predict(X)

        # ari = float(adjusted_rand_score(y, yk))
        # nmi = float(normalized_mutual_info_score(y, yk))

        print("end timing", flush=True)
    return ari, nmi

def get_silhouette_score(loader, model, device):
    print("for timing", flush = True)
    X = []
    labels_tot = []
    n = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            #print(inputs.size())
            mus, _ = model.encode(inputs.view(-1, 784).to(device))   # (batch_size, latent_dim) 784 is 28 by 28 image btw
            #print(mus.size())
            mus = mus.cpu().numpy()
            labels2 = labels.cpu().numpy()

            X.append(mus)
            labels_tot.append(labels2)
            n += labels.size(0)

        X = np.concatenate(X, axis=0)
        y = np.concatenate(labels_tot, axis=0)

        score = silhouette_score(X, y)

    print("end timing", flush = True)
    return score





def quant_clusters(model, loader_tr, loader_te, task, device):
    model.eval() #you need with no_grad, some things act differently unless in eval mode
    with torch.no_grad():
        means = find_means(loader_tr, model, device)
        inter_dists_tr, intra_dists_tr = dist_from_means(loader_tr, model, device, means)
        ari_tr, nmi_tr = get_ARI_NMI(loader_tr, model, device)
        #sil_score_tr = get_silhouette_score(loader_tr, model, device)

        means = find_means(loader_te, model, device)
        inter_dists_te, intra_dists_te = dist_from_means(loader_te, model, device, means)
        ari_te, nmi_te = get_ARI_NMI(loader_te, model, device)
        #sil_score_te = get_silhouette_score(loader_te, model, device)

        

    return inter_dists_tr, intra_dists_tr, inter_dists_te, intra_dists_te, ari_tr, nmi_tr, ari_te, nmi_te#sil_score_tr, sil_score_te
