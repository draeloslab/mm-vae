import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset 

def getDataLoaders(batch_size, shuffle=True, device='cuda'):
    if not (os.path.exists('/home/rachel/Desktop/mm-vae/data/train-ms-mnist-idx.pt')
            and os.path.exists('/home/rachel/Desktop/mm-vae/data/train-ms-svhn-idx.pt')
            and os.path.exists('/home/rachel/Desktop/mm-vae/data/test-ms-mnist-idx.pt')
            and os.path.exists('/home/rachel/Desktop/mm-vae/data/test-ms-svhn-idx.pt')):
        raise RuntimeError('Generate transformed indices with the script in bin')
    # get transformed indices
    t_mnist = torch.load('/home/rachel/Desktop/mm-vae/data/train-ms-mnist-idx.pt')
    t_svhn = torch.load('/home/rachel/Desktop/mm-vae/data/train-ms-svhn-idx.pt')
    s_mnist = torch.load('/home/rachel/Desktop/mm-vae/data/test-ms-mnist-idx.pt')
    s_svhn = torch.load('/home/rachel/Desktop/mm-vae/data/test-ms-svhn-idx.pt')

    # load base datasets
    tx = transforms.ToTensor()
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    train_mnist = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_mnist = DataLoader(datasets.MNIST('../data', train=False, download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
    train_svhn = DataLoader(datasets.SVHN('../data', split='train', download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_svhn =  DataLoader(datasets.SVHN('../data', split='test', download=True, transform=tx),
                          batch_size=batch_size, shuffle=shuffle, **kwargs)

    train_mnist_svhn = TensorDataset([
        ResampleDataset(train_mnist.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
        ResampleDataset(train_svhn.dataset, lambda d, i: t_svhn[i], size=len(t_svhn))
    ])
    test_mnist_svhn = TensorDataset([
        ResampleDataset(test_mnist.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
        ResampleDataset(test_svhn.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
    ])

    kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
    train = DataLoader(train_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
    return train, test