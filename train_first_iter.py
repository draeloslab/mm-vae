from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from get_dataset_first_iter import get_train_test_loaders
from traverse_latent import umap_vis, create_umap_data, project_umap
from continual_VAE_first_iter import default_VAE#, weighted_VAE, generative_VAE

import os, sys
print("[DEBUG] running:", __file__)
print("[DEBUG] argv:", sys.argv)

#   0. Default, no split
#   1. A = 0-4, B = 5-9
#   2. A=even, B=odd
#   3. A = 0-4, B = 0-5
#   4. A = 0-1, B = 2-3, C = 4-5...

def get_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train per task(default: 5)')
    parser.add_argument('--no-accel', action='store_true', 
                        help='disables accelerator')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--split', type=int, default=0, metavar='N',
                        help='what task splitting to use 0 = no split, 1 = 0-4, 5-9, 2 = even/odd, 3 = 0-4, 0-5, 4 = every 2 (5 total)')
    parser.add_argument('--model', type=int, default=0, metavar='N',
                        help='what model to use, 0 = default, 1 = weighted, 2 = generative')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='wheter to visualize with umaps')
    args = parser.parse_args()
    return args

#To get it to work on my device (just what gpt said worked)
def pick_device(disable=False):
    if disable:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (harmless to check on other platforms)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    # Intel GPU (oneAPI) if you happen to have it
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

def other_setup(args):
    device = pick_device(args.no_accel)
    use_accel = device.type in ("cuda", "mps", "xpu")
    #end of what gpt said. should change back to below on other
    #use_accel = not args.no_accel and torch.accelerator.is_available() was the original

    torch.manual_seed(args.seed)


    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_accel else {}
    return device, kwargs


def main(args, device, kwargs):
    possible_split_dict = {
        0: [(0,1,2,3,4,5,6,7,8,9)],
        1: [(0,1,2,3,4), (5,6,7,8,9)],
        2: [(0,2,4,6,8), (1,3,5,7,9)],
        3: [(0,1,2,3,4), (0,1,2,3,4,5)],
        4: [(0,1), (2,3), (4,5), (6,7), (8,9)]
    }

    #returns a list of training and testing loaders associated with each task
    train_loader_list, test_loader_list = get_train_test_loaders(args.batch_size, possible_split_dict[args.split])

    model_dict = {
        0: 'default',
        1: 'weighted',
        2: 'generative'
    }
    if args.model == 0:
        model = default_VAE().to(device)
    elif args.model == 1:
        model = weighted_VAE().to(device)
    elif args.model == 2:
        model = generative_VAE().to(device)
    
    
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    projection_info = {}
    projection_labels = {}
    for task in range(len(train_loader_list)):
        if args.visualize:
            umap_X_train, umap_y_train, umap_X_test, umap_y_test = create_umap_data(list(set(x for tup in possible_split_dict[args.split][:task+1] for x in tup)))
        
        #^ just makes a list of all the numbers seen so far to create the umap
        print("training task: ", task)
        for epoch in range(1, args.epochs + 1):

            avg_recon, avg_dkl = model.train_epoch(epoch, train_loader_list[task], device, optimizer, args.log_interval)
            umap_path = 'results/' + str(model_dict[args.model]) + '/split'+ str(args.split) +'/UMAPs/UMAP_' + 'task_' + str(task) + '_epoch_' +str(epoch)
            if args.visualize:
                #instead of avg loss over the epoch, you can get the final loss by returning recon_loss_tmp
                test_recon, test_dkl = model.test_epoch(epoch, test_loader_list[task], device, args.batch_size, model_dict[args.model], str(args.split), str(task), str(task), vis = False)
                mu_train, mu_test, labels_np_train, labels_np_test = umap_vis(model, epoch, task, umap_X_train, umap_y_train, umap_X_test, umap_y_test, device, umap_path, avg_recon, avg_dkl, test_recon, test_dkl)
                projection_info[str(task)+"_" + str(epoch)+"train"] = mu_train
                projection_info[str(task)+"_" + str(epoch)+"test"] = mu_test
                projection_labels[str(task)+"_" + str(epoch)+"train"] = labels_np_train
                projection_labels[str(task)+"_" + str(epoch)+"test"] = labels_np_test
            for previous_task in range(task+1):
                test_recon, test_dkl = model.test_epoch(epoch, test_loader_list[previous_task], device, args.batch_size, model_dict[args.model], str(args.split), str(task), str(previous_task), vis = args.visualize)
            with torch.no_grad():
                sample = torch.randn(64, 20).to(device) #random numbers into the decoder to generate a random digit
                sample = model.decode(sample).cpu() #generate  the new images
                save_image(sample.view(64, 1, 28, 28),
                        'results/' + model_dict[args.model] + '/split'+ str(args.split) +'/sample_' + 'task_' + str(task) + '_epoch_' +str(epoch) + '.png')
    if args.visualize:
        print("LEN PROJECCTION INFO AND PROJECTION LABELS")
        print(len(projection_info))
        print(len(projection_labels))
        project_umap(projection_info, projection_labels, len(train_loader_list), args.epochs, str(model_dict[args.model]), str(args.split))
        



if __name__ == "__main__":
    args = get_args()
    print(args)
    device, kwargs = other_setup(args)
    main(args, device, kwargs)
