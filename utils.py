import torch
import math
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd

# constants defined in the mmvae paper (for clamping values)
class Constants(object):
    logceilc = 88
    logfloorc = -104
    eta = 1e-6

def unnormalize(x, mean, std):
    if (isinstance(mean, float)):
        new_mean = -mean/std
        new_std = 1/std
    else:
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
    unnorm = transforms.Normalize(new_mean, new_std)
    #return unnorm(x) * 255
    return unnorm(x)

# def unnormalize(x, mean, std):
#     if (isinstance(mean, float)):
#         out = x * std + mean
#     else: 
#         out = [x * s + m for m, s in zip(mean, std)]
#     return out * - 1

def log_mean_exp(value, dim=0, keepdim=False):
    # calculate log(mean(exp(value)))
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

def resize_img(img, refsize):
    # adjusts size of generated images to match reference size (which is the size of the svhn data)
    batch_size = img.size(0)
    img = img.view(batch_size, 1, 28, 28)
    img_padded = F.pad(img, (2, 2, 2, 2))
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)

def save_latent_space(model, epoch, data_loader, device, output_path, num_modalities):
    model.eval()

    latent_mus = []

    for _, data in enumerate(data_loader):
        #print(data.shape)
        x_data = [data[m][0].to(device) for m in range(num_modalities)]
        labels = data[0][1].cpu().numpy()
        
        for m in range(num_modalities):
            
            mu = model.get_latent_space(x_data, encoder=m)
            mu = mu.cpu().numpy()

            latent_dim = mu.shape[1]

            latent_space = pd.DataFrame(mu, columns=[f'LV{i+1}' for i in range(latent_dim)])

            latent_space['labels'] = labels
            latent_space['encoder'] = m

            latent_mus.append(latent_space)

    full_latent_df = pd.concat(latent_mus, ignore_index=True)
    full_latent_df.to_csv(os.path.join(output_path, 'latent_space_epoch' + str(epoch) + '.csv'), index=False)
    print('Latent space at epoch '  + str(epoch) + ' saved.')
