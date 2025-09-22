from torchvision import transforms

def unnormalize(x, mean, std):
    if (isinstance(mean, float)):
        new_mean = -mean/std
        new_std = 1/std
    else:
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
    unnorm = transforms.Normalize(new_mean, new_std)
    return unnorm(x) * 255

# def unnormalize(x, mean, std):
#     if (isinstance(mean, float)):
#         out = x * std + mean
#     else: 
#         out = [x * s + m for m, s in zip(mean, std)]
#     return out * - 1