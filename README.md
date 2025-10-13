# mm-vae
Multimodal VAE development

test push branch
train_first_iter.py---------main code to run

vanillaVAE.py---------------pytorch examples VAE that continual_VAE_first_iter inherits from

continual_VAE_first_iter.py-has different VAE models that can be importated to train_first_iter and inherits from vanillaVAE

traverse_latent.py----------contains functions helpfull for traversing the latent space, used in train_first_iter.
it also contains all the functions to plot umaps and quantify clustering

get_dataset_first_iter.py---contains functions helpful for splitting the mnist dataset into different tasks

quant_format_helper---------just makes a text file in results/{model}/splitx_cluster_quant that saves the inter/intra cluster distances and metrics (I tried sillhouette score, dunn, ari/nmi compared to k means). dunn bad cuz outliers, sillhouette bad cuz pushes all to prior, air/nmi seemed to fit intuition


VAE that inherits from the original VAE that implements slowed learning rates from https://arxiv.org/pdf/1612.00796
generated images from https://arxiv.org/pdf/1906.03288, and a way to test two tasks A and B in 4 different ways
   0. Default, no split
   1. A = 0-4, B = 5-9
   2. A=even, B=odd
   3. A = 0-4, B = 0-5
   4. A = 0-1, B = 2-3, C = 4-5...

gifs_from_images.py just makes gifs from certain images, a lot of hardcoded stuff so probably wont be too usefull later on

