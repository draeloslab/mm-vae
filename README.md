# mm-vae
Multimodal VAE development

train_first_iter.py---------main code to run
vanillaVAE.py---------------pytorch examples VAE that continual_VAE_first_iter inherits from
continual_VAE_first_iter.py-has different VAE models that can be importated to train_first_iter and inherits from vanillaVAE
traverse_latent.py----------contains functions helpfull for traversing the latent space, used in train_first_iter
get_dataset_first_iter.py---contains functions helpful for splitting the mnist dataset into different tasks


First week (really weekend)
My goal is to create a VAE that inherits from the original VAE that implements slowed learning rates from https://arxiv.org/pdf/1612.00796
generated images from https://arxiv.org/pdf/1906.03288, and a way to test two tasks A and B in 4 different ways
   0. Default, no split
   1. A = 0-4, B = 5-9
   2. A=even, B=odd
   3. A = 0-4, B = 0-5
   4. A = 0-1, B = 2-3, C = 4-5...
 I also want to create a way to traverse the latent space by taking the means of two numbers and going in a straight line, 
then also by implementing stepwise density based trajectory
also want a way to visualize clusters
using accuracy to judge

Week 2:
Push to github
For the paper look to see if the classification updates the AE (the objective function)
Split the losses and do per digit and look how it relates to clusters (Umap visualization)
Quantify clusters (inter and intra distance of clusters)
Add multiple gaussian (before weighted or generative)
Look at another dataset (dsprites) (shapes). Make sure that works for any type of task type, so I can try 
Latent space linear and stepwise density
Accuracy to judge 
Complicated rna-seq data
