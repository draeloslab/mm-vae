# mm-vae
Multimodal VAE development


vanillaVAE.py---------------pytorch examples VAE that continual_VAE_first_iter inherits from. I made a few small changes to visualize, but otherwise its the same.

train_first_iter.py---------!!!main code to run!!! 
- If you want to change the arguments, you have to do it in vanillaVAE too.
- I commented out the data generation because I didn't need it, if you do I would recommend making it work for different model sizes
- This line below is where you can limit how many UMAPS are made
- if args.visualize:# and epoch == 5 and task == 4: #temp thing so I dont make a umap for each epoch (expensive)

continual_VAE_first_iter.py-has different VAE models that can be importated to train_first_iter and inherits from vanillaVAE. 
- Right now there is a single gaussian prior model and a multi-gaussian model prior. 
- !!!For the multi-gaussian, you have to change the parameters of the model in here, rather than the command line!!!

get_dataset_first_iter.py---contains functions helpful for splitting the mnist dataset into different tasks

latent_regularizer.py-------used in the loss function for the multi-gaussian. file from VAE_code_SEED from rachel

loss.py---------------------Loss function used in the multi-gaussian. I changed it to match the loss for the single gaussian from the vanilla VAE. I didnt need anything with cfm is the main difference, but the original is below commented out.

quant_format_helper---------just makes a text file in results/{model}/splitx_cluster_quant that saves the inter/intra cluster distances and metrics (I tried sillhouette score, dunn, ari/nmi compared to k means or GMM). dunn bad due to outliers, sillhouette bad due to pushing to prior, air/nmi seemed to fit intuition but plateaued with k-means, ari/nmi with GMM did better, but there are times it doesnt match recon loss.

traverse_latent.py----------contains functions helpfull for traversing the latent space (didnt get to), used in train_first_iter.
it also contains all the functions to plot umaps and quantify clustering. Including projection

gifs_from_images.py---------makes gifs from certain images, a lot of hardcoded stuff so probably wont be too usefull


VAE that inherits from the original VAE that implements slowed learning rates from https://arxiv.org/pdf/1612.00796
generated images from https://arxiv.org/pdf/1906.03288, and a way to test two tasks A and B in 4 different ways
   0. Default, no split
   1. A = 0-4, B = 5-9
   2. A=even, B=odd
   3. A = 0-4, B = 0-5
   4. A = 0-1, B = 2-3, C = 4-5...



