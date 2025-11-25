# this file will have the model architecture for the MoE mmVAE model 
# from __future__ import print_function
import torch.distributions as dist
from vae_mnist import*
from vae_svhn import*

class MMVAE(nn.Module):
    def __init__(self, mnistvae_params, svhnvae_params, latent_dim, mnist_scale, svhn_scale):
        super(MMVAE, self).__init__()

        #self.mnistvae = VAE_MNIST(**mnistvae_params)
        #self.svhnvae = VAE_SVHN(**svhnvae_params)
        ## add parameter initializations in main class
        self.mnistvae = VAE_MNIST(**mnistvae_params)
        self.svhnvae = VAE_SVHN(**svhnvae_params)
        self.vaes = nn.ModuleList([self.mnistvae, self.svhnvae])
        self.M = len(self.vaes)
        self.latent_dim = latent_dim

        # add scaling factors for calculating combined loss across the two modalities 
        self.mnistvae.like_scale = mnist_scale
        self.svhnvae.like_scale = svhn_scale

    def forward(self, x, num_samples):
        # forward pass through joint encoder distributions 
        # qz_xs stores posteriors qz_x for each modality 
        qz_xs = []
        # zss stores the latent samples taken from each modality (num samples x batch size)
        zss = []
        # px_zs computes the cross-modal likelihood matrix. 
        # ex. cross modal elements are computed by taking samples from encoder e and using the other modalities decoder
        px_zs = [[None for num in range(self.M)] for num in range(self.M)]

        for m, vae in enumerate(self.vaes):
            # qz_x is the posterior distribution 
            # px_z is the self-reconstruction likelihood (diagonal of px_zs)
            # zs is the latent samples 
            qz_x, px_z, zs = vae(x[m], num_samples)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z
        
        # cross-modal decoding 
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:
                    px_zs[e][d] = vae.px_z(vae.decode(zs))
        
        return qz_xs, zss, px_zs
    
    def pz(self):
        device = next(self.mnistvae.parameters().device)
        mu = torch.zeros(1, self.latent_dim, device=device)
        sigma = torch.ones(1, self.latent_dim, device=device)
        return dist.Normal(mu, sigma)
    
    # first initializing MoE ELBO loss and then will transition to calculating IWAE over K (num_samples)
    def moe_elbo_loss(self, ):




    
        


        
    


