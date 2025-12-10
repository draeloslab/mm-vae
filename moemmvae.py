# this file will have the model architecture for the MoE mmVAE model 
# from __future__ import print_function
import torch.distributions as dist
from vae_mnist import*
from vae_svhn import*
from utils import*
from torchvision.utils import save_image
import math

class MMVAE(nn.Module):
    def __init__(self, latent_dim, mnist_scale, svhn_scale, learn_prior: bool = False):
        super(MMVAE, self).__init__()

        #self.mnistvae = VAE_MNIST(**mnistvae_params)
        #self.svhnvae = VAE_SVHN(**svhnvae_params)
        ## add parameter initializations in main class
        self.mnistvae = VAE_MNIST(latent_dim, mnist_scale)
        self.svhnvae = VAE_SVHN(latent_dim, svhn_scale)
        self.vaes = nn.ModuleList([self.mnistvae, self.svhnvae])
        self.scaling = [mnist_scale, svhn_scale]
        self.M = len(self.vaes)
        self.latent_dim = latent_dim

        # add scaling factors for calculating combined loss across the two modalities 
        self.mnistvae.like_scale = mnist_scale
        self.svhnvae.like_scale = svhn_scale

        # initializing learned prior
        # dictionary object to determine if prior will be learned or not.
        grad = {'requires_grad': learn_prior}

        self._pz_params = nn.ParameterList([
            # mean is always fixed at 0
            nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=False), 
            # logscale may changed if learned prior is set to true 
            nn.Parameter(torch.zeros(1, self.latent_dim, **grad))
        ])
    
    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)
    
    # define a property for the prior with learned parameters 
    @property
    def pz(self): 
        mu, scale = self.pz_params
        return dist.Laplace(mu, scale)

    # define the fixed shared prior (laplace distribution with mean 0 and std 1), dimensions equal to the number of latent dimensions
    def get_pz(self):
        # finds the device that the vae parameters are on 
        device = next(self.mnistvae.parameters()).device
        # creates the mu and sigma tensors on that device
        mu = torch.zeros(1, self.latent_dim, device=device)
        sigma = torch.ones(1, self.latent_dim, device=device)

        # laplace prior for now because that is what was used in the paper
        return dist.Laplace(mu, sigma)
    
    def forward(self, x, num_samples):
        # forward pass through joint encoder distributions 
        # qz_xs stores posteriors qz_x for each modality 
        qz_xs = []
        # zss stores the latent samples taken from each modality (num samples x batch size)
        zss = []
        # px_zs computes the cross-modal likelihood matrix. 
        # ex. cross modal elements are computed by taking samples from one encoder and using the other modalities decoder
        px_zs = [[None for num in range(self.M)] for num in range(self.M)]

        for m, vae in enumerate(self.vaes):
            # qz_x is the posterior distribution 
            # px_z is the self-reconstruction likelihood (diagonal of px_zs)
            # zs is the latent samples 
            qz_x, px_z, zs = vae(x[m], num_samples) # num samples is relevant for calculating IWAE loss 
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z
        
        # cross-modal decoding 
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:
                    mean, scale = vae.decode(zs)
                    px_zs[e][d] = vae.px_z(mean, scale)
        
        return qz_xs, zss, px_zs
    
    # first initializing MoE ELBO loss and then will transition to calculating IWAE over K (num_samples)
    # computes ELBO loss (this is essentially IWAE but with K=1 samples)
    def moe_elbo_loss(self, x_data, K=1):
        #pz = self.get_pz()
        mu, scale = self.pz_params
        pz  = dist.Laplace(mu, scale)
        qz_xs, zss, px_zs = self.forward(x_data, num_samples=K)

        log_iws = [] # stores importance weights 

        for encoder in range(self.M):
            z_r = zss[encoder]
            K, B = zss[encoder].shape[:2]
            '''
            There are four terms that comprise the ELBO loss calculation below. These four terms are the log-prior term (lpz), the MoE log-posterior term (lqz_x),
            the joint scaled log-likelihood term (lpx_z) and the log importance weights terms (log_iw) 
            '''
            # log prior term
            lpz = pz.log_prob(z_r).sum(-1)
            # calculate the log posterior term averaged over all modalities
            lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(z_r).sum(-1) for qz_x in qz_xs]), dim=0)
            # joint scaled log likelihood term
            lpx_all = [] 
            for m in range(self.M):
                if m == 0 and encoder == 0: 
                    batch_length = x_data[m].shape[0]
                    x_data[m] = x_data[m].reshape(1, batch_length, -1)
                log_prob = px_zs[encoder][m].log_prob(x_data[m])
                # scale 
                #log_prob_scaled = log_prob.view(*log_prob.shape[:2], -1).sum(-1) * self.vaes[m].like_scale
                scale = self.vaes[m].like_scale
                log_prob_scaled = log_prob.reshape(K*B, -1).sum(dim=1).reshape(K, B) * scale
                lpx_all.append(log_prob_scaled)
            lpx_z = torch.stack(lpx_all).sum(0)
            # log importance weight
            lw = lpz + lpx_z - lqz_x
            #print(lpz.shape, lpx_z.shape, lqz_x.shape)
            log_iws.append(lw)
        lw_all = torch.cat(log_iws, dim=0)
        # total ELBO is averaged log weights across m modalities 
        avg_log_weight = lw_all.mean(dim=0).sum()
        return -avg_log_weight
    
    # computes IWAE loss (very similar to ELBO but with K>1 samples)
    def moe_iwae_loss(self, x_data, K=1):
        #pz = self.get_pz()
        mu, scale = self.pz_params
        pz  = dist.Laplace(mu, scale)
        qz_xs, zss, px_zs = self.forward(x_data, num_samples=K)

        log_iws = [] # stores importance weights 

        for encoder in range(self.M):
            z_r = zss[encoder]
            K, B = zss[encoder].shape[:2]
            '''
            There are four terms that comprise the ELBO loss calculation below. These four terms are the log-prior term (lpz), the MoE log-posterior term (lqz_x),
            the joint scaled log-likelihood term (lpx_z) and the log importance weights terms (log_iw) 
            '''
            # log prior term
            lpz = pz.log_prob(z_r).sum(-1) 
            # calculate the log posterior term averaged over all modalities
            lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(z_r).sum(-1) for qz_x in qz_xs]), dim=0)
            # joint scaled log likelihood term
            lpx_all = [] 
            for m in range(self.M):
                # maybe should be changed to be more robust
                batch_length = x_data[m].shape[0]
                if m == 0 and encoder == 0: 
                    x_data[m] = x_data[m].reshape(1, batch_length, -1)
                log_prob = px_zs[encoder][m].log_prob(x_data[m])
                # scale 
                #log_prob_scaled = log_prob.view(*log_prob.shape[:2], -1).sum(-1) * self.vaes[m].like_scale
                scale = self.vaes[m].like_scale
                log_prob_scaled = log_prob.reshape(K*B, -1).sum(dim=1).reshape(K, B) * scale
                lpx_all.append(log_prob_scaled)
            lpx_z = torch.stack(lpx_all).sum(0)
            # log importance weight
            lw = lpz + lpx_z - lqz_x
            #print(lpz.shape, lpx_z.shape, lqz_x.shape)
            log_iws.append(lw)
        lw_all = torch.cat(log_iws, dim=0)
        # total ELBO is averaged log weights across m modalities 
        # modify this portion to accept K values greater than 1
        lw_all_reshape = lw_all.view(self.M, K, -1) # transform from (M*K, B) to (M, K, B)
        log_p_x = log_mean_exp(lw_all_reshape)
        avg_log_weight = log_p_x.sum(dim=0).mean()
        return -avg_log_weight

    # The following functions are for visualizing the reconstructed data
    def decode_from_z(self, z):
        all_means = []
        for vae in self.vaes:
            mean = vae.decode(z)[0].detach()
            all_means.append(mean)
        return all_means

    def generate(self, output_path, epoch):
        # Generates samples (N = num samples) from the prior and decodes them for both modalities 
        N = 64 
        zs  = self.pz.rsample(torch.Size([N])).squeeze(1)
        decoded_samples = self.decode_from_z(zs)
        for i, samples in enumerate(decoded_samples):
            samples = samples.data.cpu().view(N, *samples.size()[1:])
            save_image(samples, '{}/random_generations_{}_{}.png'.format(output_path, i, epoch), nrow=int(math.sqrt(N)))

    def reconstruct(self, data, output_path, epoch):
        # Computes reconstructions for the first N samples
        N = 10 

        device = next(self.parameters()).device
        x_mnist_full = data[0].to(device)
        x_svhn_full = data[1].to(device)
        x_mnist_full = x_mnist_full.squeeze()
        x_mnist_sub = x_mnist_full[:N]
        x_svhn_sub = x_svhn_full[:N]
        svhn_size = x_svhn_sub.size()[1:]
        input_samples = [x_mnist_sub, x_svhn_sub]
        for i, vae in enumerate(self.vaes):
            input_sample = input_samples[i]
            mu, logvar = vae.encode(input_sample)
            scale = torch.exp(logvar)
            qz = vae.qz_x(mu, scale)
            z = qz.rsample().unsqueeze(0)
            for o, vae_out in enumerate(self.vaes): 
                params = vae_out.decode(z)
                recon = params[0].squeeze(0).cpu()
                in_data = input_sample.cpu()
                in_data = in_data.squeeze()
                if in_data.size()[1:] != svhn_size:
                    in_data = resize_img(in_data, svhn_size)
                if recon.size()[1:] != svhn_size:
                    recon = resize_img(recon, svhn_size)
                pair = torch.cat([in_data, recon], dim=-1)
                save_image(pair, '{}/recon_{}x{}_{:03d}.png'.format(output_path, i, o, epoch))

    @torch.no_grad()
    def get_latent_space(self, x_data, encoder=0):
        vae = self.vaes[encoder]
        input_data = x_data[encoder]

        if encoder == 0:
            B = input_data.size(0)
            input_data = input_data.view(B, -1)

        mu, logvar = vae.encode(input_data)

        return mu
    
# loss functions from paper to test 

    def _m_iwae(self, x, K=1):
        """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
        qz_xs, zss, px_zs = self.forward(x, num_samples=1)
        lws = []
        for r, qz_x in enumerate(qz_xs):
            mu, scale = self.pz_params
            pz_dist  = dist.Laplace(mu, scale)
            lpz = pz_dist.log_prob(zss[r]).sum(-1)
            lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))

            lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                        .mul(self.scaling[d]).sum(-1)
                    for d, px_z in enumerate(px_zs[r])]
            lpx_z = torch.stack(lpx_z).sum(0)
            lw = lpz + lpx_z - lqz_x
            lws.append(lw)
            lws_all = torch.cat(lws)

        return log_mean_exp(lws_all).sum()

# m_dreg is actually the loss function they ended up using for the final MNIST/SVHN results I believe 

    def _m_dreg_looser(self, x, K=1):
        """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
        This version is the looser bound---with the average over modalities outside the log
        """
        qz_xs, zss, px_zs = self.forward(x, num_samples=KeyboardInterrupt)
        qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
        lws = []
        for r, vae in enumerate(self.vaes):
            lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
            lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
            lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                        .mul(model.vaes[d].llik_scaling).sum(-1)
                    for d, px_z in enumerate(px_zs[r])]
            lpx_z = torch.stack(lpx_z).sum(0)
            lw = lpz + lpx_z - lqz_x
            lws.append(lw)
        return torch.stack(lws), torch.stack(zss)





        
            


            
        


