# this file will have the model architecture for the MoE mmVAE model 
# from __future__ import print_function
import torch.distributions as dist
from vae import*
from utils import*
from torchvision.utils import save_image
import torch.nn.functional as F
import math

class MMVAE(nn.Module):
    def __init__(self, input_dim1, input_dim2, H1_1, H2_1, H3_1, H1_2, H2_2, H3_2, latent_dim, scale1, scale2, learn_prior: bool = False):
        super(MMVAE, self).__init__()
        
        ## add parameter initializations in main class
        self.vae1 = VAE(input_dim1, H1_1, H2_1, H3_1, latent_dim, scale1)
        self.vae2 = VAE(input_dim2, H1_2, H2_2, H3_2, latent_dim, scale2)
        self.vaes = nn.ModuleList([self.vae1, self.vae2])
        self.scaling = [scale1, scale2]
        self.M = len(self.vaes)
        self.latent_dim = latent_dim

        # add scaling factors for calculating combined loss across the two modalities 
        self.vae1.like_scale = scale1
        self.vae2.like_scale = scale2

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
        #return dist.Laplace(mu, scale)
        return dist.Normal(mu, scale)

    # define the fixed shared prior (laplace distribution with mean 0 and std 1), dimensions equal to the number of latent dimensions
    # changed to defining the prior as a normal distribution
    def get_pz(self):
        # finds the device that the vae parameters are on 
        device = next(self.vae1.parameters()).device
        # creates the mu and sigma tensors on that device
        mu = torch.zeros(1, self.latent_dim, device=device)
        sigma = torch.ones(1, self.latent_dim, device=device)

        # laplace prior for now because that is what was used in the paper
        #return dist.Laplace(mu, sigma)
        # change to normal prior for testing 
        return dist.Normal(mu, sigma)
    
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
        return avg_log_weight
    
    # computes IWAE loss (very similar to ELBO but with K>1 samples)
    def moe_iwae_loss(self, x_data, beta, K=1):
        #pz = self.get_pz()
        mu, scale = self.pz_params
        #pz  = dist.Laplace(mu, scale)
        pz = dist.Normal(mu, scale)
        qz_xs, zss, px_zs = self.forward(x_data, num_samples=K)
        # print('qz_xs: ' + str(len(qz_xs)))
        # print('qz_xs: ' + str(qz_xs[0].shape))
        # print('zss: ' + str(len(zss)))
        # print('zss: ' + str(zss[0].shape))
        # print('px_zs: ' + str(len(px_zs)))
        # print('px_zs: ' + str(px_zs).shape)

        log_iws = [] # stores importance weights 
        kls = []
        lpx_zs = []
        lpxz_ind = []
        
        for encoder in range(self.M):
            #print('encoder ' + str(encoder))
            z_r = zss[encoder]
            K, B = zss[encoder].shape[:2]
            #print('B ' + str(B))
            '''
            There are four terms that comprise the ELBO loss calculation below. These four terms are the log-prior term (lpz), the MoE log-posterior term (lqz_x),
            the joint scaled log-likelihood term (lpx_z) and the log importance weights terms (log_iw) 
            '''
            # log prior term
            lpz = pz.log_prob(z_r).sum(-1) 
            #print('log prior lpz: ' + str(lpz.shape))
            # calculate the log posterior term averaged over all modalities
            lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(z_r).sum(-1) for qz_x in qz_xs]), dim=0)
            #print('log posterior averaged over modalities lqz_x: ' + str(lqz_x.shape))
            # joint scaled log likelihood term
            lpx_all = [] 
            for m in range(self.M):
                #print('Modality ' + str(m))
                log_prob = px_zs[encoder][m].log_prob(x_data[m])
                #print('log_prob ' + str(log_prob.shape))
                # scale 
                scale = self.vaes[m].like_scale
                log_prob_scaled = log_prob.reshape(K*B, -1).sum(dim=1).reshape(K, B) * scale
                #print('log prob scaled ' + str(log_prob_scaled.shape))
                #print(log_prob_scaled.shape)
                lpx_all.append(log_prob_scaled)
                lpxz_ind.append(-1*log_prob_scaled.mean().detach().cpu().numpy())
            lpx_z = torch.stack(lpx_all).sum(0)
            #print('lpx_z ' + str(lpx_z.shape))
            # log importance weight
            lw = lpx_z + ((lpz - lqz_x) * beta)
            #lw = lpz + lpx_z - lqz_x
            #print('lw: ' + str(lw.shape))
            #print(lpz.shape, lpx_z.shape, lqz_x.shape)
            log_iws.append(lw)
            lpx_zs.append(-1*lpx_z.mean().detach().cpu().numpy())
            kls.append((lqz_x - lpz).mean().detach().cpu().numpy())
        lw_all = torch.cat(log_iws, dim=0)
        #print('lw_all ' + str(lw_all.shape))
        # total ELBO is averaged log weights across m modalities 
        lw_all_reshape = lw_all.view(self.M, K, -1) # transform from (M*K, B) to (M, K, B)
        #print('lw_all_reshape ' + str(lw_all_reshape.shape))
        log_p_x = log_mean_exp(lw_all_reshape, dim=1)
        #print('log_p_x ' + str(log_p_x.shape))
        avg_log_weight = log_p_x.sum(dim=0).mean()
        return avg_log_weight, lpx_zs, kls, lpxz_ind

    # computes DReG loss (stabler version of IWAE to improve unstable/large gradients in the encoder)
    def moe_dreg_loss(self, x_data, K=1):
        #pz = self.get_pz()
        mu, scale = self.pz_params
        #pz  = dist.Laplace(mu, scale)
        pz = dist.Normal(mu, scale)
        qz_xs, zss, px_zs = self.forward(x_data, num_samples=K)
        # print('qz_xs: ' + str(len(qz_xs)))
        # print('zss: ' + str(len(zss)))
        # print('px_zs: ' + str(len(px_zs)))

        qz_xs_ = [vae.qz_x(*[dist.detach() for dist in vae.qz_x_params]) for vae in self.vaes] # add for DReG estimate 

        log_iws = [] # stores importance weights 
        lpzs = []
        lpx_zs = []
        lqz_xs = []
        
        for encoder in range(self.M):
            #print('encoder ' + str(encoder))
            z_r = zss[encoder]
            K, B = zss[encoder].shape[:2]
            #print('B ' + str(B))
            '''
            There are four terms that comprise the ELBO loss calculation below. These four terms are the log-prior term (lpz), the MoE log-posterior term (lqz_x),
            the joint scaled log-likelihood term (lpx_z) and the log importance weights terms (log_iw) 
            '''
            # log prior term
            lpz = pz.log_prob(z_r).sum(-1) 
            #print('log prior lpz: ' + str(lpz.shape))
            # calculate the log posterior term averaged over all modalities
            lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(z_r).sum(-1) for qz_x in qz_xs]), dim=0)
            #print('log posterior averaged over modalities lqz_x: ' + str(lqz_x.shape))
            # joint scaled log likelihood term
            lpx_all = [] 
            for m in range(self.M):
                #print('Modality ' + str(m))
                log_prob = px_zs[encoder][m].log_prob(x_data[m])
                #print('log_prob ' + str(log_prob.shape))
                # scale 
                scale = self.vaes[m].like_scale
                log_prob_scaled = log_prob.reshape(K*B, -1).sum(dim=1).reshape(K, B) * scale
                #print('log prob scaled ' + str(log_prob_scaled.shape))
                lpx_all.append(log_prob_scaled)
            lpx_z = torch.stack(lpx_all).sum(0)
            #print('lpx_z ' + str(lpx_z.shape))
            # log importance weight
            lw = lpz + lpx_z - lqz_x
            #print('lw: ' + str(lw.shape))
            #print(lpz.shape, lpx_z.shape, lqz_x.shape)
            log_iws.append(lw)
            lpzs.append(-1*log_mean_exp(lpz, dim=0).mean().detach().cpu().numpy())
            lpx_zs.append(-1*log_mean_exp(lpx_z, dim=0).mean().detach().cpu().numpy())
            lqz_xs.append(-1*log_mean_exp(lqz_x, dim=0).mean().detach().cpu().numpy())
        lw_all = torch.cat(log_iws, dim=0)
        #print('lw_all ' + str(lw_all.shape))
        # total ELBO is averaged log weights across m modalities 
        lw_all_reshape = lw_all.view(self.M, K, -1) # transform from (M*K, B) to (M, K, B)
        #print('lw_all_reshape ' + str(lw_all_reshape.shape))
        zss = torch.cat(zss, dim=0)
        with torch.no_grad():
            grad_wt = (lw_all_reshape - torch.logsumexp(lw_all_reshape, 0, keepdim=True)).exp()
            if zss.requires_grad: 
                zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
        return (grad_wt * lw).mean(0).sum(), lpzs, lpx_zs, lqz_xs

    def reconstruct(self, data, output_path, epoch, dataset_abbrev):
        device = next(self.parameters()).device
        x1 = data[0][0].to(device)
        x2 = data[1][0].to(device)
        x1_labels = data[0][1].to(device)
        x2_labels = data[1][1].to(device)
        x1_meta = data[0][2]
        x2_meta = data[0][2]
        input_data = [x1, x2]
        reconstruction_loss = {}
        mse_loss = {}
        for i, vae in enumerate(self.vaes):
            mu, logvar = vae.encode(input_data[i])
            qz = vae.qz_x(mu, torch.exp(logvar))
            z = qz.rsample(torch.Size([1]))
            latent_vectors = z.squeeze().detach().cpu().numpy()
            latent_space = pd.DataFrame(latent_vectors, columns=[f'LV{i+1}' for i in range(latent_vectors.shape[1])])
            if 'HC' in dataset_abbrev:
                latent_space = latent_space.assign(**data[i][2])
            else:
                latent_space['Diet'] =  data[i][2] # for DO
                latent_space['MouseID'] = data[i][3] # for DO
            latent_space['encoder'] = np.repeat(dataset_abbrev[i], latent_space.shape[0])
            latent_space.to_csv(os.path.join(output_path, f'latent_variables_epoch{epoch}_vae{dataset_abbrev[i]}.csv'), index=False)
            print(f'Encoded features from {dataset_abbrev[i]} encoder saved')
            for o, vae_out in enumerate(self.vaes): 
                mean, scale = vae_out.decode(z)
                px_z = vae_out.px_z(mean, scale)
                log_likelihood = px_z.log_prob(input_data[o]).sum(dim=-1).mean()
                reconstruction_loss[f'recon{dataset_abbrev[o]}_from_{dataset_abbrev[i]}'] = -log_likelihood.item()
                recon = mean.squeeze(0).detach().cpu().numpy()
                mse_loss[f'recon{dataset_abbrev[o]}_from_{dataset_abbrev[i]}'] = F.mse_loss(data[o][0], torch.from_numpy(recon))
                # recon_df = pd.DataFrame(recon)
                # recon_df['Diet'] = data[o][2]
                # recon_df['MouseID'] = data[o][3]
                # #recon_df = recon_df.assign(**data[o][2])
                # #recon_df.to_csv(os.path.join(output_path, f'recon{dataset_abbrev[o]}_epoch{epoch}_vae{dataset_abbrev[i]}.csv'), index=False)
                # print(f'Reconstructed {dataset_abbrev[o]} data from {dataset_abbrev[i]} encoder saved')
        mse_loss_df = pd.DataFrame([mse_loss])
        mse_loss_df.to_csv(os.path.join(output_path, f'cross_recon_mse_epoch{epoch}.csv'), index=False)
        reconstruction_loss_df = pd.DataFrame([reconstruction_loss])
        reconstruction_loss_df.to_csv(os.path.join(output_path, f'cross_recon_nloglikel_epoch{epoch}.csv'), index=False)

