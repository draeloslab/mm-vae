# this file will have the model architecture for the MoE mmVAE model 
# from __future__ import print_function
import torch.distributions as dist
from cvae import*
from utils import*
import torch.nn.functional as F

class MM_CVAE(nn.Module):
    def __init__(self, input_dim1, input_dim2, H1_1, H2_1, H3_1, H1_2, H2_2, H3_2, latent_dim, scale1, scale2, num_classes, learn_prior: bool = False):
        super(MM_CVAE, self).__init__()
        
        ## add parameter initializations in main class
        self.vae1 = CVAE(input_dim1, H1_1, H2_1, H3_1, latent_dim, scale1, num_classes)
        self.vae2 = CVAE(input_dim2, H1_2, H2_2, H3_2, latent_dim, scale2, num_classes)
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
    
    def forward(self, x, y, num_samples):
        # forward pass through joint encoder distributions 
        # qz_xs stores posteriors qz_x for each modality 
        qz_xs = []
        # zss stores the latent samples taken from each modality (num samples x batch size)
        zss = []
        ys = []
        # px_zs computes the cross-modal likelihood matrix. 
        # ex. cross modal elements are computed by taking samples from one encoder and using the other modalities decoder
        px_zs = [[None for num in range(self.M)] for num in range(self.M)]

        for m, vae in enumerate(self.vaes):
            # qz_x is the posterior distribution 
            # px_z is the self-reconstruction likelihood (diagonal of px_zs)
            # zs is the latent samples 
            #print(y[m].shape)
            qz_x, px_z, zs, y = vae(x[m], y, num_samples) # num samples is relevant for calculating IWAE loss 
            qz_xs.append(qz_x)
            zss.append(zs)
            ys.append(y)
            px_zs[m][m] = px_z
        
        # cross-modal decoding 
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:
                    mean, scale = vae.decode(zs, y)
                    px_zs[e][d] = vae.px_z(mean, scale)
        
        return qz_xs, zss, px_zs
    
    # computes IWAE loss (very similar to ELBO but with K>1 samples)
    def moe_iwae_cvae_loss(self, x_data, y_data, beta, K=1):
        #pz = self.get_pz()
        mu, scale = self.pz_params
        #pz  = dist.Laplace(mu, scale)
        pz = dist.Normal(mu, scale)
        qz_xs, zss, px_zs = self.forward(x_data, y_data, num_samples=K)
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
                #print(px_zs[encoder][m].device)
                log_prob = px_zs[encoder][m].log_prob(x_data[m])
                #print('log_prob ' + str(log_prob.shape))
                # scale 
                scale = self.vaes[m].like_scale
                log_prob_scaled = log_prob.reshape(K*B, -1).sum(dim=1).reshape(K, B) * scale
                #print('log prob scaled ' + str(log_prob_scaled.shape))
                #print(log_prob_scaled.shape)
                lpx_all.append(log_prob_scaled)
                #lpxz_ind.append(-1*log_prob_scaled.mean().detach().cpu().numpy())
                lpxz_ind.append(log_prob_scaled)
            lpx_z = torch.stack(lpx_all).sum(0)
            #print('lpx_z ' + str(lpx_z.shape))
            # log importance weight
            lw = lpx_z + ((lpz - lqz_x) * beta)
            #lw = lpz + lpx_z - lqz_x
            #print('lw: ' + str(lw.shape))
            log_iws.append(lw)
            w = torch.softmax(lw, dim=0)
            if encoder == 0: 
                for place in [0, 1]:
                    lpxz_ind[place] = ((-1 * w * lpxz_ind[place]).sum(0).sum()).detach().cpu().numpy()
            else:
                for place in [2, 3]:
                    lpxz_ind[place] = ((-1 * w * lpxz_ind[place]).sum(0).sum()).detach().cpu().numpy()
            lpx_zs.append(((-1 * w * lpx_z).sum(0).sum()).detach().cpu().numpy())
            #print(lpx_z.shape, (lqz_x - lpz).shape)
            kls.append(((w * (lqz_x - lpz)).sum(0).sum()).detach().cpu().numpy())
        lw_all = torch.cat(log_iws, dim=0)
        #print('lw_all ' + str(lw_all.shape))
        # total ELBO is averaged log weights across m modalities 
        lw_all_reshape = lw_all.view(self.M, K, -1) # transform from (M*K, B) to (M, K, B)
        #print('lw_all_reshape ' + str(lw_all_reshape.shape))
        log_p_x = log_mean_exp(lw_all_reshape, dim=1)

        #print('log_p_x ' + str(log_p_x.shape))
        #avg_log_weight = log_p_x.sum(dim=0).mean()
        # CHANGING, because I think this may have been wrong 
        avg_log_weight = log_p_x.sum(dim=1).mean(dim=0)

        return avg_log_weight, lpx_zs, kls, lpxz_ind

    def reconstruct(self, data, output_path, epoch, dataset_abbrev):
        device = next(self.parameters()).device
        x1 = data[0][0].to(device)
        x2 = data[1][0].to(device)
        y = data[1][1].to(device)
        input_data = [x1, x2]
        reconstruction_loss = {}
        mse_loss = {}
        for i, vae in enumerate(self.vaes):
            mu, logvar = vae.encode(input_data[i], y)
            qz = vae.qz_x(mu, torch.exp(logvar))
            z = qz.rsample(torch.Size([1]))
            latent_vectors = z.squeeze().detach().cpu().numpy()
            latent_space = pd.DataFrame(latent_vectors, columns=[f'LV{i+1}' for i in range(latent_vectors.shape[1])])
            if 'HC' in dataset_abbrev:
                latent_space = latent_space.assign(**data[i][2])
            else:
                latent_space['Label'] = data[i][1] # for DO 
                latent_space['Diet'] =  data[i][2] # for DO
                latent_space['MouseID'] = data[i][3] # for DO
            latent_space['encoder'] = np.repeat(dataset_abbrev[i], latent_space.shape[0])
            latent_space.to_csv(os.path.join(output_path, f'latent_variables_epoch{epoch}_vae{dataset_abbrev[i]}.csv'), index=False)
            print(f'Encoded features from {dataset_abbrev[i]} encoder saved')
            for o, vae_out in enumerate(self.vaes): 
                mean, scale = vae_out.decode(z, y)
                px_z = vae_out.px_z(mean, scale)
                log_likelihood = px_z.log_prob(input_data[o]).sum(dim=-1).mean()
                reconstruction_loss[f'recon{dataset_abbrev[o]}_from_{dataset_abbrev[i]}'] = -log_likelihood.item()
                recon = mean.squeeze(0).detach().cpu().numpy()
                mse_loss[f'recon{dataset_abbrev[o]}_from_{dataset_abbrev[i]}'] = F.mse_loss(data[o][0], torch.from_numpy(recon))
                recon_df = pd.DataFrame(recon)
                # recon_df['Diet'] = data[o][2]
                # recon_df['MouseID'] = data[o][3]
                recon_df = recon_df.assign(**data[o][2])
                recon_df.to_csv(os.path.join(output_path, f'recon{dataset_abbrev[o]}_epoch{epoch}_vae{dataset_abbrev[i]}.csv'), index=False)
                print(f'Reconstructed {dataset_abbrev[o]} data from {dataset_abbrev[i]} encoder saved')
        mse_loss_df = pd.DataFrame([mse_loss])
        mse_loss_df.to_csv(os.path.join(output_path, f'cross_recon_mse_epoch{epoch}.csv'), index=False)
        reconstruction_loss_df = pd.DataFrame([reconstruction_loss])
        reconstruction_loss_df.to_csv(os.path.join(output_path, f'cross_recon_nloglikel_epoch{epoch}.csv'), index=False)

