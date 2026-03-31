# this file will have the model architecture for the MoE mmVAE model 
# from __future__ import print_function
import torch.distributions as dist
from cgmvae import*
from utils import*
import torch.nn.functional as F
from latent_regularizer import*

class MM_CGMVAE(nn.Module):
    def __init__(self, input_dim1, input_dim2, H1_1, H2_1, H3_1, H1_2, H2_2, H3_2, latent_dim, scale1, scale2, num_classes, gmm_centers, gmm_std, ks_weight, cv_weight, learn_prior: bool = False):
        super(MM_CGMVAE, self).__init__()
        
        ## add parameter initializations in main class
        self.vae1 = CGMVAE(input_dim1, H1_1, H2_1, H3_1, latent_dim, scale1, num_classes)
        self.vae2 = CGMVAE(input_dim2, H1_2, H2_2, H3_2, latent_dim, scale2, num_classes)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.vaes = nn.ModuleList([self.vae1, self.vae2])
        self.scaling = [scale1, scale2]
        self.M = len(self.vaes)
        self.latent_dim = latent_dim

        # gaussian mixture parameters 
        self.gmm_centers = gmm_centers
        self.gmm_std = gmm_std
        self.ks_weight = ks_weight
        self.cv_weight = cv_weight 

        # add scaling factors for calculating combined loss across the two modalities 
        self.vae1.like_scale = scale1
        self.vae2.like_scale = scale2

        # initializing learned prior
        # dictionary object to determine if prior will be learned or not.
        grad = {'requires_grad': learn_prior}

    
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
    
    #def moe_elbo_cgmvae_loss(self, x_data, y_data, beta, gmm_centers, gmm_std, ks_weight, cv_weight, K=1):
    def moe_elbo_cgmvae_loss(self, x_data, y_data, beta, K=1):

        qz_xs, zss, px_zs = self.forward(x_data, y_data, num_samples=K)

        encoder_loss = 0.0
        kls = []
        lpx_zs = []
        lpxz_ind = []

        mse_loss = []
        mse_cfm_loss = []

        for encoder in range(self.M):
            z_r = zss[encoder]
            K, B = zss[encoder].shape[:2]

            z_r = z_r.squeeze()
            ks_loss = mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(z_r, self.gmm_centers, self.gmm_std)
            cv_loss = mean_squared_covariance_gmm(z_r, self.gmm_centers, self.gmm_std)
            kl_loss = self.ks_weight * ks_loss + self.cv_weight * cv_loss
            
            recon_loss = 0.0
            for m in range(self.M):
                criterion_mse = nn.MSELoss(reduction='none')
                rec_mod_loss = criterion_mse(px_zs[encoder][m].loc.squeeze()[:, :-1], x_data[m][:, :-1])
                scale = self.vaes[m].like_scale
                recon_loss += rec_mod_loss.mean() * scale
                lpxz_ind.append(rec_mod_loss.mean().detach().cpu().numpy() * scale)
                criterion_mean = nn.MSELoss(reduction='none')
                mse = criterion_mean(px_zs[encoder][m].loc.squeeze()[:, :-1], x_data[m][:, :-1])
                mse_cfm = criterion_mean(px_zs[encoder][m].loc.squeeze()[:, -1], x_data[m][:, -1])
                mse_loss.append(mse.mean().detach().cpu().numpy())
                mse_cfm_loss.append(mse_cfm.mean().detach().cpu().numpy())
            
            encoder_loss += recon_loss / self.M + beta * kl_loss
            kls.append(beta * kl_loss.detach().cpu().numpy())
            lpx_zs.append(np.mean(recon_loss.detach().cpu().numpy()))
        
        return encoder_loss / self.M, lpx_zs, kls, lpxz_ind, mse_loss, mse_cfm_loss

    def reconstruct(self, data, output_path, epoch, dataset_abbrev):
        device = next(self.parameters()).device
        x1 = data[0][0].to(device)
        x2 = data[1][0].to(device)
        y = data[1][1].to(device)
        input_data = [x1, x2]
        reconstruction_loss = {}
        mse_loss = {}
        mse_cfm_loss = {}
        for i, vae in enumerate(self.vaes):
            mu, std = vae.encode(input_data[i], y)
            mu, std, z = vae.gaussian_sampler(mu.view(-1, self.num_classes, self.latent_dim), std.view(-1, self.num_classes, self.latent_dim), y)
            latent_vectors = z.detach().cpu().numpy()
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
                mse_loss[f'recon{dataset_abbrev[o]}_from_{dataset_abbrev[i]}'] = F.mse_loss(data[o][0][:, :-1], torch.from_numpy(recon)[:, :-1])
                mse_cfm_loss[f'recon{dataset_abbrev[o]}_from_{dataset_abbrev[i]}'] = F.mse_loss(data[o][0][:, -1], torch.from_numpy(recon)[:, -1])
                recon_df = pd.DataFrame(recon)
                # recon_df['Diet'] = data[o][2]
                # recon_df['MouseID'] = data[o][3]
                recon_df = recon_df.assign(**data[o][2])
                recon_df.to_csv(os.path.join(output_path, f'recon{dataset_abbrev[o]}_epoch{epoch}_vae{dataset_abbrev[i]}.csv'), index=False)
                print(f'Reconstructed {dataset_abbrev[o]} data from {dataset_abbrev[i]} encoder saved')
        mse_loss_df = pd.DataFrame([mse_loss])
        mse_loss_df.to_csv(os.path.join(output_path, f'cross_recon_mse_epoch{epoch}.csv'), index=False)
        mse_cfm_loss_df = pd.DataFrame([mse_cfm_loss])
        mse_cfm_loss_df.to_csv(os.path.join(output_path, f'cross_recon_cfm_mse_epoch{epoch}.csv'), index=False)
        reconstruction_loss_df = pd.DataFrame([reconstruction_loss])
        reconstruction_loss_df.to_csv(os.path.join(output_path, f'cross_recon_nloglikel_epoch{epoch}.csv'), index=False)

