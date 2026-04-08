# this file will have the model architecture for the MoE mmVAE model 
# from __future__ import print_function
import torch.distributions as dist
from VAE_model_architectures import*
from utils import*
import torch.nn.functional as F
from latent_regularizer import*

class MM_CGMVAE(nn.Module):
    def __init__(self, input_dim, H1, H2, H3, latent_dim, scales, num_classes, gmm_centers, gmm_std, ks_weight, cv_weight):
        super(MM_CGMVAE, self).__init__()
        
        ## add parameter initializations in main class
        self.M = len(input_dim)
        vae_list = []
        for m in range(self.M):
            vae_list.append(Autoencoder_CGMVAE(input_dim[m], input_dim[m], H1[m], H2[m], H3[m], latent_dim, num_classes))
        self.vaes = nn.ModuleList(vae_list)

        for i, vae in enumerate(self.vaes):
            vae.like_scale = scales[i]

        # self.vae1 = Autoencoder_CGMVAE(input_dim1, input_dim1, H1_1, H2_1, H3_1, latent_dim, num_classes)
        # self.vae2 = Autoencoder_CGMVAE(input_dim2, input_dim2, H1_2, H2_2, H3_2, latent_dim, num_classes)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        #self.vaes = nn.ModuleList([self.vae1, self.vae2])
        self.scaling = scales
        self.latent_dim = latent_dim

        # gaussian mixture parameters 
        self.gmm_centers = gmm_centers
        self.gmm_std = gmm_std
        self.ks_weight = ks_weight
        self.cv_weight = cv_weight 

        # # add scaling factors for calculating combined loss across the two modalities 
        # self.vae1.like_scale = scale1
        # self.vae2.like_scale = scale2
    
    def forward(self, x, y, cfm):
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
            qz_x, px_z, zs = vae(x[m], y, cfm) 
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
    def moe_elbo_cgmvae_loss(self, x_data, y_data, cfm, layer1_data):

        qz_xs, zss, px_zs = self.forward(x_data, y_data, cfm)

        encoder_loss = 0.0
        kls = []
        lpx_zs = []
        lpxz_ind = []

        mse_loss = []
        mse_cfm_loss = []

        for encoder in range(self.M):
            z_r = zss[encoder]

            ks_loss = mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(z_r, self.gmm_centers, self.gmm_std)
            cv_loss = mean_squared_covariance_gmm(z_r, self.gmm_centers, self.gmm_std)
            kl_loss = self.ks_weight * ks_loss + self.cv_weight * cv_loss
            
            recon_loss = 0.0
            for m in range(self.M):
                criterion_mse = nn.MSELoss(reduction='none')
                rec_data_loss = criterion_mse(px_zs[encoder][m].loc, layer1_data[m])
                scale = self.vaes[m].like_scale
                loss_count = rec_data_loss.mean(dim=0)[0:-1].mean()
                loss_cfm = rec_data_loss.mean(dim=0)[-1]
                mse_loss.append(loss_count.detach().cpu().numpy())
                mse_cfm_loss.append(loss_cfm.detach().cpu().numpy())

                total_recon = loss_count + loss_cfm
                recon_loss += total_recon * scale
                lpxz_ind.append(total_recon.detach().cpu().numpy() * scale)
            
            encoder_loss += recon_loss / self.M + kl_loss
            kls.append(kl_loss.detach().cpu().numpy())
            lpx_zs.append(np.mean(recon_loss.detach().cpu().numpy()))
        
        return encoder_loss / self.M, lpx_zs, kls, lpxz_ind, mse_loss, mse_cfm_loss

    def reconstruct(self, data, output_path, epoch, dataset_abbrev, meta):
        device = next(self.parameters()).device
        df1_layer1 = data[0][0].to(device)
        df2_layer1 = data[1][0].to(device)
        x1 = data[0][1].to(device)
        x2 = data[1][1].to(device)
        y = data[0][2].to(device)
        cfm = data[0][3].to(device)
        input_data = [x1, x2]
        input_layers = [df1_layer1, df2_layer1]
        reconstruction_loss = {}
        mse_loss = {}
        mse_cfm_loss = {}
        for i, vae in enumerate(self.vaes):
            mu, std, z, y, cfm = vae.encode(input_data[i], y, cfm)
            latent_vectors = z.detach().cpu().numpy()
            latent_space = pd.DataFrame(latent_vectors, columns=[f'LV{i+1}' for i in range(latent_vectors.shape[1])])
            # else:
            #     latent_space['Label'] = data[i][1] # for DO 
            #     latent_space['Diet'] =  data[i][2] # for DO
            #     latent_space['MouseID'] = data[i][3] # for DO
            latent_space['bin'] = y.detach().cpu().numpy()
            latent_space['cfm'] = cfm.detach().cpu().numpy()
            latent_space['encoder'] = np.repeat(dataset_abbrev[i], latent_space.shape[0])
            for col_name, values in meta.items():
                latent_space[col_name] = values
            latent_space.to_csv(os.path.join(output_path, f'latent_variables_epoch{epoch}_vae{dataset_abbrev[i]}.csv'), index=False)
            print(f'Encoded features from {dataset_abbrev[i]} encoder saved')
            for o, vae_out in enumerate(self.vaes): 
                mean, scale = vae_out.decode(z, y)
                px_z = vae_out.px_z(mean, scale)
                log_likelihood = px_z.log_prob(input_layers[o]).sum(dim=-1).mean()
                reconstruction_loss[f'recon{dataset_abbrev[o]}_from_{dataset_abbrev[i]}'] = -log_likelihood.item()
                criterion_mse = nn.MSELoss(reduction='none')
                overall_mse = criterion_mse(mean, input_layers[o])

                mse_loss[f'recon{dataset_abbrev[o]}_from_{dataset_abbrev[i]}'] = overall_mse.mean(dim=0)[0:-1].mean().detach().cpu().numpy()
                mse_cfm_loss[f'recon{dataset_abbrev[o]}_from_{dataset_abbrev[i]}'] = overall_mse.mean(dim=0)[-1].mean().detach().cpu().numpy()

                # recon = mean.detach().cpu().numpy()
                # recon_df = pd.DataFrame(recon)
                # recon_df['bin'] = y.detach().cpu().numpy()
                # recon_df['cfm'] = cfm.detach().cpu().numpy()
                # recon_df.to_csv(os.path.join(output_path, f'recon{dataset_abbrev[o]}_epoch{epoch}_vae{dataset_abbrev[i]}.csv'), index=False)
                # print(f'Reconstructed {dataset_abbrev[o]} data from {dataset_abbrev[i]} encoder saved')
        mse_series = pd.Series(mse_loss, name='mse_loss')
        mse_cfm_series = pd.Series(mse_cfm_loss, name='mse_cfm_loss')
        llik_series = pd.Series(reconstruction_loss, name='loglik_loss')
        recon_eval_df = pd.concat([mse_series, mse_cfm_series, llik_series], axis=1)
        recon_eval_df.index.name = 'encoder/decoder'
        recon_eval_df = recon_eval_df.reset_index()
        recon_eval_df.to_csv(os.path.join(output_path, f'reconstruction_metrics_epoch{epoch}.csv'), index=False)

    def get_latent_space(self, data, dataset_abbrev, meta):
        device = next(self.parameters()).device
        df1_layer1 = data[0][0].to(device)
        df2_layer1 = data[1][0].to(device)
        x1 = data[0][1].to(device)
        x2 = data[1][1].to(device)
        y = data[0][2].to(device)
        cfm = data[0][3].to(device)
        input_data = [x1, x2]
        input_layers = [df1_layer1, df2_layer1]
        full_latent_space = []
        for i, vae in enumerate(self.vaes):
            mu, std, z, y, cfm = vae.encode(input_data[i], y, cfm)
            latent_vectors = z.detach().cpu().numpy()
            latent_space = pd.DataFrame(latent_vectors, columns=[f'LV{i+1}' for i in range(latent_vectors.shape[1])])
            latent_space['bin'] = y.detach().cpu().numpy()
            latent_space['cfm'] = cfm.detach().cpu().numpy()
            latent_space['encoder'] = np.repeat(dataset_abbrev[i], latent_space.shape[0])
            for col_name, values in meta.items():
                latent_space[col_name] = values
            full_latent_space.append(latent_space)
        return pd.concat(full_latent_space, ignore_index=True)

