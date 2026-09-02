# this file will have the model architecture for the MoE mmVAE model 
# from __future__ import print_function
import torch.distributions as dist
from VAE_model_architectures import*
from utils import*
import torch.nn.functional as F
from latent_regularizer import*

class MM_CGMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, scales, num_classes, gmm_centers, gmm_std, ks_weight, cv_weight):
        super(MM_CGMVAE, self).__init__()
        
        ## add parameter initializations in main class
        self.M = len(input_dim)
        vae_list = []
        for m in range(self.M):
            vae_list.append(Autoencoder_CGMVAE(input_dim[m], input_dim[m], hidden_dims[m], latent_dim, num_classes))
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
    
    def forward(self, x, y, cfm, lifespan):
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
            qz_x, px_z, zs = vae(x[m], y, cfm, lifespan) 
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
    def moe_elbo_cgmvae_loss(self, x_data, y_data, cfm, lifespan, layer1_data, beta, d_target, temperature):

        qz_xs, zss, px_zs = self.forward(x_data, y_data, cfm, lifespan)

        encoder_loss = 0.0
        kls = []
        lpx_zs = []
        lpxz_ind = []

        mse_loss = []
        mse_cfm_loss = []
        mse_lifespan_loss = []

        bin_means = []

        for encoder in range(self.M):
            z_r = zss[encoder]

            ks_loss = mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(z_r, self.gmm_centers[encoder], self.gmm_std)
            cv_loss = mean_squared_covariance_gmm(z_r, self.gmm_centers[encoder], self.gmm_std)
            kl_loss = self.ks_weight[encoder] * ks_loss + self.cv_weight[encoder] * cv_loss

            # track bin means for epoch 
            bin_means.append(torch.stack([z_r[y_data == label].mean(dim=0) for label in torch.unique(y_data)]).detach().cpu().numpy())
            
            recon_loss = 0.0
            for m in range(self.M):
                criterion_mse = nn.MSELoss(reduction='none')
                rec_data_loss = criterion_mse(px_zs[encoder][m].loc, layer1_data[m])
                scale = self.vaes[m].like_scale
                loss_count = rec_data_loss.mean(dim=0)[0:-2].mean()
                loss_cfm = rec_data_loss.mean(dim=0)[-2]
                loss_lifespan = rec_data_loss.mean(dim=0)[-1]
                mse_loss.append(loss_count.detach().cpu().numpy())
                mse_cfm_loss.append(loss_cfm.detach().cpu().numpy())
                mse_lifespan_loss.append(loss_lifespan.detach().cpu().numpy())

                total_recon = loss_count + loss_cfm + loss_lifespan

                recon_loss += total_recon * scale
                lpxz_ind.append(total_recon.detach().cpu().numpy() * scale)
            
            encoder_loss += recon_loss / self.M + (kl_loss * beta)
            kls.append(kl_loss.detach().cpu().numpy())
            lpx_zs.append(np.mean(recon_loss.detach().cpu().numpy()))

            _, overall_overlap, overlap_vals, overall_corr, corr_vals, sus, res = get_osd_from_latents_torch(zss, y_data.detach().cpu().numpy(), cfm, d_target, temperature, state='train')
        # return encoder_loss / self.M, lpx_zs, kls, lpxz_ind, mse_loss, mse_cfm_loss, mse_lifespan_loss, osd.detach().cpu().numpy(), osd_vals, bin_means, torch.stack([sus, res], dim=0)

        return {
                    "overall_loss": encoder_loss / self.M,
                    "lpx_zs": lpx_zs, 
                    "kls": kls, 
                    "lpxz_ind": lpxz_ind, 
                    "mse_loss": mse_loss, 
                    "mse_cfm_loss": mse_cfm_loss, 
                    "mse_lifespan_loss": mse_lifespan_loss, 
                    "overall_overlap": overall_overlap.detach().cpu().numpy(), 
                    "overlap_vals": overlap_vals,
                    "overall_corr": overall_corr.detach().cpu().numpy(), 
                    "corr_vals": corr_vals,
                    "bin_means": bin_means, 
                    "endpoints": torch.stack([sus, res], dim=0)
                }

    def moe_elbo_cgmvae_loss_train_osd(self, x_data, y_data, cfm, lifespan, layer1_data, beta, mode, osd_weight, d_target, temperature):

        qz_xs, zss, px_zs = self.forward(x_data, y_data, cfm, lifespan)

        encoder_loss = 0.0
        kls = []
        lpx_zs = []
        lpxz_ind = []

        mse_loss = []
        mse_cfm_loss = []
        mse_lifespan_loss = []

        bin_means = []
        for encoder in range(self.M):
            z_r = zss[encoder]

            ks_loss = mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(z_r, self.gmm_centers[encoder], self.gmm_std)
            cv_loss = mean_squared_covariance_gmm(z_r, self.gmm_centers[encoder], self.gmm_std)
            kl_loss = self.ks_weight[encoder] * ks_loss + self.cv_weight[encoder] * cv_loss

            # track bin means for epoch 
            bin_means.append(torch.stack([z_r[y_data == label].mean(dim=0) for label in torch.unique(y_data)]).detach().cpu().numpy())
            
            recon_loss = 0.0
            for m in range(self.M):
                criterion_mse = nn.MSELoss(reduction='none')
                rec_data_loss = criterion_mse(px_zs[encoder][m].loc, layer1_data[m])
                scale = self.vaes[m].like_scale
                loss_count = rec_data_loss.mean(dim=0)[0:-2].mean()
                loss_cfm = rec_data_loss.mean(dim=0)[-2]
                loss_lifespan = rec_data_loss.mean(dim=0)[-1]
                mse_loss.append(loss_count.detach().cpu().numpy())
                mse_cfm_loss.append(loss_cfm.detach().cpu().numpy())
                mse_lifespan_loss.append(loss_lifespan.detach().cpu().numpy())

                total_recon = loss_count + loss_cfm + loss_lifespan
                recon_loss += total_recon * scale
                lpxz_ind.append(total_recon.detach().cpu().numpy() * scale)

            encoder_loss += recon_loss / self.M + (kl_loss * beta)
            kls.append(kl_loss.detach().cpu().numpy())
            lpx_zs.append(np.mean(recon_loss.detach().cpu().numpy()))
            _, overall_overlap, overlap_vals, overall_corr, corr_vals, sus, res = get_osd_from_latents_torch_new(zss, y_data.detach().cpu().numpy(), cfm.detach().cpu().numpy(), d_target, temperature, state='train')
            
        return {
            "overall_loss": encoder_loss / self.M + overall_overlap * 1000 + overall_corr * 1000,
            #"overall_loss": encoder_loss / self.M,
            "lpx_zs": lpx_zs, 
            "kls": kls, 
            "lpxz_ind": lpxz_ind, 
            "mse_loss": mse_loss, 
            "mse_cfm_loss": mse_cfm_loss, 
            "mse_lifespan_loss": mse_lifespan_loss, 
            "overall_overlap": overall_overlap, 
            "overlap_vals": overlap_vals,
            "overall_corr": overall_corr, 
            "corr_vals": corr_vals,
            "bin_means": bin_means, 
            "endpoints": torch.stack([sus, res], dim=0)
        }
    
    def reconstruct(self, data, output_path, epoch, dataset_abbrev, physio_cols, mode, d_target, temperature):
        device = next(self.parameters()).device
        df1_layer1 = data[0][0].to(device)
        df2_layer1 = data[1][0].to(device)
        x1 = data[0][1].to(device)
        x2 = data[1][1].to(device)
        input_data = [x1, x2]

        y = data[0][2].to(device)
        cfm = data[0][3].to(device)
        input_layers = [df1_layer1, df2_layer1]
        lifespan = data[0][4].to(device)
        reconstruction_loss = {}
        mse_loss = {}
        mse_cfm_loss = {}
        zss = []
        encoder_col = []
        for i, vae in enumerate(self.vaes):
            mu, std, z, y, cfm = vae.encode(input_data[i], y, cfm, lifespan)
            zss.append(z)
            encoder_col.append(np.repeat(dataset_abbrev[i], z.shape[0]))
            latent_vectors = z.detach().cpu().numpy()
            latent_space = pd.DataFrame(latent_vectors, columns=[f'LV{i+1}' for i in range(latent_vectors.shape[1])])
            latent_space['Label'] = data[i][2] # for DO 
            latent_space['Diet'] =  data[i][6] # for DO
            latent_space['MouseID'] = data[i][5] # for DO
            latent_space['Bin'] = y.detach().cpu().numpy()
            latent_space['CFM'] = cfm.detach().cpu().numpy()
            latent_space['Lifespan'] = lifespan.detach().cpu().numpy()
            latent_space['Encoder'] = np.repeat(dataset_abbrev[i], latent_space.shape[0])
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
                recon = mean.detach().cpu().numpy()
                recon_df = pd.DataFrame(recon)
                if dataset_abbrev[o] == 'Physio':
                    recon_df.columns = physio_cols
                else: 
                    recon_df = recon_df.rename(columns={recon_df.columns[-2]: 'cfm'})
                    recon_df = recon_df.rename(columns={recon_df.columns[-1]: 'lifespan'})
                recon_df['orig_lifespan'] = lifespan.detach().cpu().numpy()
                recon_df['bin'] = y.detach().cpu().numpy()
                recon_df['orig_cfm'] = cfm.detach().cpu().numpy()
                recon_df.to_csv(os.path.join(output_path, f'recon{dataset_abbrev[o]}_epoch{epoch}_vae{dataset_abbrev[i]}.csv'), index=False)
                print(f'Reconstructed {dataset_abbrev[o]} data from {dataset_abbrev[i]} encoder saved')
        proj, overall_overlap, overlap_vals, overall_corr, corr_vals, sus, res = get_osd_from_latents_torch_new(zss, y.detach().cpu().numpy(), cfm.squeeze().detach().cpu().numpy(), d_target, temperature)
        if mode == 'single':
            geno_proj = pd.DataFrame(proj[0].detach().cpu().numpy(), columns = ['value', 'bin'])
            geno_proj['encoder'] = np.repeat('Geno', len(geno_proj))
            geno_proj['overlap'] = np.repeat(overlap_vals[0].detach().cpu().numpy(), len(geno_proj))
            geno_proj['corr'] = np.repeat(corr_vals[0].detach().cpu().numpy(), len(geno_proj))
            geno_proj.to_csv(os.path.join(output_path, f'Geno_phenotypic_projection_epoch{epoch}.csv'), index=False)
            physio_proj = pd.DataFrame(proj[1].detach().cpu().numpy(), columns = ['value', 'bin'])
            physio_proj['encoder'] = np.repeat('Physio', len(physio_proj))
            physio_proj['overlap'] = np.repeat(overlap_vals[1].detach().cpu().numpy(), len(physio_proj))
            physio_proj['corr'] = np.repeat(corr_vals[1].detach().cpu().numpy(), len(physio_proj))
            physio_proj.to_csv(os.path.join(output_path, f'Physio_phenotypic_projection_epoch{epoch}.csv'), index=False)
            all_proj = pd.DataFrame(proj[2].detach().cpu().numpy(), columns = ['value', 'bin'])
            all_proj['encoder'] = np.concatenate(encoder_col, axis=0)
            all_proj['overlap'] = np.repeat(overlap_vals[2].detach().cpu().numpy(), all_proj.shape[0])
            all_proj['corr'] = np.repeat(corr_vals[2].detach().cpu().numpy(), all_proj.shape[0])
            all_proj.to_csv(os.path.join(output_path, f'all_phenotypic_projection_epoch{epoch}.csv'), index=False)
        # elif mode == 'together':
        #     geno_proj = pd.DataFrame(proj[:len(y)], columns = ['value', 'bin'])
        #     geno_proj['encoder'] = np.repeat('Geno', len(geno_proj))
        #     geno_proj['osd'] = np.repeat(osd_vals[0], len(geno_proj))
        #     geno_proj.to_csv(os.path.join(output_path, f'Geno_phenotypic_projection_epoch{epoch}.csv'), index=False)
        #     physio_proj = pd.DataFrame(proj[len(y):], columns = ['value', 'bin'])
        #     physio_proj['encoder'] = np.repeat('Physio', len(physio_proj))
        #     physio_proj['osd'] = np.repeat(osd_vals[1], len(physio_proj))
        #     physio_proj.to_csv(os.path.join(output_path, f'Physio_phenotypic_projection_epoch{epoch}.csv'), index=False)
        #     all_proj = pd.DataFrame(proj, columns = ['value', 'bin'])
        #     all_proj['encoder'] = np.concatenate(encoder_col, axis=0)
        #     all_proj['osd'] = np.repeat(osd_vals[2], all_proj.shape[0])
        #     all_proj.to_csv(os.path.join(output_path, f'all_phenotypic_projection_epoch{epoch}.csv'), index=False)
        # osd_across_proj = []
        # for i, encoder in enumerate(dataset_abbrev):
        #     half = int(all_proj.shape[0] / 2)
        #     if i == 0:
        #         osd = calc_osd_diff(proj[:half, 0], proj[:half, 1])
        #     else: 
        #         osd = calc_osd_diff(proj[half:, 0], proj[half:, 1])
        #     osd_across_proj.append(np.repeat(osd.detach().cpu().numpy(), half))
        # all_proj['encoder_osd'] = np.concatenate(osd_across_proj, axis=0)
        # all_proj.to_csv(os.path.join(output_path, f'phenotypic_projection_epoch{epoch}.csv'), index=False)
        mse_series = pd.Series(mse_loss, name='mse_loss')
        mse_cfm_series = pd.Series(mse_cfm_loss, name='mse_cfm_loss')
        llik_series = pd.Series(reconstruction_loss, name='loglik_loss')
        recon_eval_df = pd.concat([mse_series, mse_cfm_series, llik_series], axis=1)
        recon_eval_df.index.name = 'encoder/decoder'
        recon_eval_df = recon_eval_df.reset_index()
        recon_eval_df.to_csv(os.path.join(output_path, f'reconstruction_metrics_epoch{epoch}.csv'), index=False)

    def get_latent_space(self, data, dataset_abbrev):
        device = next(self.parameters()).device
        df1_layer1 = data[0][0].to(device)
        df2_layer1 = data[1][0].to(device)
        x1 = data[0][1].to(device)
        x2 = data[1][1].to(device)
        pair_id = np.arange(1, x1.shape[0] + 1)
        y = data[0][2].to(device)
        cfm = data[0][3].to(device)
        lifespan = data[0][4].to(device)
        mouse_id = data[0][5].to(device)
        diet = data[0][6].to(device)
        input_data = [x1, x2]
        input_layers = [df1_layer1, df2_layer1]
        full_latent_space = []
        for i, vae in enumerate(self.vaes):
            mu, std, z, y, cfm = vae.encode(input_data[i], y, cfm)
            latent_vectors = z.detach().cpu().numpy()
            latent_space = pd.DataFrame(latent_vectors, columns=[f'LV{i+1}' for i in range(latent_vectors.shape[1])])
            latent_space['bin'] = y.detach().cpu().numpy()
            latent_space['cfm'] = cfm.detach().cpu().numpy()
            latent_space['lifespan'] = lifespan.detach().cpu().numpy()
            latent_space['encoder'] = np.repeat(dataset_abbrev[i], latent_space.shape[0])
            latent_space['pair'] = pair_id
            latent_space['mouse_id'] = mouse_id
            latent_space['diet'] = diet
            full_latent_space.append(latent_space)
        return pd.concat(full_latent_space, ignore_index=True)

