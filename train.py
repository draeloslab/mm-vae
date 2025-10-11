import torch
import torch.utils.data
from torchvision.utils import save_image
from torchvision import datasets
from loss import*
from loss_mse import*
from utils import*
import pandas as pd
import os

def train(epoch, model, train_loader, device, optimizer, log_interval, ks_weight, cv_weight, kl_weight, gmm_centers, gmm_std, model_type, test_loader, output_folder, latent_dim):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    for batch_idx, (data, labels, df_layer1, mouse_ids) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        df_layer1 = df_layer1.to(device)
        optimizer.zero_grad()
        if model_type == 'VAE':
            recon_batch, mu, logvar, z = model(data)
            loss, recons, kld = loss_function(recon_batch, df_layer1, mu, logvar)
        elif model_type == 'GMVAE':
            recon_batch, mu, logvar, z = model(data)
            loss, recons, kld = get_gmmvaeloss(recon_batch, z, df_layer1, ks_weight, cv_weight, kl_weight, gmm_centers, gmm_std)
        elif model_type == 'CGMVAE':
            recon_batch, mu, logvar, z = model(data, labels)
            loss, recons, kld = get_gmmvaeloss(recon_batch, z, df_layer1, ks_weight, cv_weight, kl_weight, gmm_centers, gmm_std)
        loss.backward()
        train_loss += loss.item()
        recon_loss += recons.item()
        kld_loss += kld.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    
    if epoch in [200, 500, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6500, 7000, 8000, 10000, 15000, 20000]:
        try:
            with torch.no_grad():
                test_data, test_labels, df_layer1, mouse_ids = next(iter(test_loader))
                test_data = test_data.to(device)
                test_labels = test_labels.to(device)
                if model_type == 'CGMVAE':
                    recons, mu, logvar, z = model.forward(test_data, test_labels)
                else: 
                    recons, mu, logvar, z = model.forward(test_data)
                latent_vectors = z.detach().cpu().numpy()
                latent_space = pd.DataFrame(latent_vectors, columns=[f'LV{i+1}' for i in range(latent_dim)])
                latent_space['labels'] = test_labels.detach().cpu().numpy()
                latent_space['mouse_ids'] = list(mouse_ids)
                latent_space.to_csv(os.path.join(output_folder, f'latent_variables_epoch{epoch}.csv'), index=False)
                print('Latent variables saved')

                recons = pd.DataFrame(recons.detach().cpu().numpy())
                recons.to_csv(os.path.join(output_folder, f'recons_epoch{epoch}.csv'), index=False)
                print('Reconstruction features saved')

                torch.save(model, os.path.join(output_folder, f'saved_model_epoch{epoch}.pth'))
                print('Model saved')

        except Exception as e: 
            print(f'Error during saving/plotting LV at epoch {epoch}: {e}')

    train_loss /= len(train_loader.dataset)
    recon_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss))
    
    return train_loss, recon_loss, kld_loss
