# this file will have the training process for the Vanilla VAE

#from __future__ import print_function
import torch
import torch.utils.data
import pandas as pd
import os
from loss import*

def train(epoch, model, train_loader, device, optimizer, log_interval, test_loader, latent_dim, output_folder):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    for batch_idx, (data, fly_id, genotype) in enumerate(train_loader):
        data = data.to(device)
        #fly_id = fly_id.to(device)
        #genotype = genotype.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        loss, recons, kld = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        # recon_loss += recons.item() * len(data)
        # kld_loss += kld.item() * len(data)
        recon_loss += recons.item()
        kld_loss += kld.item()
        optimizer.step()
        #if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    train_loss /= len(train_loader.dataset)
    recon_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)

    if epoch in [10, 200, 500, 1000, 2000, 3000, 4000, 5000]:
        try: 
            with torch.no_grad():
                test_data, test_flyids, test_genotype = next(iter(test_loader))
                test_data = test_data.to(device)
                #test_flyids = test_flyids.to(device)
                #genotype = genotype.to(device)
                recons, mu, logvar, z = model.forward(test_data)
                latent_vectors = z.detach().cpu().numpy()
                latent_space = pd.DataFrame(latent_vectors, columns=[f'LV{i+1}' for i in range(latent_dim)])
                latent_space['flyID'] = list(test_flyids)
                latent_space['genotype'] = list(test_genotype)
                latent_space.to_csv(os.path.join(output_folder, f'latent_variables_epoch{epoch}.csv'), index=False)
                print('Latent variables saved')

                mu = pd.DataFrame(mu.detach().cpu().numpy())
                mu.to_csv(os.path.join(output_folder, f'mu_epoch{epoch}.csv'), index=False)

                recons = pd.DataFrame(recons.detach().cpu().numpy())
                recons.to_csv(os.path.join(output_folder, f'recons_epoch{epoch}.csv'), index=False)

                torch.save(model, os.path.join(output_folder, f'saved_model_epoch{epoch}.pth'))
                print('Model saved')

        except Exception as e: 
            print(f'Error during saving/plotting LV at epoch {epoch}: {e}')


    return train_loss, recon_loss, kld_loss

# def test(epoch, model, test_loader, device, batch_size, output_folder):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             # if i == 0:
#             #     n = min(data.size(0), 8)
#             #     comparison = torch.cat([data[:n],
#             #                           recon_batch.view(batch_size, 1, 28, 28)[:n]])
#             #     save_image(comparison.cpu(),
#             #              output_folder + '/reconstruction_' + str(epoch) + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))