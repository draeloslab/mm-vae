import torch
import torch.utils.data
from utils import*

def train(epoch, model, optimizer, train_loader, num_samples, device, epochs, model_type):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_loader):
        
        # unpack/prepare data
        x_1 = data[0][0].to(device)
        x_2 = data[1][0].to(device)
        x_data = [x_1, x_2]
        # prepare corresponding labels for inputting into the model 
        y = data[0][1].to(device)
        #y_2 = data[1][1].to(device)
        #y_data = [y_1, y_1]
        optimizer.zero_grad()

        step = int(epochs / 3)
        # if epoch < step: 
        #     beta = 0.0
        # elif epoch >= step or epoch <= (2 * step): 
        #     beta = (epoch - step) / (step)
        # else: 
        #     beta = 1.0
        beta = 1
        if model_type == 'VAE':
            evidence, lpx_zs, kls, lpxz_ind, mse_loss, mse_cfm_loss = model.moe_iwae_loss(x_data, beta, K=num_samples)
            #evidence, lqzs, lpx_zs, lqz_xs = model.moe_dreg_loss(x_data, K=num_samples)
            #loss = model._m_iwae(x_data, K=num_samples)
            #loss = model.moe_elbo_loss(x_data, K=num_samples)
            loss = -evidence
        elif model_type == 'CVAE':
            evidence, lpx_zs, kls, lpxz_ind, mse_loss, mse_cfm_loss = model.moe_iwae_cvae_loss(x_data, y, beta, K=num_samples)
            loss = -evidence
        else: 
            loss, lpx_zs, kls, lpxz_ind, mse_loss, mse_cfm_loss = model.moe_elbo_cgmvae_loss(x_data, y, beta, K=num_samples)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    #print(f'====> Epoch: {epoch:03d} Train loss: {total_loss / len(train_loader.dataset):.4f}')
    print(f'====> Epoch: {epoch:03d} Train loss: {total_loss:.4f}')

    #eturn total_loss / len(train_loader.dataset), lpx_zs, kls, lpxz_ind
    return total_loss, lpx_zs, kls, lpxz_ind, mse_loss, mse_cfm_loss

def test(epoch, model, optimizer, test_loader, num_samples, device, output_path, dataset_abbrev):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            
            optimizer.zero_grad()
            # loss = model.moe_iwae_loss(x_test_data, K=num_samples)
            # total_loss += loss.item()
            #total_loss += loss.item()
            model.reconstruct(test_data, output_path, epoch, dataset_abbrev)
            print('Finished test reconstruction')

            torch.save(model, os.path.join(output_path, f'saved_model_epoch{epoch}.pth'))
            print('Model saved')