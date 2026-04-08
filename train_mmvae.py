import torch
import torch.utils.data
from utils import*

def train(epoch, model, optimizer, train_loader, num_samples, device, epochs, model_type):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_loader):
        
        # unpack/prepare data
        df1_layer1 = data[0][0].to(device)

        df2_layer1 = data[1][0].to(device)
        layer1_data = [df1_layer1, df2_layer1]
        x_1 = data[0][1].to(device)
        x_2 = data[1][1].to(device)
        x_data = [x_1, x_2]
        y = data[0][2].to(device)
        cfm = data[0][3].to(device)

        optimizer.zero_grad()

        if model_type == 'CGMVAE':
            loss, lpx_zs, kls, lpxz_ind, mse_loss, mse_cfm_loss = model.moe_elbo_cgmvae_loss(x_data, y, cfm, layer1_data)
        
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    
    #print(f'====> Epoch: {epoch:03d} Train loss: {total_loss / len(train_loader.dataset):.4f}')
    print(f'====> Epoch: {epoch:03d} Train loss: {total_loss:.4f}')

    #eturn total_loss / len(train_loader.dataset), lpx_zs, kls, lpxz_ind
    return total_loss, lpx_zs, kls, lpxz_ind, mse_loss, mse_cfm_loss

def test(epoch, model, optimizer, test_loader, num_samples, device, output_path, dataset_abbrev, meta):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            
            optimizer.zero_grad()
            # loss = model.moe_iwae_loss(x_test_data, K=num_samples)
            # total_loss += loss.item()
            #total_loss += loss.item()
            model.reconstruct(test_data, output_path, epoch, dataset_abbrev, meta)
            print('Finished test reconstruction')

            torch.save(model, os.path.join(output_path, f'saved_model_epoch{epoch}.pth'))
            print('Model saved')