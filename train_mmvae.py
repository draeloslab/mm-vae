import torch
import torch.utils.data
from utils import*

def train(epoch, model, optimizer, train_loader, num_samples, device, epochs, model_type, mode, d_target, temperature, train_osd=False, kl_annealing=False):
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
        lifespan = data[0][4].to(device)

        if kl_annealing:
            beta = min(1.0, epoch/int(epochs * 0.1))
        else: 
            beta = 1.0

        # if epoch < 3500: 
        #     osd_weight = 100
        # else: 
        #     osd_weight = 0
        osd_weight = 1000

        optimizer.zero_grad()

        osd_enc = None
        if model_type == 'CGMVAE':
            if train_osd:
                #loss, lpx_zs, kls, lpxz_ind, mse_loss, mse_cfm_loss, mse_lifespan_loss, osd_enc, osd_vals, bin_means, endpoints = model.moe_elbo_cgmvae_loss_train_osd(x_data, y, cfm, lifespan, layer1_data, beta, mode, osd_weight)
                output = model.moe_elbo_cgmvae_loss_train_osd(x_data, y, cfm, lifespan, layer1_data, beta, mode, osd_weight, d_target, temperature)
            else: 
                output = model.moe_elbo_cgmvae_loss(x_data, y, cfm, lifespan, layer1_data, beta, d_target, temperature)

        loss = output['overall_loss']
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    print(f'====> Epoch: {epoch:03d} Train loss: {total_loss:.4f}')

    return output

def test(epoch, model, optimizer, test_loader, num_samples, device, output_path, dataset_abbrev, physio_cols, mode, d_target, temperature):
    model.train()
    total_loss = 0
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            model.reconstruct(test_data, output_path, epoch, dataset_abbrev, physio_cols, mode, d_target, temperature)
            print('Finished test reconstruction')

            torch.save(model, os.path.join(output_path, f'saved_model_epoch{epoch}.pth'))
            print('Model saved')