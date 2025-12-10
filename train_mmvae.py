import torch
import torch.utils.data
from utils import*

def train(epoch, model, optimizer, train_loader, num_samples, device):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_loader):
        
        # unpack/prepare data
        x_mnist = data[0][0].to(device)
        x_svhn = data[1][0].to(device)
        x_data = [x_mnist, x_svhn]

        # x_data.to(device) maybe don't need this?
        optimizer.zero_grad()
        #loss = model.moe_elbo_loss(x_data, K=num_samples)
        loss = model.moe_iwae_loss(x_data, K=num_samples)
        #loss = model._m_iwae(x_data, K=num_samples)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f'====> Epoch: {epoch:03d} Train loss: {total_loss / len(train_loader.dataset):.4f}')

    return total_loss / len(train_loader.dataset)

def test(epoch, model, optimizer, test_loader, num_samples, device, output_path):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            x_mnist = test_data[0][0].to(device)
            x_svhn = test_data[1][0].to(device)
            x_test_data = [x_mnist, x_svhn]

            optimizer.zero_grad()
            #loss = model.moe_elbo_loss(x_test_data, K=num_samples)
            #loss = model._m_iwae(x_test_data, K=num_samples)
            loss = model.moe_iwae_loss(x_test_data, K=num_samples)
            total_loss += loss.item()
            if i == 0:
                model.reconstruct(x_test_data, output_path, epoch)
                print('Finished test reconstruction')
            total_loss += loss.item()
        save_latent_space(model, epoch, test_loader, device, output_path, 2)

    print(f'====> Epoch: {epoch:03d} Test loss: {total_loss / len(test_loader.dataset):.4f}')

    return total_loss / len(test_loader)


