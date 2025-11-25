# this file will have the training process for the MoE mmVAE

#from __future__ import print_function
import torch
import torch.utils.data
from torchvision.utils import save_image
from loss_mse import*
from loss_bce import*
from utils import*

def train(epoch, model, train_loader, device, optimizer, log_interval, loss_type):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        # trying adding kl annealing 
        #beta = min(1.0, epoch / 10)
        beta = 1
        if loss_type == 'bce':
            loss, recon, kld = loss_function_mse(recon_batch, data, mu, logvar, beta)
        elif loss_type == 'mse':
            loss, recon, kld = loss_function_mse(recon_batch, data, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        recon_loss += recon.item()
        kld_loss += kld.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    train_loss /= len(train_loader.dataset)
    recon_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss))
    
    return train_loss, recon_loss, kld_loss


def test(epoch, model, test_loader, device, batch_size, output_folder, loss_type):
    model.eval()
    test_loss = 0
    recon_loss = 0
    kld_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            # trying adding kl annealing 
            #beta = min(1.0, epoch / 10)
            beta = 1
            if loss_type == 'bce':
                loss, recon, kld = loss_function_mse(recon_batch, data, mu, logvar, beta)
            elif loss_type == 'mse':
                loss, recon, kld = loss_function_mse(recon_batch, data, mu, logvar, beta)
            test_loss += loss.item()
            recon_loss += recon.item()
            kld_loss += kld.item()
            if i == 0:
                n = min(data.size(0), 8)
                if recon_batch.shape[-1] % 28 == 0:
                    comparison = torch.cat([data[:n],
                    recon_batch.view(batch_size, 1, 28, 28)[:n]])
                    #comparison = unnormalize(comparison, [0.1307], [0.3801])
                else:
                    comparison = torch.cat([data[:n],
                                       recon_batch.view(batch_size, 3, 32, 32)[:n]])
                    #comparison = unnormalize(comparison, (0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
                save_image(comparison.cpu(),
                         output_folder + '/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    recon_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss, recon_loss, kld_loss