import os 
import yaml 
import pandas as pd
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import sys
from simple_mlp import*
from mlp_data_builder import*
import pickle 

def load_config(config_name):
    with open(os.path.join(config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config_mlp.yaml") 

epochs = config['training_params']['epochs']
learning_rate = config['training_params']['learning_rate']
batch_size = config['training_params']['batch_size']
num_k_folds = config['training_params']['num_k_folds']
data_path = config['data']['latent_variable_path']
path = config['data']['path']
label_type = config['data']['label_type']
data_type = config['data']['data_type']

input_size = config['model_params']['input_size']
hidden_size = config['model_params']['hidden_size']
output_size = config['model_params']['output_size']
latent_dim = config['model_params']['latent_dim']

output_folder =  config['output']['output_folder']
os.makedirs(output_folder, exist_ok=True)
accelerator = config['runtime_config']['accelerator']

data_df = pd.read_csv(data_path)
df = pd.read_csv(path)
if data_type == 'latent':
    dataset = pd.merge(data_df, df, left_on='mouse_ids', right_on='Mouse.ID')
elif data_type == 'full':
    dataset = pd.merge(data_df, df, left_on='MouseID', right_on='Mouse.ID')
if label_type == 'cfc': 
    label_col = 'CFC.24mo.Average'
elif label_type == 'ymaze_distance':
    label_col = 'Ymaze.Distance.10mo'
elif label_type == 'ymaze_speed':
    label_col = 'Ymaze.MeanSpeed.10mo'
elif label_type == 'ymaze_alt':
    label_col = 'SpontaneousAlternations.10mo'
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)
if data_type == 'latent':
    latent_columns = [f'LV{i+1}' for i in range(latent_dim)]
elif data_type == 'full':
    latent_columns = ['Generation', 'SurvDays', 'Y1_CBC_Hgb', 'Y2_CBC_Hgb', 'Y3_CBC_Hgb', 'Y1_Rotarod_Mean', 'Y2_Rotarod_Mean', 'Y3_Rotarod_Mean', 'Y1_Wheel_AvgSpeedLFC', 'Y2_Wheel_AvgSpeedLFC', 'Y3_Wheel_AvgSpeedLFC', 'Y1_Wheel_AvgDistLFC', 'Y2_Wheel_AvgDistLFC', 'Y3_Wheel_AvgDistLFC', 'Y1A_Grip_All', 'Y2A_Grip_All', 'Y3A_Grip_All', 'Y1A_BW_BW', 'Y2A_BW_BW', 'Y3A_BW_BW', 'Y1_Glu.F_Glucose', 'Y2_Glu.F_Glucose', 'Y3_Glu.F_Glucose', 'Y1_Echo_BPM', 'Y2_Echo_BPM', 'Y3_Echo_BPM', 'Y1_Echo_CardiacOutput', 'Y2_Echo_CardiacOutput', 'Y3_Echo_CardiacOutput', 'Y1_AS_MeanLog', 'Y2_AS_MeanLog', 'Y3_AS_MeanLog','Y1A_Frailty_FrailtyAdj', 'Y2A_Frailty_FrailtyAdj', 'Y3A_Frailty_FrailtyAdj']
scaler = StandardScaler()
y_scaler = StandardScaler()
train_df.loc[:, latent_columns] = scaler.fit_transform(train_df[latent_columns])
with open(os.path.join(output_folder, 'scaler.pkl'), 'wb') as file: 
    pickle.dump(scaler, file)
#train_df['mouse_ids'].to_csv('/Users/racheliritani/Desktop/AD Project/mm-vae/mouse_ids_2.csv')
test_df.loc[:, latent_columns] = scaler.transform(test_df[latent_columns])
train_df.loc[:, label_col] = y_scaler.fit_transform(train_df[[label_col]])
with open(os.path.join(output_folder, 'y_scaler.pkl'), 'wb') as file: 
    pickle.dump(y_scaler, file)
test_df.loc[:, label_col] = y_scaler.transform(test_df[[label_col]])
#X_train_tensor = torch.tensor(train_df[latent_columns + ['labels']].values, dtype=torch.float32)
# X_train_tensor = torch.tensor(train_df[latent_columns].values, dtype=torch.float32)
# y_train_tensor = torch.tensor(train_df[label_col].values, dtype=torch.float32)
train_dataset = MLPDataBuilder(train_df, latent_columns, label_col)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#X_test_tensor = torch.tensor(test_df[latent_columns + ['labels']].values, dtype=torch.float32)
# X_test_tensor = torch.tensor(test_df[latent_columns].values, dtype=torch.float32)
# y_test_tensor = torch.tensor(test_df[label_col].values, dtype=torch.float32)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_dataset = MLPDataBuilder(test_df, latent_columns, label_col)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_mouse_ids = test_df['mouse_ids']
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

if accelerator:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = simple_mlp().to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

train_losses = []
test_losses = []
mean_absolute_errors = []
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader): 
        data = data.to(device)
        labels = labels.to(device)
        output = model(data)
        loss = criterion(output, labels.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss / len(train_loader):.8f}')
    
    model.eval()
    test_loss = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader): 
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            loss = criterion(output, labels.unsqueeze(1))
            test_loss += loss.item()
            all_outputs.append(output.cpu())
            all_labels.append(labels.cpu())

    all_outputs = torch.cat(alcfl_outputs)
    all_labels = torch.cat(all_labels)

    if epoch % 10 == 0:
        #diet_labels = data[:, -1].detach().cpu().numpy()
        labels = all_labels.numpy()
        output = all_outputs.numpy()

        original_scale_labels = y_scaler.inverse_transform(labels.reshape(-1, 1))
        original_scale_output = y_scaler.inverse_transform(output.reshape(-1, 1))
        original_scale_labels = original_scale_labels.flatten()
        original_scale_output = original_scale_output.flatten()
        out_combined = {
            #'diet': diet_labels, 
            f'true_{label_type}': original_scale_labels, 
            f'predicted_{label_type}': original_scale_output, 
        }
        mae = mean_absolute_error(original_scale_labels, original_scale_output)
        mean_absolute_errors.append({
            'epoch': epoch, 
            'mae': mae
        })
        if epoch in [10, 50, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 20000]:
            out_combined_df = pd.DataFrame(out_combined)
            #out_combined_df['mouse_ids'] = test_mouse_ids[-37]
            out_combined_df['mouse_ids'] = test_mouse_ids.values
            out_combined_df.to_csv(os.path.join(output_folder, f'predicted_{label_type}_epoch{epoch}.csv'), index=False)
            print('Predicted values saved')

            torch.save(model, os.path.join(output_folder, f'saved_model_epoch{epoch}.pth'))
            print('Model saved')
    
    train_losses.append(train_loss / (len(train_loader)))
    test_losses.append(test_loss / len(test_loader))

total_error = pd.DataFrame(mean_absolute_errors)
print(total_error)
total_error.plot(x='epoch', y='mae', title=f'MAE over epochs for {label_type} prediction')

fig, ax = plt.subplots(1, 3, figsize=(12, 5))
ax[0].plot(train_losses, label='train')
ax[0].plot(test_losses, label='test')
ax[0].set_title('Non-log scale')
ax[0].legend()
ax[1].plot(train_losses, label='train')
ax[1].plot(test_losses, label='test')
ax[1].set_title('Log scale')
ax[1].legend()
ax[1].set_yscale('log')
ax[2].plot(total_error['epoch'], total_error['mae'], label='test', color='orange')
ax[2].set_title('Mean absolute error')
ax[2].legend()
plt.savefig(os.path.join(output_folder, str(label_type) + '_train_and_test_loss.png'))