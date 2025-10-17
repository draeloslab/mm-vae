from torchvision import transforms
import numpy as np
import sys

def load_data(df):
    x = df.iloc[:, 1:-5].values.astype('float32')
    #diet_label = df['Diet_num'].values - 1 
    cfc_label = df['cfc_bin'].values
    mouse_ids = df['MouseID']
    for column in range(x.shape[1]):
        x[:, column] = standardize(x[:, column])
    df_layer1 = x.copy()
    return x, cfc_label, df_layer1, mouse_ids

def standardize(input_array):
    mean = np.nanmean(input_array)
    std = np.nanstd(input_array)
    return (input_array - mean) / std

def set_nans_extreme(df):
    # min = CBC, rotarod, wheel dist, wheel speed, grip strength
    # max = BW, frailty, glucose, heart rate, cardiac output, acoustic 
    min_columns = ['Y1_CBC_Hgb', 'Y2_CBC_Hgb', 'Y3_CBC_Hgb', 'Y1_Rotarod_Mean', 'Y2_Rotarod_Mean', 'Y3_Rotarod_Mean', 'Y1_Wheel_AvgSpeedLFC', 'Y2_Wheel_AvgSpeedLFC', 'Y3_Wheel_AvgSpeedLFC', 'Y1_Wheel_AvgDistLFC', 'Y2_Wheel_AvgDistLFC', 'Y3_Wheel_AvgDistLFC', 'Y1A_Grip_All', 'Y2A_Grip_All', 'Y3A_Grip_All']
    max_columns = ['Y1A_BW_BW', 'Y2A_BW_BW', 'Y3A_BW_BW', 'Y1_Glu.F_Glucose', 'Y2_Glu.F_Glucose', 'Y3_Glu.F_Glucose', 'Y1_Echo_BPM', 'Y2_Echo_BPM', 'Y3_Echo_BPM', 'Y1_Echo_CardiacOutput', 'Y2_Echo_CardiacOutput', 'Y3_Echo_CardiacOutput', 'Y1_AS_MeanLog', 'Y2_AS_MeanLog', 'Y3_AS_MeanLog','Y1A_Frailty_FrailtyAdj', 'Y2A_Frailty_FrailtyAdj', 'Y3A_Frailty_FrailtyAdj']
    for min_col in min_columns: 
        df[min_col] = df[min_col].fillna(np.min(df[min_col]))
    for max_col in max_columns: 
        df[max_col] = df[max_col].fillna(np.max(df[max_col]))
    return df

def load_data_pseudo(df):
    x = df.iloc[:, 1:-2].values.astype('float32')
    cfc_label = df['pseudo_label'].values
    mouse_ids = df['MouseID']
    for column in range(x.shape[1]):
        x[:, column] = standardize(x[:, column])
    df_layer1 = x.copy()
    return x, cfc_label, df_layer1, mouse_ids
