from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

Y_COL = 'FULLVAL'

def fit_transform_on(data_path):
    df = pd.read_csv(data_path)
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    numerical_columns = set(numerical_columns).symmetric_difference([Y_COL])
    
    scaler = StandardScaler()
    scaler.fit(df[numerical_columns])
    scaled = scaler.transform(df[numerical_columns])
    df.loc[:, numerical_columns] = scaled
    
    scaler_y = StandardScaler()
    scaler_y.fit(df[[Y_COL]])
    scaled_y = scaler_y.transform(df[[Y_COL]])
    df.loc[:, Y_COL] = scaled_y.squeeze()
    
    return scaler, scaler_y, df

def save_scaler_as(scaler, path):
    joblib.dump(scaler, path)
    
def load_scaler(path):
    return joblib.load(path) 

def transform(data, scaler_path):
    scaler = load_scaler(scaler_path)
    return scaler.transform(data)

def log_scale(data):
    log_data = data.copy()
    log_data.loc[log_data <= 0] = 1e-5
    log_data = np.log(log_data)
    log_data.loc[log_data < -3] = 0
    return log_data

def reverse_log(data):
    return np.exp(data)

def reverse_transform(data, scaler_path):
    scaler = load_scaler(scaler_path)
    return scaler.inverse_transform(data)
    
    