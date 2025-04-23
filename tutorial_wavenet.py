import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from wavenet_model import create_wavenet_model, train_model, predict_and_plot
from tqdm import tqdm
import time

# Configure TensorFlow for M3
from test_m3 import setup_m3_environment
setup_m3_environment()

# Load and prepare the data
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    start_time = time.time()
    
    # Use chunksize to load large files more efficiently
    chunks = []
    for chunk in tqdm(pd.read_table(file_path, sep="\s+", header=None, chunksize=10000), desc="Loading data chunks"):
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    print("Filtering activities...")
    hand_dataset = df[[0, 1, 3, 4, 7, 10, 13]]
    
    # get sitting(2), running(5), ascending stairs(12), vacuum cleaning(16)
    sit = hand_dataset.loc[hand_dataset[1] == 2]
    run = hand_dataset.loc[hand_dataset[1] == 5]
    stairs = hand_dataset.loc[hand_dataset[1] == 12]
    vac = hand_dataset.loc[hand_dataset[1] == 16]
    
    def prep_df(df):
        df.reset_index(inplace=True)
        df.drop('index', axis=1, inplace=True)
        return df
    
    print("Preparing datasets...")
    sit2 = prep_df(sit)
    run2 = prep_df(run)
    stairs2 = prep_df(stairs)
    vac2 = prep_df(vac)
    
    print(f"Data preparation completed in {time.time() - start_time:.2f} seconds")
    return sit2, run2, stairs2, vac2

def plotter(df, activity_name):
    print(f"Plotting {activity_name} data...")
    start_time = time.time()
    
    # Visualize response variable (total count) with other vars in hourly dataset
    col_list = [3, 4, 7, 10, 13]
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(15,15))
    for i in range(len(col_list)):
        ax[i].plot(df[col_list[i]], label=col_list[i])
        ax[i].set_title(col_list[i])
    plt.tight_layout()
    
    # Save the plot instead of showing it
    plt.savefig(f"{activity_name}_plot.png")
    plt.close()
    
    print(f"Plotting completed in {time.time() - start_time:.2f} seconds")

def get_time_block_series(series_array, start, end):
    return series_array[:,start:end]

def transform_series_encode(series_array):
    print("Encoding series data...")
    start_time = time.time()
    
    series_array = np.log1p(series_array) 
    series_array = np.nan_to_num(series_array)  # filling NaN with 0
    series_mean = series_array.mean(axis=1).reshape(-1,1) 
    series_array = series_array - series_mean
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))
    
    print(f"Encoding completed in {time.time() - start_time:.2f} seconds")
    return series_array, series_mean

def transform_series_decode(series_array, encode_series_mean):
    series_array = np.log1p(series_array) 
    series_array = np.nan_to_num(series_array)  # filling NaN with 0
    series_array = series_array - encode_series_mean
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))
    return series_array

def main():
    # File path
    file_path = "data/PAMAP2_Dataset/Protocol/PAMAP2_Dataset/Protocol/subject102.dat"
    
    # Load and prepare data
    print("Loading data...")
    sit2, run2, stairs2, vac2 = load_data(file_path)
    
    # Plot the data
    print("Plotting data for all activities...")
    plotter(sit2, "sitting")
    plotter(run2, "running")
    plotter(stairs2, "stairs")
    plotter(vac2, "vacuum")
    
    # Prepare data for model
    print("Preparing data for model...")
    start_time = time.time()
    
    cols = [3, 4, 7, 10, 13]
    sit_ = sit2[cols]
    sit_t = sit_.transpose()
    
    # Set up prediction parameters
    pred_steps = 3000
    pred_length = 3000
    first_day = 0
    last_day = sit_t.shape[1]
    
    val_pred_start = last_day - pred_length - 1
    val_pred_end = last_day
    
    train_pred_start = val_pred_start - pred_length - 1
    train_pred_end = val_pred_start - 1
    
    enc_length = train_pred_start - first_day
    
    train_enc_start = first_day
    train_enc_end = train_enc_start + enc_length - 1
    
    val_enc_start = train_enc_start + pred_length
    val_enc_end = val_enc_start + enc_length
    
    print('\nTraining parameters:')
    print('Train encoding:', train_enc_start, '-', train_enc_end)
    print('Train prediction:', train_pred_start, '-', train_pred_end)
    print('Val encoding:', val_enc_start, '-', val_enc_end)
    print('Val prediction:', val_pred_start, '-', val_pred_end)
    print('\nEncoding interval:', enc_length)
    print('Prediction interval:', pred_length)
    
    # Prepare training data
    print("Preparing training data...")
    series_array = sit_t[sit_t.columns[1:]].values
    first_n_samples = 15000
    
    # Training history
    encoder_input_data = get_time_block_series(series_array, train_enc_start, train_enc_end)[:first_n_samples]
    encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)
    
    # Training target
    decoder_target_data = get_time_block_series(series_array, train_pred_start, train_pred_end)[:first_n_samples]
    decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)
    
    print("\nEncoder input shape:", encoder_input_data.shape)
    print("Decoder target shape:", decoder_target_data.shape)
    
    print(f"Data preparation completed in {time.time() - start_time:.2f} seconds")
    
    # Create and train model
    print("\nCreating model...")
    model = create_wavenet_model(seq_length=pred_steps)
    model.summary()
    
    print("\nTraining model...")
    model, history = train_model(model, encoder_input_data, decoder_target_data)
    
    # Prepare validation data
    print("\nPreparing validation data...")
    encoder_input_val = get_time_block_series(series_array, val_enc_start, val_enc_end)
    encoder_input_val, encode_series_mean_val = transform_series_encode(encoder_input_val)
    
    decoder_target_val = get_time_block_series(series_array, val_pred_start, val_pred_end)
    decoder_target_val = transform_series_decode(decoder_target_val, encode_series_mean_val)
    
    # Make predictions and plot results for each sensor
    print("\nMaking predictions...")
    for sensor_idx in range(5):
        print(f"\nPredicting for sensor {sensor_idx}...")
        pred_sensor, truth_sensor = predict_and_plot(model, encoder_input_val, decoder_target_val, sensor_idx)

if __name__ == "__main__":
    main()

