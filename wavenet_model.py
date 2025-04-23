import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def create_wavenet_model(seq_length, n_filters=32, filter_width=2):
    """Create a WaveNet model for time series prediction."""
    print("Creating WaveNet model...")
    start_time = time.time()
    
    # Define dilations for the causal convolutions
    dilation_rates = [2**i for i in range(8)]
    
    # Define input layer
    history_seq = Input(shape=(None, 1), name='input_layer')
    
    # Create a casting layer
    class CastToFloat16(tf.keras.layers.Layer):
        def call(self, inputs):
            return tf.cast(inputs, dtype=tf.float16)
    
    x = CastToFloat16()(history_seq)
    
    # Stack of dilated causal convolutions
    for i, dilation_rate in enumerate(dilation_rates):
        x = Conv1D(filters=n_filters,
                  kernel_size=filter_width, 
                  padding='causal',
                  dilation_rate=dilation_rate)(x)
        if i % 2 == 0:  # Add batch normalization every other layer
            x = tf.keras.layers.BatchNormalization()(x)
    
    # Dense layers for final processing
    x = Dense(128, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(1)(x)
    
    # Extract the last seq_length time steps as the training target
    def slice(x, seq_length):
        return x[:,-seq_length:,:]
    
    pred_seq_train = Lambda(slice, arguments={'seq_length':seq_length})(x)
    
    # Create and return model
    model = Model(history_seq, pred_seq_train)
    
    print(f"Model created in {time.time() - start_time:.2f} seconds")
    return model

def train_model(model, encoder_input_data, decoder_target_data, batch_size=2048, epochs=50):
    """Train the WaveNet model."""
    print("\nTraining model...")
    start_time = time.time()
    
    # Use mixed precision optimizer
    optimizer = Adam(learning_rate=0.001)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(optimizer, loss='mean_absolute_error')
    
    # Add teacher forcing: append lagged target history to input data
    print("Preparing training data...")
    lagged_target_history = decoder_target_data[:,:-1,:1]
    encoder_input_data = np.concatenate([encoder_input_data, lagged_target_history], axis=1)
    
    # Calculate steps per epoch
    steps_per_epoch = len(encoder_input_data) // batch_size
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    # Custom progress bar callback
    class TqdmProgressBar(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epochs = None
            self.steps = None
            self.progress_bar = None
            
        def on_train_begin(self, logs=None):
            self.epochs = self.params['epochs']
            self.steps = self.params['steps']
            
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            self.progress_bar = tqdm(total=self.steps, desc='Training')
            
        def on_train_batch_end(self, batch, logs=None):
            self.progress_bar.update(1)
            self.progress_bar.set_postfix({
                'loss': f"{logs['loss']:.4f}",
                'val_loss': f"{logs.get('val_loss', 0):.4f}"
            })
            
        def on_epoch_end(self, epoch, logs=None):
            self.progress_bar.close()
    
    # Train the model
    history = model.fit(
        encoder_input_data, 
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks + [TqdmProgressBar()],
        verbose=0
    )
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error Loss')
    plt.title('Model Training History')
    plt.legend(['Train', 'Validation'])
    plt.savefig('training_history.png')
    plt.close()
    
    return model, history

def predict_and_plot(model, encoder_input_data, decoder_target_data, sensor_idx=0, tail_length=5000):
    """Make predictions and plot results for a specific sensor."""
    print(f"\nMaking predictions for sensor {sensor_idx}...")
    start_time = time.time()
    
    # Make predictions in batches to manage memory
    batch_size = 32
    predictions = []
    
    for i in tqdm(range(0, len(encoder_input_data), batch_size), desc="Making predictions"):
        batch = encoder_input_data[i:i+batch_size]
        batch_pred = model.predict(batch, verbose=0)
        predictions.append(batch_pred)
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Extract predictions and ground truth for the specified sensor
    pred_sensor = predictions[sensor_idx, :, :]
    truth_sensor = decoder_target_data[sensor_idx, :]
    truth_sensor = np.reshape(truth_sensor, (len(truth_sensor), -1))
    
    # Get the last tail_length timesteps from the input data
    last_steps = (encoder_input_data[sensor_idx, :, :])[-tail_length:]
    
    # Plot results
    plt.figure(figsize=(20, 6))
    plt.plot(range(1, tail_length + 1), last_steps)
    plt.plot(range(tail_length, tail_length + pred_sensor.shape[0]), truth_sensor, color='orange')
    plt.plot(range(tail_length, tail_length + pred_sensor.shape[0]), pred_sensor, color='teal', linestyle='--')
    
    plt.title(f'Sensor {sensor_idx}: History, Target, and Predictions')
    plt.legend(['Input History', 'Target', 'Predictions'])
    plt.savefig(f'sensor_{sensor_idx}_predictions.png')
    plt.close()
    
    print(f"Predictions completed in {time.time() - start_time:.2f} seconds")
    return pred_sensor, truth_sensor 