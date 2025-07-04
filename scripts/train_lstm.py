# Updated LSTM Model for Drought Forecasting Across All Locations (Multivariate + Multi-Location)
# Complete LSTM Model for Multi-Location Weekly Drought Prediction Using Selected Features

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters

time_steps = 12
future_steps = 12
features = [
    'score',  # target variable
    'PRECTOT',  # precipitation
    'T2M',  # temperature
    'QV2M',  # humidity
    'PS',  # surface pressure
    'precipitation_mean_12',  # 12-week precipitation mean
    'humidity_mean_12'  # 12-week humidity mean
]
DROUGHT_THRESHOLD = 0.5

def prepare_sequences(df_loc, scaler):
    df_loc = df_loc.sort_index()
    # Ensure all required features are present
    missing_features = [f for f in features if f not in df_loc.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    df_loc = df_loc[features].copy().interpolate(method='linear').dropna()
    scaled = scaler.fit_transform(df_loc)
    X, y = [], []
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i - time_steps:i])
        y.append(scaled[i, 0])  # 'score' as target
    return np.array(X), np.array(y), scaler

# Load and prepare data
try:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    data_files = [
        'ph_combined_train_data_weekly.csv',
        'ph_combined_val_data_weekly.csv',
        'ph_combined_test_data_weekly.csv'
    ]

    dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in data_files]
    df_all = pd.concat(dfs)
    df_all['start_date'] = pd.to_datetime(df_all['start_date'], dayfirst=True, errors='coerce')

   
    df_all.set_index('start_date', inplace=True)
    logger.info("Merged dataset shape: %s", df_all.shape)
    logger.info("Available features: %s", df_all.columns.tolist())

    all_locations = df_all[['lat', 'lon']].drop_duplicates().values
    logger.info("Total unique locations: %d", len(all_locations))

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(time_steps, len(features))),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2, min_lr=0.0001)

    all_X, all_y = [], []
    scalers = {}

    for lat, lon in all_locations:
        df_loc = df_all[(df_all['lat'] == lat) & (df_all['lon'] == lon)].copy()
        if df_loc.empty or df_loc[features].isnull().sum().sum() > len(df_loc):
            continue
        scaler = MinMaxScaler()
        try:
            X, y, fitted_scaler = prepare_sequences(df_loc, scaler)
            all_X.append(X)
            all_y.append(y)
            scalers[(lat, lon)] = fitted_scaler
        except Exception as e:
            logger.warning(f"Skipping location ({lat}, {lon}) due to: {e}")

    if not all_X:
        raise ValueError("No valid data sequences were created. Check your data and features.")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    logger.info("Final training shape: X=%s, y=%s", X_all.shape, y_all.shape)

    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, shuffle=False)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Save model and scalers
    os.makedirs('models', exist_ok=True)
    model.save('models/multiloc_lstm_model.keras')
    
    # Save scalers
    import pickle
    with open('models/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig('models/training_loss.png')
    plt.close()

except Exception as e:
    logger.error(f"An error occurred: {e}")
    raise
