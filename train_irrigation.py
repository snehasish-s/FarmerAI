import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

# Create models dir if not exist
os.makedirs('models', exist_ok=True)

# Dummy data
np.random.seed(42)
n_samples = 2000

crops = ['wheat', 'corn', 'rice', 'soybean', 'cotton']
soils = ['sandy', 'loamy', 'clay', 'silty']

data = {
    'crop_type': np.random.choice(crops, n_samples),
    'soil_type': np.random.choice(soils, n_samples),
    'temperature': np.random.uniform(15, 40, n_samples),
    'rainfall_mm': np.random.uniform(0, 300, n_samples),
    'area_acres': np.random.uniform(1, 100, n_samples)
}

df = pd.DataFrame(data)

# Manual label encoding
crop_map = {crop: i for i, crop in enumerate(crops)}
soil_map = {soil: i for i, soil in enumerate(soils)}
df['crop_encoded'] = df['crop_type'].map(crop_map)
df['soil_encoded'] = df['soil_type'].map(soil_map)

# Generate targets
df['water_mmday'] = np.clip(
    5 + 2*df['crop_encoded'] + 1.5*df['soil_encoded'] + 
    0.3*df['temperature'] + 0.1*(300 - df['rainfall_mm']) +
    0.5*np.random.randn(n_samples), 1, 15
)

df['freq_perweek'] = np.clip(np.round(df['water_mmday'] / 3 + np.random.uniform(0.5, 2, n_samples)), 1, 7).astype(int)

# Features
X = df[['crop_encoded', 'soil_encoded', 'temperature', 'rainfall_mm', 'area_acres']].values
y_water = df['water_mmday'].values.reshape(-1,1)
y_freq = df['freq_perweek'].values.reshape(-1,1)

# Manual normalize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_scaled = (X - X_mean) / X_std

# Split
split = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_water_train, y_water_test = y_water[:split], y_water[split:]
y_freq_train, y_freq_test = y_freq[:split], y_freq[split:]

# Model
input_layer = tf.keras.layers.Input(shape=(5,))
x = tf.keras.layers.Dense(64, activation='relu')(input_layer)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
water_out = tf.keras.layers.Dense(1, name='water')(x)
freq_out = tf.keras.layers.Dense(1, activation='relu', name='freq')(x)

model = tf.keras.Model(inputs=input_layer, outputs=[water_out, freq_out])
model.compile(optimizer='adam', loss={'water': 'mse', 'freq': 'mse'})

# Train
model.fit(X_train, {'water': y_water_train, 'freq': y_freq_train}, 
          epochs=50, batch_size=32, verbose=0, validation_split=0.1)

# Save model
model.save('models/irrigation_model.keras')
print("✅ irrigation_model.keras saved!")

# Save preprocessors
preproc = {
    'crop_map': crop_map,
    'soil_map': soil_map,
    'X_mean': X_mean,
    'X_std': X_std,
    'crops': crops,
    'soils': soils
}
with open('models/irrigation_preproc.pkl', 'wb') as f:
    pickle.dump(preproc, f)
print("✅ irrigation_preproc.pkl saved!")
print("Model ready for app.py!")
