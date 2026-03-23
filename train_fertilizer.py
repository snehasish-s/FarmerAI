import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

# Create models dir if not exist
os.makedirs('models', exist_ok=True)

# Synthetic dataset for fertilizer recommendation
np.random.seed(42)
n_samples = 2500

crops = ['wheat', 'corn', 'rice', 'soybean', 'cotton']
soils = ['sandy', 'loamy', 'clay', 'silty']

data = {
    'crop_type': np.random.choice(crops, n_samples),
    'soil_type': np.random.choice(soils, n_samples),
    'target_yield_t_ha': np.random.uniform(2, 8, n_samples),  # Target yield tons/ha
    'current_soil_n': np.random.uniform(20, 120, n_samples),   # Current N ppm
}

df = pd.DataFrame(data)

# Label encoding
crop_map = {crop: i for i, crop in enumerate(crops)}
soil_map = {soil: i for i, soil in enumerate(soils)}
df['crop_encoded'] = df['crop_type'].map(crop_map)
df['soil_encoded'] = df['soil_type'].map(soil_map)

# Fertilizer requirements (kg/ha) - realistic base values adjusted by factors
base_npk = {
    'wheat': [120, 60, 40],
    'corn': [150, 70, 50],
    'rice': [100, 50, 40],
    'soybean': [30, 60, 30],
    'cotton': [120, 50, 60]
}

# Calculate required NPK
df['req_n'] = np.zeros(n_samples)
df['req_p'] = np.zeros(n_samples)
df['req_k'] = np.zeros(n_samples)

for i in range(n_samples):
    crop = df.loc[i, 'crop_type']
    base = base_npk[crop]
    soil_factor = 1.0 + 0.2 * df.loc[i, 'soil_encoded']  # Soil affects needs
    yield_factor = df.loc[i, 'target_yield_t_ha'] / 5.0   # Scale by yield goal
    current_n = df.loc[i, 'current_soil_n']
    
    df.loc[i, 'req_n'] = max(0, base[0] * soil_factor * yield_factor - current_n * 2)
    df.loc[i, 'req_p'] = base[1] * soil_factor * yield_factor * 0.8
    df.loc[i, 'req_k'] = base[2] * soil_factor * yield_factor * 0.9

# Features: crop, soil, yield_goal, current_N
X = df[['crop_encoded', 'soil_encoded', 'target_yield_t_ha', 'current_soil_n']].values
y_n = df['req_n'].values.reshape(-1,1)
y_p = df['req_p'].values.reshape(-1,1)
y_k = df['req_k'].values.reshape(-1,1)

# Normalize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_scaled = (X - X_mean) / X_std

# Split
split = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_n_train, y_n_test = y_n[:split], y_n[split:]
y_p_train, y_p_test = y_p[:split], y_p[split:]
y_k_train, y_k_test = y_k[:split], y_k[split:]

# Multi-output model for N,P,K
input_layer = tf.keras.layers.Input(shape=(4,))
x = tf.keras.layers.Dense(128, activation='relu')(input_layer)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)

n_out = tf.keras.layers.Dense(1, activation='relu', name='nitrogen')(x)
p_out = tf.keras.layers.Dense(1, activation='relu', name='phosphorus')(x)
k_out = tf.keras.layers.Dense(1, activation='relu', name='potassium')(x)

model = tf.keras.Model(inputs=input_layer, outputs=[n_out, p_out, k_out])
model.compile(optimizer='adam', loss={'nitrogen': 'mse', 'phosphorus': 'mse', 'potassium': 'mse'})

# Train
history = model.fit(X_train, {'nitrogen': y_n_train, 'phosphorus': y_p_train, 'potassium': y_k_train},
                   epochs=100, batch_size=64, verbose=1, validation_split=0.1)

# Save model
model.save('models/fertilizer_model.keras')
print("✅ fertilizer_model.keras saved!")

# Save preprocessors
preproc = {
    'crop_map': crop_map,
    'soil_map': soil_map,
    'X_mean': X_mean,
    'X_std': X_std,
    'crops': crops,
    'soils': soils,
    'base_npk': base_npk
}
with open('models/fertilizer_preproc.pkl', 'wb') as f:
    pickle.dump(preproc, f)
print("✅ fertilizer_preproc.pkl saved!")
print("Model ready for app.py! Test with: python train_fertilizer.py")
