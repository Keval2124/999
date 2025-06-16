import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import soundfile as sf

# Clear any existing TensorFlow session
tf.keras.backend.clear_session()

# 1. Load and preprocess the first 20 audio logs
folder = '/home/keval/Music/999/archive/911_recordings'
audio_extensions = ('.wav', '.mp3', '.flac')
audio_files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(audio_extensions)][:20]

X = []
fixed_len = 1000  # Limit time dimension for memory efficiency
for file in audio_files:
    try:
        audio, sr = librosa.load(os.path.join(folder, file), sr=16000)  # Downsample to 16kHz
        melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)  # Reduce n_mels
        if melspec.shape[1] > fixed_len:
            melspec = melspec[:, :fixed_len]  # Crop
        elif melspec.shape[1] < fixed_len:
            melspec = np.pad(melspec, ((0, 0), (0, fixed_len - melspec.shape[1])), mode='constant')  # Pad
        X.append(melspec)
    except Exception as e:
        print(f"Error loading {file}: {e}")
print("Calls loaded:", len(X))

# 2. Preprocess: normalize
if not X:
    raise ValueError("No valid audio files loaded.")
X_processed = np.array(X)
X_processed = (X_processed - X_processed.mean()) / (X_processed.std() + 1e-8)
X_processed = X_processed[..., np.newaxis]  # Add channel dimension for Conv2D
print("Calls processed:", X_processed.shape)

# 3. Build a convolutional autoencoder
input_shape = (64, fixed_len, 1)
encoder = tf.keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
])

decoder = tf.keras.Sequential([
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
])

autoencoder = tf.keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
print("Convolutional autoencoder built.")

# 4. Train the autoencoder
autoencoder.fit(X_processed, X_processed, epochs=10, batch_size=1, verbose=1)

# 5. Generate a new spectrogram
generated_melspec = autoencoder.predict(X_processed[:1])[0, ..., 0]  # Remove channel dimension

# 6. Convert back to audio
try:
    generated_audio = librosa.feature.inverse.mel_to_audio(generated_melspec, sr=sr)
    sf.write('generated_call.wav', generated_audio, sr)
    print("Generated audio saved as 'generated_call.wav'.")
except Exception as e:
    print(f"Error generating audio: {e}")