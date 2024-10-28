# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load the data (replace with the actual path to your numpy file)
data = np.load('data_000.npy')  # Update with your actual file path

# Separate spectra and labels
spectra = data[:, :35692]  # First 35,692 columns are power spectra
labels = data[:, 35692:]   # Last 36 columns are the parameter values (labels)

# Normalize spectra and labels using StandardScaler
scaler_spectra = StandardScaler()
spectra_scaled = scaler_spectra.fit_transform(spectra)

scaler_labels = StandardScaler()
labels_scaled = scaler_labels.fit_transform(labels)

# Apply Incremental PCA to reduce dimensionality
n_components = 1000  # Number of PCA components (you can try different values)
batch_size = 1000    # Batch size for Incremental PCA, now set to 1000 to match n_components

# Set up IncrementalPCA
ipca = IncrementalPCA(n_components=n_components)
reduced_spectra = np.empty((spectra_scaled.shape[0], n_components))  # Prepare array for reduced data

# Fit and transform the data in batches
for i in range(0, spectra_scaled.shape[0], batch_size):
    end = i + batch_size if i + batch_size <= spectra_scaled.shape[0] else spectra_scaled.shape[0]
    reduced_spectra[i:end] = ipca.fit_transform(spectra_scaled[i:end])

print("Incremental PCA transformation completed.")

# Split into training and cross-validation sets (90% training, 10% validation)
num_train = int(0.9 * len(data))

train_spectra = reduced_spectra[:num_train]
train_labels = labels_scaled[:num_train]
cv_spectra = reduced_spectra[num_train:]
cv_labels = labels_scaled[num_train:]

# Reshape spectra for RNN input (needed for LSTM)
train_spectra = train_spectra.reshape(train_spectra.shape[0], train_spectra.shape[1], 1)
cv_spectra = cv_spectra.reshape(cv_spectra.shape[0], cv_spectra.shape[1], 1)

# Define model parameters
rnn_units = 64  # Adjust as needed (more units can increase model capacity)
batch_size = 128  # Adjust based on your system's memory
max_epochs = 50   # Increase if the model needs more time to converge

# Build RNN model
model = Sequential([
    LSTM(rnn_units, activation='relu', input_shape=(n_components, 1), return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    LSTM(rnn_units, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),  # Dense layer with ReLU activation
    Dense(36)  # Output layer for 36 label values
])

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Display model summary
model.summary()

# Define callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

# Train the model
history = model.fit(train_spectra, train_labels,
                    validation_data=(cv_spectra, cv_labels),
                    epochs=max_epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, reduce_lr])

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Plot sample predictions vs actual labels
predictions = model.predict(cv_spectra[:5])

for i in range(5):
    plt.figure(figsize=(10, 5))
    plt.plot(predictions[i], label='Predicted', color='red')
    plt.plot(cv_labels[i], label='Actual', color='blue')
    plt.title(f'Predicted vs Actual for Sample {i+1}')
    plt.xlabel('Parameter Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()