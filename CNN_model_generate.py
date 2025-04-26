import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('training.csv')  # adjust the path if needed

# Preprocessing
# Fill missing values
data.fillna(method='ffill', inplace=True)

# Images are stored as space-separated pixels in a string
X = np.array([np.fromstring(img, sep=' ') for img in data['Image']])
X = X / 255.0  # Normalize pixel values
X = X.reshape(-1, 96, 96, 1)  # Reshape for CNN input

# Target keypoints
y = data.drop('Image', axis=1).values

# Normalize keypoints to [0,1] range
# Since original images are 96x96, divide by 96
y = y / 96.0

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(96, 96, 1)),
    MaxPooling2D(2,2),
    Dropout(0.1),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Flatten(),
    Dense(500, activation='relu'),
    Dropout(0.5),
    Dense(500, activation='relu'),
    Dropout(0.5),
    Dense(30)  # 15 keypoints * (x,y)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    verbose=1
)

# Plot training history
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Save model
model.save('facial_keypoints_model.h5')
