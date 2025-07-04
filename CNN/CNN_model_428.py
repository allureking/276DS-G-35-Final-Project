import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model_save_name = 'facial_keypoints_adam_428_4.h5'
final_model_save_name= 'facial_keypoints_sgd_428_4.h5'

# Load the data & IdLookupTable
data = pd.read_csv('training.csv')
lookup_table = pd.read_csv('IdLookupTable.csv')

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
    Conv2D(32, (3,3)),
    LeakyReLU(alpha=0.01),
    MaxPooling2D(2,2),
    Dropout(0.05),

    Conv2D(64, (3,3)),
    LeakyReLU(alpha=0.01),
    MaxPooling2D(2,2),
    Dropout(0.01),

    Conv2D(128, (3,3)),
    LeakyReLU(alpha=0.01),
    MaxPooling2D(2,2),
    Dropout(0.15),

    Flatten(),
    Dense(500),
    LeakyReLU(alpha=0.01),
    Dropout(0.5),
    Dense(500),
    LeakyReLU(alpha=0.01),
    Dropout(0.5),
    Dense(30)  # 15 keypoints * (x,y)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=Huber(delta=1.0),
              metrics=['mae'])


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=250,
    verbose=1,
    batch_size=64,
    callbacks=[early_stopping]
)
# Save model
model.save(model_save_name)

#--------------------------------------------------------------------------------------
# Compile the model with SGD
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
              loss=Huber(delta=1.0),
              metrics=['mae'])

early_stopping_sgd = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    verbose=1,
    min_lr=1e-6
)

# Fine-tune with SGD
history_sgd = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    verbose=1,
    batch_size=64,
    callbacks=[early_stopping_sgd, reduce_lr]
)

model.save(final_model_save_name)

#-----------------------------------------------------------------------------
# Load test data
test_data = pd.read_csv('test.csv')
X_test = np.array([np.fromstring(img, sep=' ') for img in test_data['Image']])
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 96, 96, 1)
# Predict
predictions = model.predict(X_test)
predictions = predictions * 96

# Define keypoint columns (this must match your training labels)
keypoint_columns = [
    'left_eye_center_x', 'left_eye_center_y',
    'right_eye_center_x', 'right_eye_center_y',
    'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
    'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
    'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
    'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
    'nose_tip_x', 'nose_tip_y',
    'mouth_left_corner_x', 'mouth_left_corner_y',
    'mouth_right_corner_x', 'mouth_right_corner_y',
    'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
    'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'
]

# Make a DataFrame with predictions
preds_df = pd.DataFrame(predictions, columns=keypoint_columns)

# Prepare final submission
locations = []
for idx, row in lookup_table.iterrows():
    image_id = row['ImageId'] - 1
    feature_name = row['FeatureName']
    locations.append(preds_df.loc[image_id, feature_name])

submission = lookup_table.copy()
submission['Location'] = locations
submission = submission[['RowId', 'Location']]
submission.to_csv('submission.csv', index=False)

print("'submission.csv' created")

# Plot training history
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
