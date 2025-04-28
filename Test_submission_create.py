import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

lookup_table = pd.read_csv('IdLookupTable.csv')
model = load_model('Model File')

test_data = pd.read_csv('test.csv')
X_test = np.array([np.fromstring(img, sep=' ') for img in test_data['Image']])
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 96, 96, 1)

predictions = model.predict(X_test)
predictions = predictions * 96

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

preds_df = pd.DataFrame(predictions, columns=keypoint_columns)

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

