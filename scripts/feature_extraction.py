import cv2
import numpy as np
import os

def extract_features(dataset_path):
    features = []
    labels = []

    for label in ['real_smiles', 'fake_smiles']:
        folder_path = os.path.join(dataset_path, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping...")
                continue
            
            # Resize image to ensure consistent feature length
            img = cv2.resize(img, (64, 64))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Flatten the gray image to form the feature vector
            features.append(gray.flatten())
            labels.append(1 if label == 'real_smiles' else 0)

    np.save('models/features.npy', np.array(features))
    np.save('models/labels.npy', np.array(labels))

extract_features('dataset/')
