import cv2
import numpy as np
import pandas as pd
import os


# Define the function to extract color histogram features
def extract_color_histogram(img_path):
    img = cv2.imread(img_path)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


# Define the path to the cropped folder
cropped_folder_path = "C:/xampp/htdocs/Image Classification/Model/dataset/cropped"

# Define the list to store the extracted features
features = []

# Loop through the folders in the cropped folder
for folder_name in os.listdir(cropped_folder_path):
    folder_path = os.path.join(cropped_folder_path, folder_name)

    # Loop through the images in each folder
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        # Extract the color histogram features from the image
        hist = extract_color_histogram(img_path)

        # Append the features and the label to the list
        features.append(np.append(hist, folder_name))

# Convert the features list to a DataFrame and save it to a CSV file
df = pd.DataFrame(features)
df.to_csv("cropped_features.csv", index=False)