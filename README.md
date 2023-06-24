# Image Classification using Machine Learning Models

This repository contains a project focused on the image classification of famous athletes, specifically Lionel Messi, Maria Sharapova, Roger Federer, and Serena Williams, using machine learning models such as SVM, logistic regression, and random forest. The objective is to compare the performance of these models in accurately identifying and classifying images of these athletes.

## Dataset

The dataset used for this project consists of a collection of labeled images of Lionel Messi, Maria Sharapova, Roger Federer, and Serena Williams. The dataset is divided into a training set and a test set, with a balanced distribution of images for each athlete.

## Machine Learning Models

Three machine learning models are employed for image classification: SVM, logistic regression, and random forest. Each model has its own strengths and weaknesses, and the project aims to compare their performance in terms of accuracy and other relevant metrics.

### Support Vector Machine (SVM)

SVM is a powerful supervised learning algorithm used for classification tasks. It finds an optimal hyperplane that separates the different classes by maximizing the margin. In this project, SVM is utilized for image classification by training it on the labeled images of the athletes.

### Logistic Regression

Logistic regression is a popular statistical model used for binary classification problems. It estimates the probability of an input belonging to a specific class. Although originally designed for binary classification, it can be extended to handle multi-class classification tasks using various techniques such as one-vs-rest or softmax regression.

### Random Forest

Random forest is an ensemble learning method that combines multiple decision trees to make predictions. Each tree is trained on a different subset of the dataset, and the final prediction is obtained through voting or averaging. Random forest can handle both classification and regression tasks and is known for its robustness and ability to handle high-dimensional data.

## Evaluation and Comparison

To compare the performance of the SVM, logistic regression, and random forest models, the following evaluation metrics are used:

- Accuracy: The proportion of correctly classified images in the test set.
- Precision: The ability of the model to correctly classify positive instances.
- Recall: The ability of the model to identify all positive instances.
- F1 Score: The harmonic mean of precision and recall, providing a balanced measure.

These metrics will be calculated for each model based on their predictions on the test set. By comparing the scores, we can determine which model performs better for the image classification task.

## Usage

To reproduce the results or apply these models to new images, follow these steps:

1. Clone the repository:
   git clone https://github.com/ApoorvSaxena13/Image-Classification-using-Machine-Learning

3. Prepare the dataset:
   - Ensure that the dataset contains labeled images of Lionel Messi, Maria Sharapova, Roger Federer, and Serena Williams, properly split into a training set and a test set.
   - Update the file paths in the code to point to the appropriate locations of the dataset.

4. Feature extraction:
   - Use appropriate techniques to extract meaningful features from the images. Common approaches include using pre-trained convolutional neural networks (CNNs) and extracting features from intermediate layers or using OpenCV and Wavelet Function.

5. Train the models:
   - Train SVM, logistic regression, and random forest models using the training set and the extracted features. Adjust the hyperparameters as needed.

6. Evaluate the models:
   - Use the trained models to predict the labels of the images in the test set.
   - Calculate the accuracy, precision, recall, and F1 score for each model.

7. Compare the scores:
   - Compare the performance of the models based on the evaluation metrics.
   - Analyze which model provides the highest accuracy and best trade-off between precision and recall.

## Conclusion

This project demonstrates the use of machine learning models, namely SVM, logistic regression, and random forest, for image classification of Lionel Messi, Maria Sharapova, Roger Federer, and Serena Williams. By evaluating the models on a test set and comparing their scores, we can determine which model performs better for the image classification task.
