## **Project Overview**

The objective of this project is to create a deep learning model capable of accurately classifying grape leaf diseases. This is achieved by employing a hybrid model that combines Vision Transformer (ViT) and MobileNetV2 architectures. The dataset used consists of high-resolution images of grape leaves, labeled according to disease type or healthy status. By integrating the capabilities of both ViT and MobileNetV2, the project aims to enhance the model's ability to detect and classify different grape leaf diseases more accurately than traditional methods.

## **Introduction**

Grapevine diseases can significantly affect the quality and quantity of grape production, making early detection crucial for effective management. Manual inspection methods for disease detection are often labor-intensive and error-prone. To address this challenge, this project leverages modern deep learning techniques to automate the classification of grape leaf diseases. By combining Vision Transformer (ViT) and MobileNetV2, the project seeks to develop a robust and efficient model that can quickly and accurately identify diseases from leaf images.

## **Data Collection and Preparation**

This section describes how the data was gathered and prepared for model training and evaluation.

### **Key Steps:**

- **Sourcing the Data:** Images were collected from various agricultural research databases and online repositories. These sources provide a diverse range of images that represent different types of grape leaf diseases.
  
- **Dataset Composition:** The dataset includes thousands of images, each labeled according to the disease type (Leaf Blight, Esca, Black Rot) or healthy status. This ensures a comprehensive coverage of possible conditions that the model needs to identify.

- **Data Splitting:** To ensure the model is trained effectively and evaluated fairly, the dataset was split into three subsets:
  - **Training Set (70%):** Used to train the model, allowing it to learn and adjust its parameters based on the data.
  - **Validation Set (15%):** Used to tune the model’s hyperparameters and assess its performance during training.
  - **Test Set (15%):** Used for the final evaluation of the model’s performance to verify its ability to generalize to new, unseen data.

## **Data Preprocessing**

Preprocessing ensures that the data is in a suitable format for training the model and improves overall model performance.

### **Steps Include:**

- **Image Resizing:** All images are resized to a consistent dimension to ensure they can be processed by the model without issues related to varying image sizes.

- **Color Inversion:** For images where the background color is similar to the leaf color, color inversion is applied to enhance contrast and make features more distinguishable.

- **Normalization:** Pixel values are scaled to a range of [0, 1] to standardize the input data and improve the model’s training stability.

- **Augmentation:** Techniques such as rotation, flipping, and cropping are used to artificially increase the diversity of the training set. This helps the model generalize better by exposing it to a wider variety of image conditions.

- ![Screenshot 2024-08-07 231431](https://github.com/user-attachments/assets/521a80ea-f6f3-4095-b7b4-4844f2e3cf3c)

![Screenshot 2024-08-07 231452](https://github.com/user-attachments/assets/c413167d-a108-40cd-860b-c448f5d4d294)

## **Model Setup**

The model setup involves configuring the architecture and components used for training.

### **Components:**

- **Vision Transformer (ViT):** This architecture processes images to capture global features and long-range dependencies. ViT is used here to handle the broader context of the images, capturing high-level patterns and relationships.

- **MobileNetV2:** This architecture focuses on local feature extraction, capturing detailed and fine-grained information from the images. MobileNetV2’s efficiency makes it suitable for this task.

- **Feature Fusion:** The features extracted from both ViT and MobileNetV2 are combined to create a unified feature vector. This fusion allows the model to leverage both global and local information for more accurate classification.

- **Classification Layer:** A dense layer with softmax activation is used to convert the combined feature vector into class probabilities. This layer predicts the likelihood of each class (disease or healthy) for the input image.

## **Model Construction**

This section covers the actual building of the model, integrating ViT and MobileNetV2.

### **Key Steps:**

- **Feature Extraction:** ViT and MobileNetV2 each extract different types of features from the input images. ViT captures global context, while MobileNetV2 focuses on local details.

- **Feature Fusion:** Features from both models are concatenated to form a comprehensive representation of the input images. This fusion helps in capturing a wide range of information.

- **Classification Layer:** The concatenated features are passed through a dense layer with a softmax activation function to output the predicted class probabilities.

## **Model Compilation**

Compiling the model involves configuring it for training with the appropriate settings.

### **Settings:**

- **Optimizer:** The Adam optimizer is used for its effectiveness in training deep learning models. It adjusts learning rates dynamically to improve convergence.

- **Loss Function:** Categorical Cross-Entropy is used to measure the difference between the predicted probabilities and the actual class labels. This is appropriate for multi-class classification tasks.

- **Metrics:** Accuracy, precision, recall, and F1 score are monitored to evaluate how well the model performs. These metrics provide insight into different aspects of the model’s performance.

## **Model Training**

Training involves feeding data into the model and optimizing it.

### **Steps:**

- **Epochs and Batch Size:** The number of epochs (iterations over the entire dataset) and batch size (number of images processed at a time) are set to balance training efficiency and memory usage.

- **Learning Rate Schedule:** Adjustments to the learning rate during training help in achieving better convergence and avoiding overshooting.

- **Training Procedure:** The model is trained by feeding batches of training images, computing gradients, and updating weights through backpropagation. Validation performance is monitored to adjust hyperparameters and prevent overfitting.

## **Model Evaluation**

Evaluating the model’s performance on the test set provides an understanding of how well it generalizes.

### **Metrics:**

- **Accuracy:** The percentage of correctly classified images in the test set.

- **Confusion Matrix:** A detailed breakdown of true positives, false positives, true negatives, and false negatives for each class.

- **Precision, Recall, F1 Score:** These metrics are calculated for each class to assess the model’s performance in detecting specific diseases.

- **Cross-Validation:** If applicable, k-fold cross-validation ensures that the model’s performance is robust and not overly dependent on a single data subset.

- ![Screenshot 2024-08-07 231713](https://github.com/user-attachments/assets/8ca44801-aea3-48bb-8e52-d85d3d040640)


## **Predictions and Results Analysis**

This section involves analyzing the results and understanding the model’s performance.

![download (3)](https://github.com/user-attachments/assets/f1aa8e6b-e7ef-4950-9636-b32a20da475f)

![download (4)](https://github.com/user-attachments/assets/33cbc867-7c35-49aa-8faf-57d03eb7affb)

### **Analysis Includes:**

- **Test Set Predictions:** Analysis of the model’s predictions on the test set to determine classification accuracy.

- **Visualization:** Visualization of example predictions to qualitatively assess model performance.

- **Error Analysis:** Examination of misclassified images to identify common patterns or areas for improvement.

- **Performance Comparison:** Comparing the hybrid model’s performance with that of individual ViT and MobileNetV2 models to highlight the benefits of combining the two approaches.
![download (5)](https://github.com/user-attachments/assets/7ba9a947-6eac-4fb0-b37b-6947499e37d0)

AUC scores:
Class 1: 0.9998051821546854
Class 2: 0.9998239940157966
Class 3: 1.0
Class 4: 1.0
