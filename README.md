# Classification of Historical Sites on Penyengat Island with MobileNetV2

## Introduction

This project aims to develop an image classification model using the MobileNetV2 architecture to identify 7 historical sites on Penyengat Island, Riau Islands Province, Indonesia. The dataset used is a combination of images taken directly at the location and additional data from the internet.

## Table of Contents

1.  [About Penyengat Island](#about-penyengat-island)
2.  [Dataset](#dataset)
3.  [Data Preprocessing](#data-preprocessing)
4.  [Model Architecture](#model-architecture)
5.  [Model Training](#model-training)
6.  [Results and Analysis](#results-and-analysis)
7.  [Important Files](#important-files)
8.  [How to Use](#how-to-use)
9.  [Contribution](#contribution)
10. [License](#license)

---

## 1. About Penyengat Island

Penyengat Island is a small island located in Tanjungpinang City, Riau Islands Province. It holds significant historical value, particularly as the center of government and culture for the Riau-Lingga Sultanate in the past. Many important historical sites still stand on the island, such as mosques, tombs of kings and important figures, and traditional communal halls.

---

## 2. Dataset

The dataset used is divided into two main parts:

* **`dataset-training-500`**: Contains original images taken directly on Penyengat Island, then augmented to 500 images per class. This dataset is used for model training.
* **`dataset-valtest`**: Contains new image data collected from the internet for each class. This dataset is then further split into validation data (50%) and test data (50%).

Total data used:
* **Training**: 3501 images
* **Validation**: 350 images
* **Testing**: 350 images
* **Number of Classes**: 7 classes (each class represents a specific historical site or building).

The identified classes in the dataset are:
* `balai-adat-melayu`
* `gedung-tabib`
* `makam-engku-putri`
* `makam-raja-ali-haji`
* `masjid-raya-sultan-riau`
* `meriam-bukit-kursi`
* `rumah-hakim`

---

## 3. Data Preprocessing

Before training, images are preprocessed using `ImageDataGenerator` from TensorFlow.

* **Image Size**: All images are resized to $224 \times 224$ pixels.
* **Batch Size**: $32$.
* **Normalization**: Pixel values are rescaled from $0-255$ to $0-1$ (`rescale=1./255`).
* **Data Augmentation (for training data)**:
    * `rotation_range`: Rotates images up to 10 degrees.
    * `zoom_range`: Zooms images in/out by 10%.
    * `brightness_range`: Adjusts image brightness within a range of 85% to 115%.
    * `width_shift_range` & `height_shift_range`: Shifts images horizontally and vertically up to 5%.
    * `horizontal_flip`: Flips images horizontally.
    This augmentation aims to increase image variability and prevent *overfitting*.

---

## 4. Model Architecture

The model uses a *transfer learning* approach with **MobileNetV2** as the *base model*.

* **Base Model**: `MobileNetV2` pretrained on the ImageNet dataset (`weights='imagenet'`) and without the classification head (`include_top=False`).
* **Input Shape**: $224 \times 224 \times 3$ (height, width, RGB color channels).
* **Layer Freezing**: A portion of the `MobileNetV2` layers are frozen to retain the basic features learned. In this project, 30% of the initial layers are frozen and will not be re-trained (`FROZEN_PERCENT = 0.3`).
* **Additional Classifier**:
    * After the feature extraction part of `MobileNetV2`, a `GlobalAveragePooling2D()` layer is added to reduce dimensions.
    * Then, a `Dense` layer is added as the output layer with the number of classes matching the dataset (7 classes), using the `softmax` activation function for multi-class classification.
* **Model Compilation**:
    * **Optimizer**: `Adam` with an initial `learning_rate` of `1e-6` (very small for fine-tuning).
    * **Loss Function**: `categorical_crossentropy` (suitable for multi-class classification).
    * **Metrics**: `accuracy`.

---

## 5. Model Training

The model is trained for 30 *epochs* using the training and validation data. Several *callbacks* are used during training for optimization:

* **`ModelCheckpoint`**: Saves the best model based on `val_loss` to the path `/content/drive/MyDrive/modelku/bocchichan_model_best.h5`. The model is only saved if the `val_loss` is better than before.
* **`ReduceLROnPlateau`**: Reduces the `learning_rate` if `val_loss` does not improve after 2 *epochs* (`patience=2`). The `learning_rate` will be halved (`factor=0.5`), with `min_lr=1e-7`. This callback helps the model escape *local minima* and accelerates convergence.

After training is complete, the model is saved in two versions:
* **`bocchichan_model_full.h5`**: The complete model, including the *optimizer*, for continued training.
* **`bocchichan_model_inference.h5`**: The model without the *optimizer*, ready for *deployment* or *inference*.

---

## 6. Results and Analysis

### Classification Report
```
                         precision    recall  f1-score   support
balai-adat-melayu           0.87      0.52      0.65        50
gedung-tabib                0.75      0.82      0.78        50
makam-engku-putri           0.92      0.90      0.91        50
makam-raja-ali-haji         0.89      0.96      0.92        50
masjid-raya-sultan-riau     0.81      0.94      0.87        50
meriam-bukit-kursi          0.82      0.98      0.89        50
rumah-hakim                 0.89      0.78      0.83        50

accuracy                                        0.84       350
macro avg                   0.85      0.84      0.84       350
weighted avg                0.85      0.84      0.84       350
```

### Analysis of Results

* **Accuracy**: The model achieved an overall accuracy of 84% on the test data, indicating that most of its predictions were correct.

* **Precision, Recall, and F1-Score per Class**:
    * **Precision**: The `makam-engku-putri` (0.92) and `makam-raja-ali-haji` (0.89) classes have very high precision, meaning the model is highly accurate in predicting images from these classes.
    * **Recall**: The `meriam-bukit-kursi` (0.98) and `masjid-raya-sultan-riau` (0.94) classes show high recall, meaning the model rarely misses actual images from these classes.
    * **F1-Score**: The `makam-engku-putri` (0.91) and `makam-raja-ali-haji` (0.92) classes have high F1-scores, indicating a good balance between precision and recall for these classes.

* **Macro Avg and Weighted Avg**:
    * `Macro Avg` (average per class without considering class size) resulted in 0.84 for recall and 0.85 for precision.
    * `Weighted Avg` (average considering the number of examples per class) yielded similar results (0.84 for recall and 0.85 for precision). This indicates a balanced class distribution in the dataset.

* **Insights and Required Improvements**:
    * While the overall model performance is quite good, there is one class with a relatively low `recall`: `balai-adat-melayu` (0.52). This suggests that the model often fails to identify images that actually belong to this class.
    * To improve performance for the `balai-adat-melayu` class, it is recommended to add more specific training data for this class or apply more aggressive data augmentation techniques.

---

## 7. Important Files

* `training-model.ipynb`: Jupyter Notebook containing all the code for data preprocessing, model building, training, and evaluation.
* `bocchichan_model_best.h5`: The best model saved during training (including weights, without optimizer).
* `bocchichan_model_full.h5`: The complete model saved after training (including weights and optimizer).
* `bocchichan_model_inference.h5`: The model saved for deployment (weights only, without optimizer).

---

## 8. How to Use

To run and replicate this project:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Nuswapada/training_model_ai.git
    cd training_model_ai
    ```
2.  **Prepare the Dataset**: Ensure you have the dataset structured as described in the [Dataset](#dataset) section, with `dataset-training-500` and `dataset-valtest` folders in the appropriate location (`/content/drive/MyDrive/dataset-gila/`).
3.  **Open the Notebook in Google Colab**: Upload `training-model.ipynb` to Google Colab.
4.  **Mount Google Drive**: Follow the steps in the notebook to mount your Google Drive so the dataset can be accessed.
5.  **Run Notebook Cells**: Execute the cells in sequence within the notebook to preprocess data, build and train the model, and evaluate the results.

---

## 9. Contribution

Contributions in the form of *bug reports*, *feature requests*, or *pull requests* are highly welcome. Please feel free to open an *issue* or create a *pull request* in this repository.

---

## 10. License

This project is licensed under the NUSWAPADA TEAM

---
