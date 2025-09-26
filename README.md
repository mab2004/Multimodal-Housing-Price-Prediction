# Multimodal Housing Price Prediction

## 🎯 Objective

The primary goal was to predict housing sales prices by fusing information from two distinct data modalities:
1.  **Tabular Data:** Structured house attributes (e.g., area, bedrooms).
2.  **Image Data:** Visual features extracted from house photos (e.g., kitchen, bedroom, frontal view).

The final model must be evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).


## 🛠️ Methodology / Approach

This solution employs a **dual-input neural network** built using TensorFlow/Keras Functional API, allowing for parallel processing of the two data types before fusion.

### 1. Data Source and Preprocessing
* **Dataset:** The **Houses Dataset** (by Ahmed and Moustafa, 2016) was used, which contains 535 house listings with both textual attributes and four corresponding images each.
* **Data Access:** The dataset was cloned directly into the execution environment (Google Colab) using `git clone`.
* **Tabular Data:** Features (`bedrooms`, `bathrooms`, `area`, `zipcode`) were loaded and scaled using `MinMaxScaler`.
* **Image Data:** The four images per house (`frontal`, `bedroom`, `bathroom`, `kitchen`) were resized to (64x64) and **tiled** into a single (128x128x3) input image. Images were normalized to the range [0, 1].

### 2. Model Architecture (Feature Fusion)
The model consists of two branches:
* **Image Branch (CNN):** A custom Convolutional Neural Network with two `Conv2D` and `MaxPooling2D` layers, followed by a dense layer, to extract visual features.
* **Tabular Branch (MLP):** A small Multi-Layer Perceptron (MLP) with two dense layers to process the structured features.
* **Fusion Layer:** The outputs of the CNN and the MLP were concatenated (`concatenate`) to form a combined feature vector, which was then passed to a final regression head for price prediction. 

### 3. Training and Evaluation
* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (`loss='mean_squared_error'`)
* **Epochs:** 25
* **Metric:** The final prediction accuracy was measured using MAE and RMSE on the held-out test set (25% of the data).


## 📊 Key Results or Observations

The model was trained and evaluated on the test set, yielding the following results:

| Metric | Value |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | **$367,638.51** |
| **Root Mean Squared Error (RMSE)** | **$629,051.61** |

### Observations:
1.  **Feasibility Confirmed:** The results successfully validate the multimodal methodology, proving that the two disparate data types can be effectively combined to perform a complex regression task.
2.  **Performance Baseline:** The high error values (relative to typical house prices) establish a baseline. This suggests the model requires further optimization, such as using **transfer learning** (e.g., VGG16) for image feature extraction, and better **categorical encoding** for the `zipcode` feature.
3.  **Future Work:** The next steps would involve fine-tuning the model using a pre-trained CNN and applying more robust feature engineering to improve predictive accuracy.


### Skills Demonstrated
* Multimodal Machine Learning
* Convolutional Neural Networks (CNNs)
* Feature Fusion (Image + Tabular)
* Regression Modeling and Evaluation

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](LICENSE) file for details.

