# Bug Images Data Analysis And Type Classification
<table>
  <tr>
    <td>
      <h2>Project Introduction</h2>
      <p>This project, <strong>"To Bee or Not to Bee"</strong>, focuses on applying data analysis and machine learning techniques to identify pollinator insects (like bees and bumblebees) from other species. The dataset consists of 347 high-resolution images with corresponding segmentation masks, precisely delineating insects and enabling us to extract relevant features. The first 250 images are associated with a label, which means that we can recursively train and evaluate our models, while the remaining 97 don’t and will be used to evaluate the performance of our models.</p>
    </td>
      <td>
      <img src="train/122.JPG" alt="Example of Bug Image" width="5000">
    </td>
  </tr>
</table>

### Goals

- Detect and classify pollinator insects (bees, bumblebees, and other insects) based on key features extracted from images.
- Build and evaluate machine learning models using both supervised and unsupervised methods to classify the insects.
- Analyze the impact of feature extraction, data augmentation, and dimensionality reduction techniques on model performance.

---


## Features & Functionality

- **Feature Extraction**: Extract features such as color, shape, and texture from insect images to use as input for machine learning algorithms.
- **Data Augmentation**: Handle data imbalance with techniques like SMOTE and ADASYN to generate synthetic data for underrepresented classes.
- **Machine Learning Models**: Implement a variety of models, including logistic regression, support vector machines, K-Nearest Neighbors, and neural networks for insect classification.
- **Dimensionality Reduction**: Utilize PCA and t-SNE to project features into lower dimensions for visualization and analysis.

---


## Project Structure

- **Root Folder**: Contains Jupyter notebooks documenting the different steps of the project. Each file is named according to the step it corresponds to:
  - `1_data_extraction.ipynb`: Data extraction and cleaning.
  - `2_feature_extraction.ipynb`: Code for extracting color, shape, and texture-based features.
  - `3_modeling.ipynb`: Includes the machine learning and deep learning models implemented.
  - `tools.py`: A collection of helper functions used across multiple steps of the project.

- **`train/` Folder**:
  - Contains the training images of bees and other insects.
  - **`masks/` subfolder**: Contains the `.tif` files for each bee image mask.

- **`models/` Folder**:
  - Saved hypertuned and trained machine learning models (`.joblib` format) for reuse.

- **`data/` Folder**:
  - `classif.xlsx`: Original dataset including the bug type and species for each image.
  - `processed_data.csv`: Processed data with extracted features and target labels for modeling.

---


## Technologies Used

- **Python**: Main programming language for data preprocessing, analysis and modeling.
- **Jupyter Notebooks**: For documenting the steps and visualizations in an interactive manner.
- **Pandas**: Used for data manipulation, feature engineering, and exploratory data analysis.
- **OpenCV**: Leveraged to extract features from the insect images (image processing).
- **Scikit-Image**: For advanced image processing tasks like contour detection.
- **Scikit-learn**: Utilized for building traditional machine learning models (logistic regression, SVM, KNN, RF, etc.) and performing dimensionality reduction techniques (PCA, t-SNE, ISOMAP).
- **XGBoost & LightGBM**: Implemented for building ensemble learning models such as gradient-boosted decision trees.
- **SMOTE/ADASYN**: Used for addressing the huge class imbalance of the dataset by generating synthetic samples of minority classes (oversampling).
- **Matplotlib & Seaborn**: Libraries for visualizing various features, correlations, and model performance metrics.
- **Hyperopt**: Used for hyperparameter optimization of the machine learning models which proved effective.
- **PyTorch**: Deep learning library used for implementing neural networks, including custom architectures.
- **SHAP**: For model interpretability and to explain feature importance in complex models.

---


### Final Algorithms Used for Prediction

- **Logistic Regression**: A simple linear model used for classification, showing excellent performance on the dataset.
- **Support Vector Machines (SVM)**: Both RBF and Polynomial kernels were tested for classification.
- **K-Nearest Neighbors (KNN)**: A distance-based algorithm used for classification, that gave decent, but not great, results.
- **Random Forest**: An ensemble learning method used for combining decision trees.
- **Extra Trees**: Another ensemble method that provided slightly better results than Random Forest.
- **XGBoost & LightGBM**: Gradient-boosted tree algorithms that got the best results out of all algorithms tested.
- **Neural Networks**: Deep learning models, built with custom architectures, used for classifying insect images.
- **Stacking Classifier**: An ensemble of the previous models combined to improve prediction accuracy, with a voting system.






