MediCore AI – Heart Disease Prediction

<p align="center">
  <strong>A Machine Learning project using Decision Tree Classification to explore heart disease prediction.</strong>
</p><p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Flet-Interactive%20UI-green" />
  <img src="https://img.shields.io/badge/Project-Educational-informational" />
</p>---

Overview

MediCore AI is a Heart Disease Prediction project developed using Python and Machine Learning.

The project uses a Decision Tree Classifier trained on the UCI Heart Disease dataset to classify input data into two categories:

- 0 – No Heart Disease
- 1 – Heart Disease

It combines a complete machine learning workflow with an interactive user interface where users can enter input features and view the model's prediction.

This project demonstrates practical applications of supervised learning, data preprocessing, exploratory data analysis, model optimization, and model evaluation.

Demo

<!-- Add your recorded animated GIF here --><p align="center">
  <img src="assets/demo.gif" alt="MediCore AI Application Demo" width="850"/>
</p>Features

- UCI Heart Disease dataset integration.
- Data cleaning and missing value handling.
- Exploratory Data Analysis (EDA).
- Data preprocessing using Scikit-learn pipelines.
- Decision Tree classification.
- Hyperparameter tuning using GridSearchCV.
- Model evaluation and performance metrics.
- Confusion Matrix and ROC Curve visualizations.
- Feature Importance analysis.
- Interactive Flet user interface.
- Patient data input and prediction.
- Terminal-based prediction testing.
- Saved model for reuse without retraining.

Technologies Used

Technology| Purpose
Python| Main programming language
Pandas| Data manipulation
NumPy| Numerical operations
Scikit-learn| Machine Learning
UCI ML Repository| Dataset source
Matplotlib| Data visualization
Seaborn| Statistical visualization
Joblib| Model persistence
Flet| User interface

Dataset

Source: UCI Machine Learning Repository – Heart Disease Dataset

Dataset ID: "45"

The dataset contains clinical features used for heart disease classification.

Input Features

Feature| Description
age| Age
sex| Sex
cp| Chest pain type
trestbps| Resting blood pressure
chol| Serum cholesterol
fbs| Fasting blood sugar
restecg| Resting ECG results
thalach| Maximum heart rate
exang| Exercise-induced angina
oldpeak| ST depression
slope| Slope of peak exercise ST segment
ca| Number of major vessels
thal| Thalassemia result

Target Transformation

The original target contains values from 0 to 4.

For this project, it is transformed into binary classification:

- "0" → No Heart Disease
- "1, 2, 3, 4" → Heart Disease

Machine Learning Workflow

The project follows these steps:

1. Data Collection: Retrieve the dataset using "ucimlrepo".
2. Data Cleaning: Convert data types and handle missing target values.
3. Exploratory Data Analysis: Explore distributions, correlations, and missing values.
4. Preprocessing: Apply median imputation to numerical features and most-frequent imputation with One-Hot Encoding to categorical features.
5. Train-Test Split: Split the dataset into 80% training and 20% testing data using stratification.
6. Model Training: Train a Decision Tree Classifier.
7. Hyperparameter Tuning: Use GridSearchCV with cross-validation to select model parameters.
8. Evaluation: Evaluate the model using classification metrics and visualizations.
9. Model Saving: Save the trained pipeline using Joblib.
10. Prediction: Load the saved model and predict new input data.

Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

The training script also generates visualizations such as:

- Target Distribution
- Feature Distributions
- Correlation Heatmap
- Missing Values Heatmap
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance
- Decision Tree Visualization
- Training vs Testing Accuracy

Actual performance values are generated when the training script runs and are displayed in the application dashboard.

Application Interface

The application is built using Flet and includes four main sections:

1. Dashboard

- Model performance metrics.
- Training and testing accuracy.
- Project overview.

2. Prediction

- Input fields for the 13 dataset features.
- Prediction result.
- Model confidence score.

3. Model Analysis

- Model evaluation charts.
- Feature importance.
- Decision Tree visualization.

4. About Project

- Algorithm explanation.
- Dataset information.
- Project details.

Project Structure

MediCore-AI/
│
├── ml_model.py
├── ui.py
├── terminal_test.py
│
├── models/
│   ├── decision_tree_model.pkl
│   └── model_info.pkl
│
├── target_distribution.png
├── correlation_heatmap.png
├── confusion_matrix.png
├── roc_curve.png
├── precision_recall_curve.png
├── feature_importance.png
├── decision_tree.png
├── train_test_accuracy.png
├── feature_distributions.png
├── missing_values.png
│
├── assets/
│   └── demo.gif
│
├── requirements.txt
└── README.md

Installation

1. Clone the repository

git clone YOUR_REPOSITORY_URL

2. Open the project folder

cd MediCore-AI

3. Install dependencies

pip install pandas numpy scikit-learn matplotlib seaborn joblib ucimlrepo flet

How to Run

Step 1 – Train the Model

Run:

python ml_model.py

This will:

- Load the dataset.
- Perform preprocessing.
- Train and evaluate the Decision Tree model.
- Generate visualization files.
- Save the trained model inside the models folder.

Step 2 – Launch the User Interface

Run:

python ui.py

The MediCore AI interface will launch.

Step 3 – Terminal Prediction

Run:

python terminal_test.py

Enter the requested feature values in the terminal to test the saved model.

Model Information

Algorithm: Decision Tree Classifier

Learning Type: Supervised Learning

Problem Type: Binary Classification

Dataset: UCI Heart Disease

Train-Test Split: 80% Training / 20% Testing

Hyperparameter Optimization: GridSearchCV

Preprocessing: Scikit-learn Pipeline

Limitations

- The dataset has limited size and may not represent every population.
- Model performance depends on the quality and distribution of the training data.
- Predictions on unseen data may differ from training results.
- The model has not been validated for clinical use.

Future Improvements

- Compare Decision Tree with other classification algorithms.
- Improve model explainability.
- Add more evaluation techniques.
- Improve the user interface.
- Test the model on additional datasets.
- Add automated testing.

Disclaimer

This project is developed for educational and research purposes only.

It is not a medical diagnostic tool and must not be used to make healthcare decisions. Model predictions and confidence scores are experimental outputs, not confirmed medical probabilities.

Acknowledgements

- UCI Machine Learning Repository
- Scikit-learn Documentation
- Python Data Science Community

---

<p align="center">
  <strong>Built with Python, Machine Learning, and Flet.</strong>
</p>
