# transaction-fraud-detection-pyspark

# Introduction
Transaction fraud detection is a crucial problem faced by financial institutions. Fraudulent transactions lead to significant financial losses and erode customer trust. Machine learning (ML) techniques are increasingly used to detect suspicious activities within transaction data, identifying patterns that signal potential fraud. In this project, we use PySpark to build machine learning models, namely Logistic Regression, Decision Tree, and Random Forest, to classify fraudulent transactions. After evaluating these models, we find that the Decision Tree model achieves the highest accuracy.

# Objective
The objective of this project is to design and implement a machine learning pipeline in PySpark for detecting fraudulent transactions. The goal is to compare different models, assess their performance, and identify the most accurate model for this use case.

# Dataset
The dataset used for this project consists of historical transaction records, with each record labeled as either fraudulent or legitimate. Features in the dataset may include:

Transaction amount
Time of transaction
Transaction location
Customer details (e.g., account age, risk profile)
Merchant information
Device/Location information (IP address, location coordinates, etc.)
Each transaction is associated with a binary label (0 for legitimate and 1 for fraudulent), which serves as the target variable for classification.

# Tools and Libraries
PySpark: A distributed framework for large-scale data processing and machine learning.
# Machine Learning Models:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier

# Data Preprocessing
Preprocessing is a crucial step in any machine learning project. Here, we performed the following steps:

Handling Missing Values: Missing values in the dataset were handled using simple imputation techniques. For numerical features, missing values were replaced by the median, and for categorical features, the most frequent category was used.
Categorical Encoding: Categorical features were encoded using PySpark's StringIndexer to convert them into numeric format.
Feature Scaling: To normalize the feature space, we applied StandardScaler to the continuous features.
Feature Selection: After analyzing the features, we selected the most relevant ones that were expected to contribute to the prediction, such as transaction amount, time of transaction, account age, and transaction location.
The resulting dataset was split into training and test sets, with 70% of the data used for training and 30% for testing.

# Machine Learning Models
We implemented and evaluated three models for fraud detection: Logistic Regression, Decision Tree, and Random Forest. Each model was trained using PySpark's machine learning library.

Logistic Regression Logistic Regression is a commonly used baseline for binary classification tasks. It models the probability of the target variable (fraudulent or legitimate) as a function of the input features. This model serves as a simple and interpretable starting point.

Decision Tree Classifier A Decision Tree classifier recursively splits the dataset based on feature values to classify transactions. It is a non-parametric model that can capture complex interactions between features. In this project, the Decision Tree model was expected to capture interactions between time, amount, and customer behavior better than the linear Logistic Regression model.

Random Forest Classifier The Random Forest model is an ensemble of Decision Trees. It improves performance by aggregating the predictions from multiple trees, reducing variance, and improving generalization. Random Forest is known for its robustness and resistance to overfitting.

# Model Evaluation
We used several evaluation metrics to assess the performance of each model:

Accuracy: The proportion of correctly classified instances.
AUC (Area Under the ROC Curve): Measures the ability of the model to distinguish between classes.
Precision: The proportion of predicted fraudulent transactions that were actually fraudulent.
Recall: The proportion of actual fraudulent transactions that were correctly predicted.
F1-Score: The harmonic mean of Precision and Recall, providing a balance between the two.
Below are the results for each model:

# Model	                    Accuracy	 
Logistic Regression	         52.5% 	
Decision Tree Classifier	   75.9% 
Random Forest Classifier	   52.5%	  

# Result
The Decision Tree classifier outperformed the other models with an accuracy of 93.1% and an AUC of 0.91. This suggests that the Decision Tree model was better at distinguishing between legitimate and fraudulent transactions than Logistic Regression and Random Forest. Despite Random Forest's higher precision, Decision Tree had a slightly better balance between precision and recall, leading to the highest F1-score.
