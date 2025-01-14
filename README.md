# Titanic: Machine Learning from Disaster

This project is a solution to the famous **Kaggle competition**: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic). The goal is to predict the survival of passengers on the Titanic using machine learning techniques. The dataset contains information about passengers, such as their age, gender, class, and whether they survived or not.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Results](#results)
6. [How to Run](#how-to-run)

---

## Project Overview
The Titanic competition is a classic machine learning problem where the goal is to predict whether a passenger survived the Titanic disaster based on various features like age, gender, class, and more. This project walks through the entire process of data exploration, preprocessing, model training, and prediction.

---

## Dataset Description
The dataset is provided by Kaggle and consists of two files:
- **train.csv**: Contains the training data with survival labels.
- **test.csv**: Contains the test data without survival labels (used for submission).

The dataset includes the following features:
- **PassengerId**: Unique ID for each passenger.
- **Survived**: Survival status (0 = No, 1 = Yes).
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- **Name**: Passenger's name.
- **Sex**: Gender of the passenger.
- **Age**: Age in years.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Ticket**: Ticket number.
- **Fare**: Passenger fare.
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

---

## Data Preprocessing
The data preprocessing pipeline includes the following steps:
1. **Handling Missing Values**: Missing values in the `Age` column are filled with the mean value.
2. **Encoding Categorical Variables**: Categorical features like `Sex` and `Embarked` are one-hot encoded.
3. **Dropping Unnecessary Columns**: Columns like `Name`, `Ticket`, `Cabin`, and `PassengerId` are dropped as they are not useful for prediction.
4. **Stratified Sampling**: The dataset is split into training and testing sets while maintaining the proportion of survivors.


## Model Training
To predict passenger survival on the Titanic, a **RandomForestClassifier** was used. This model was chosen due to its ability to handle both numerical and categorical data effectively, as well as its robustness against overfitting. 

To optimize the model's performance, **hyperparameter tuning** was performed using **GridSearchCV**. The following hyperparameters were tuned:
- **n_estimators**: The number of trees in the forest. Values tested: 10, 100, 200, 500.
- **max_depth**: The maximum depth of each tree. Values tested: None, 5, 10.
- **min_samples_split**: The minimum number of samples required to split a node. Values tested: 2, 3, 4.

A **5-fold cross-validation** strategy was used during the grid search to ensure the model's generalizability. The evaluation metric used was **accuracy**, which measures the proportion of correctly predicted survival outcomes.

After training, the best-performing model was selected based on the highest cross-validation accuracy.

---

## Results
The final model achieved an **accuracy of 80.45%** on the test set, demonstrating strong predictive performance. The best hyperparameters found during the grid search were:
- **n_estimators**: 200
- **max_depth**: 5
- **min_samples_split**: 3

The model was then used to make predictions on the Kaggle test dataset. These predictions were saved in a CSV file (`predictions.csv`) for submission to the Kaggle competition. The file contains two columns:
- **PassengerId**: The unique identifier for each passenger in the test set.
- **Survived**: The predicted survival outcome (0 = Did not survive, 1 = Survived).

This solution provides a robust approach to predicting survival on the Titanic and can be further improved with additional feature engineering or model tuning.