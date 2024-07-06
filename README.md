
# Adult Census Income Prediction

## Introduction

This repository contains the code and documentation for predicting adult census income using various machine learning algorithms. The dataset used in this project contains information about individuals such as age, education, occupation, and more, and the task is to predict whether an individual earns more than $50,000 annually.



## Technology Used
<img src="https://github.com/MDS53/Adultcensusincome-_/assets/82602774/4a1d188d-a4bc-4d25-881d-80e6db331914" alt="Vscode" width="200"/>

<img src="https://github.com/MDS53/Adultcensusincome-_/assets/82602774/4f0730f0-0f7f-4536-af72-ca89802f8b77" alt="Python Logo" width="200"/>

<img src="https://github.com/MDS53/Adultcensusincome-_/assets/82602774/b5ab334c-4f77-4e07-9556-99c6874022e1" alt="Jupyter notebook" width="200"/>

<img src="https://github.com/MDS53/Adultcensusincome-_/assets/82602774/df040526-5e3f-4309-92a2-b092e0bfea9f" alt="Pandas" width="200"/>

<img src="https://github.com/MDS53/Adultcensusincome-_/assets/82602774/708ee0a7-f846-494b-b1ec-a22fa1bf572e" alt="Numpy" width="200"/>

<img src="https://github.com/MDS53/Adultcensusincome-_/assets/82602774/9ff1d913-4cd5-4411-a6b4-0cd66eaf6acb" alt="Sklearn icon" width="200"/>

<img src="https://github.com/MDS53/Adultcensusincome-_/assets/82602774/7b006b84-93e2-4edc-b3ff-bd0d0672b2f5" alt="Flask icon" width="200"/>

<img src="https://github.com/MDS53/Adultcensusincome-_/assets/82602774/05210c5c-bbec-4efa-a6c6-f859b2fa9121" alt="Docker icon" width="200"/>

####
<img src="https://github.com/MDS53/Adultcensusincome-_/assets/82602774/a7812838-6e90-4ecb-8122-db94b0c76a33" alt="Render icon" width="200"/>



  

## Dataset Description

The dataset used in this project is the Adult Census Income dataset, which is publicly available and often used for classification tasks. It contains various demographic features of individuals such as age, education, marital status, occupation, etc., along with the target variable indicating whether an individual earns more than $50,000 annually.

## Approach Overview

### Handling Missing Values

The dataset contains many missing values, which could potentially affect the performance of the machine learning models. To handle these missing values, a novel approach was adopted. Instead of imputing missing values or dropping records with missing values, an algorithm was used to handle them.

Since using the normal mode of handling missing values could lead to an imbalanced dataset, it was decided to split the dataset into training and testing sets based on the presence of missing values. Records with missing values were kept aside for testing, while records with non-missing values were used for training the models.

### Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was performed to gain insights into the dataset, understand the distributions of features, identify patterns, and detect outliers. Various statistical methods and visualizations were used to explore the data and understand its characteristics.

### Feature Engineering

Feature engineering plays a crucial role in building effective machine learning models. In this project, feature engineering techniques such as one-hot encoding were applied to categorical variables. The `ColumnTransformer` and `Pipeline` were used to efficiently preprocess the data and incorporate feature engineering into the machine learning pipeline.

### Model Building

Several machine learning algorithms were applied to build predictive models for the task of income prediction. The following algorithms were used:

- DecisionTreeClassifier
- LogisticRegression
- CatBoostClassifier
- XGBClassifier
- RandomForestClassifier

## Results

After evaluating the performance of each model, it was found that the CatBoostClassifier achieved the highest accuracy of 0.8732725969904801 on the test dataset. This accuracy score indicates how well the model predicts whether an individual earns more than $50,000 annually based on the given features.

## Conclusion

This project demonstrates an approach to predict adult census income using machine learning algorithms while effectively handling missing values and incorporating feature engineering techniques. The CatBoostClassifier emerged as the best-performing model, achieving a high accuracy score on the test dataset. The code and documentation provided here can serve as a reference for similar classification tasks involving demographic data.

## Repository Structure

- `artifacts/`: This directory stores all outputs generated during the project, including:
  - `model.pkl`: Serialized machine learning model.
  - `preprocessor.pkl`: Serialized preprocessor for data transformation.
  - `raw.csv`: Raw dataset file.
  - `train.csv`: Processed training dataset file.
  - `test.csv`: Processed testing dataset file.

- `src/`: This directory contains the source code for various components of the project:
  - `components/`: Contains individual components used in the machine learning pipeline.
  - `pipeline/`: Main pipeline file orchestrating the data preprocessing, model training, and evaluation.
  - `utils/`: Utility functions used across the project.
  - `template/`: Templates for the web application:
    - `form.html`: HTML template for input form.
    - `result.html`: HTML template for displaying prediction results.
    - `admin.html`: HTML template for displaying model information.
  - `Dockerfile`: Dockerfile for containerizing the application.
  - `app.py`: Flask application code for serving the machine learning model.
  - `setup.py`: Setup script to download the repository as a package.


## Project working demo: 👇👇👇:

https://github.com/MDS53/Adultcensusincome-_/assets/82602774/4dce71bf-c4c7-4ac9-8c15-12a5fc89d55c


## Usage

1. Clone this repository to your local machine.
2. Navigate to the `src/` directory.
3. Install the required dependencies listed in `requirements.txt`.
4. Run the Flask application by executing `python app.py`.
5. Access the application through your web browser.
6. Alternatively, use the Dockerfile to build a Docker container for the application. You can pull the Docker image from Docker Hub using the command: docker pull mds2019/adultbhai



## Docker Hub Link:
[mds2019/adultbhai](https://hub.docker.com/r/mds2019/adultbhai)

This will pull the Docker image containing the pre-built application environment and dependencies, allowing you to run the application in a containerized environment.


Try out the app link 
## Deployed app: https://adultcensusincomeprediction-9oyw.onrender.com


