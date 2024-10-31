#!/usr/bin/env python
# coding: utf-8

# # References : 
# 
# Main data pre-processing and data cleaning, comparision of randomforests with other models
# ### https://www.kaggle.com/code/jieyima/income-classification-model
# 
# 
# 
# Random forest training, confusion matrix, ROC curves, visualization, feature importance etc
# ### https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/Random%20Forest%20Tutorial.ipynb
# 
# 
# ### https://nthu-datalab.github.io/ml/labs/03_Decision-Tree_Random-Forest/03_Decision-Tree_Random-Forest.html
# 

# ### Step 1 : Import necessary libraries

# In[1]:


from tqdm import tqdm
import psutil
import time
import math
import os
from subprocess import call

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import Image, SVG, HTML

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.tree import export_graphviz
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree


#import warnings
#import kagglehub
#from sklearn.linear_model import Perceptron
#from sklearn.svm import SVC
### cross validation

#from sklearn.model_selection import cross_val_predict #prediction
#from sklearn.cluster import KMeans

#from sklearn.metrics import silhouette_samples
#from sklearn.metrics import silhouette_score
#from sklearn.metrics import accuracy_score

#from sklearn.decomposition import PCA
#from pandas.tools.plotting import scatter_matrix
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.model_selection import GridSearchCV

#importing all the required ML packages
#from sklearn import metrics #accuracy measure
RSEED = 100


# ### Step 2: Data Cleaning functions
# 
# The reference code contained lot of individual blocks, I condensed them all into one single function

# In[2]:


#This function cleans data before EDA
def clean_data(df):
    df = df.copy() 
    df.age = df.age.astype(float)
    df['hours-per-week'] = df['hours-per-week'].astype(float)
    df = df.dropna()
    df['predclass'] = df['income']
    df = df.drop(columns=['income'])
    df['education-num'] = df['educational-num']
    df = df.drop(columns=['educational-num'])
    
    education_replacements = {
        'Preschool': 'dropout', '10th': 'dropout', '11th': 'dropout', '12th': 'dropout',
        '1st-4th': 'dropout', '5th-6th': 'dropout', '7th-8th': 'dropout', '9th': 'dropout',
        'HS-Grad': 'HighGrad', 'HS-grad': 'HighGrad',
        'Some-college': 'CommunityCollege', 'Assoc-acdm': 'CommunityCollege', 'Assoc-voc': 'CommunityCollege',
        'Bachelors': 'Bachelors',
        'Masters': 'Masters', 'Prof-school': 'Masters',
        'Doctorate': 'Doctorate'
    }
    df['education'] = df['education'].replace(education_replacements)
    
    marital_status_replacements = {
        'Never-married': 'NotMarried',
        'Married-AF-spouse': 'Married', 'Married-civ-spouse': 'Married',
        'Married-spouse-absent': 'NotMarried',
        'Separated': 'Separated', 'Divorced': 'Separated',
        'Widowed': 'Widowed'
    }
    df['marital-status'] = df['marital-status'].replace(marital_status_replacements)
    
    return df

# This function prepares data after EDA
# https://stackoverflow.com/questions/35576509/making-a-jupyter-notebook-output-cell-fullscreen
def prepare_data(df, sample_size=15000):
    # URL for the dataset
    #dataset_url = "https://archive.ics.uci.edu/ml/datasets/Adult"
    dataset_url = "https://www.kaggle.com/datasets/wenruliu/adult-income-dataset?resource=download"
    
    # Attributes and value ranges (obtained from commented out UCI link)
    attributes_info = {
        'age': 'continuous',
        'workclass': 'Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked',
        'fnlwgt': 'continuous',
        'education': 'Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool',
        'education-num': 'continuous',
        'marital-status': 'Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse',
        'occupation': 'Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces',
        'relationship': 'Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried',
        'race': 'White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black',
        'sex': 'Female, Male',
        'capital-gain': 'continuous',
        'capital-loss': 'continuous',
        'hours-per-week': 'continuous',
        'native-country': 'United-States, Cambodia, England, Puerto-Rico, Canada, Germany, ...',
        'income': '<=50K, >50K'
    }
    
    # Note: I am limiting the data to get low accuracies + faster training times
    df = df.sample(n=sample_size, random_state=RSEED)
    
    # Print dataset information before label encoding and scaling
    print("Dataset URL:", dataset_url)
    print("\nAttributes and Value Ranges:")
    for attr, val_range in attributes_info.items():
        print(f"{attr}: {val_range}")
    
    # Split the data into training and testing sets before encoding and scaling
    X = df.drop('predclass', axis=1)
    y = df['predclass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RSEED)
    
    # Select 10 training and 10 test examples before label encoding and scaling
    sample_train = X_train.sample(n=10, random_state=RSEED)
    sample_test = X_test.sample(n=10, random_state=RSEED)
    
    sample_train['predclass'] = y_train.loc[sample_train.index].values
    sample_test['predclass'] = y_test.loc[sample_test.index].values
    
    print("\nSample Training Examples (Before Label Encoding and Scaling):")
    display(HTML(sample_train.to_html()))
    
    print("\nSample Testing Examples (Before Label Encoding and Scaling):")
    display(HTML(sample_test.to_html()))
    
    # Apply Label Encoding to the entire dataset
    df_encoded = df.apply(LabelEncoder().fit_transform)
    
    # Drop specified columns --> redundant as we performed feature engineering in clean_data function
    drop_elements = ['education', 'native-country', 'predclass', 'age_bin', 'age-hours_bin', 'hours-per-week_bin']
    y_encoded = df_encoded["predclass"]
    X_encoded = df_encoded.drop(drop_elements, axis=1)
    
    # Split the data into training and testing sets after encoding
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=RSEED)
    
    # Scale the entire training and testing data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    
    # Apply Label Encoding to the stored 10 training and 10 test examples
    sample_train_encoded = sample_train.apply(LabelEncoder().fit_transform)
    sample_test_encoded = sample_test.apply(LabelEncoder().fit_transform)
    
    # Drop specified columns for the stored examples
    sample_train_encoded = sample_train_encoded.drop(drop_elements, axis=1)
    sample_test_encoded = sample_test_encoded.drop(drop_elements, axis=1)
    
    # Scale the stored 10 training and 10 test examples
    sample_train_encoded_scaled = scaler.transform(sample_train_encoded)
    sample_test_encoded_scaled = scaler.transform(sample_test_encoded)
    
    sample_train_encoded_scaled_df = pd.DataFrame(sample_train_encoded_scaled, columns=X_encoded.columns)
    sample_test_encoded_scaled_df = pd.DataFrame(sample_test_encoded_scaled, columns=X_encoded.columns)
    
    sample_train_encoded_scaled_df['predclass'] = y_train.loc[sample_train.index].values
    sample_test_encoded_scaled_df['predclass'] = y_test.loc[sample_test.index].values
    
    print("\nSample Training Examples (After Label Encoding and Scaling):")
    display(HTML(sample_train_encoded_scaled_df.to_html()))
    
    print("\nSample Testing Examples (After Label Encoding and Scaling):")
    display(HTML(sample_test_encoded_scaled_df.to_html()))
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, X_encoded.columns





# ### Step 3: Exploratory Data Analysis Function
# 
# The reference code contained lot of individual blocks, I condensed them all into one single function

# In[3]:


def perform_eda(df):
    print(df.info())
    print(df.isnull().sum())
    
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'predclass']
    for col in categorical_columns:
        print(f'{col}: {df[col].unique()}')
    
    plt.figure(figsize=(20, 1))
    sns.countplot(y="predclass", data=df, hue="predclass", palette="rainbow", legend=False)
    plt.title('Count of Predclass')
    plt.show()
    
    plt.figure(figsize=(20, 3))
    sns.countplot(y="education", data=df, hue="education", palette="rainbow", legend=False)
    plt.title('Count of Education Levels')
    plt.show()
    
    plt.figure(figsize=(20, 2))
    sns.countplot(y="marital-status", data=df, hue="marital-status", palette="rainbow", legend=False)
    plt.title('Count of Marital Status')
    plt.show()
    
    plt.figure(figsize=(20, 3))
    sns.countplot(y="workclass", data=df, hue="workclass", palette="rainbow", legend=False)
    plt.title('Count of Workclass')
    plt.show()
    
    df['age_bin'] = pd.cut(df['age'], 20)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(y="age_bin", data=df, hue="age_bin", palette="rainbow", legend=False)
    plt.title('Count of Age Bins')
    plt.subplot(1, 2, 2)
    sns.histplot(df[df['predclass'] == '>50K']['age'], kde=True, label=">$50K", color="blue")
    sns.histplot(df[df['predclass'] == '<=50K']['age'], kde=True, label="<=$50K", color="red")
    plt.title('Age Distribution by Predclass')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(20, 5))
    sns.countplot(x="race", hue="gender", data=df, palette="rainbow")
    plt.title('Count of Race by Gender')
    plt.show()
    
    df['hours-per-week_bin'] = pd.cut(df['hours-per-week'], 10)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(y="hours-per-week_bin", data=df, hue="hours-per-week_bin", palette="rainbow", legend=False)
    plt.title('Count of Hours-per-Week Bins')
    plt.subplot(1, 2, 2)
    sns.histplot(df['hours-per-week'], kde=True, color="green")
    sns.histplot(df[df['predclass'] == '>50K']['hours-per-week'], kde=True, label=">$50K", color="blue")
    sns.histplot(df[df['predclass'] == '<=50K']['hours-per-week'], kde=True, label="<$50K", color="red")
    plt.title('Hours-per-Week Distribution by Predclass')
    plt.legend()
    plt.show()
    
    df['age-hours'] = df['age'] * df['hours-per-week']
    df['age-hours_bin'] = pd.cut(df['age-hours'], 10)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(y="age-hours_bin", data=df, hue="age-hours_bin", palette="rainbow", legend=False)
    plt.title('Count of Age-Hours Bins')
    plt.subplot(1, 2, 2)
    sns.histplot(df[df['predclass'] == '>50K']['age-hours'], kde=True, label=">$50K", color="blue")
    sns.histplot(df[df['predclass'] == '<=50K']['age-hours'], kde=True, label="<$50K", color="red")
    plt.title('Age-Hours Distribution by Predclass')
    plt.legend()
    plt.show()
    
    sns.pairplot(df, hue='predclass', palette='deep', height=3, diag_kind='kde', diag_kws=dict(fill=True), plot_kws=dict(s=20)).set(xticklabels=[])
    plt.title('Pairplot of Features by Predclass')
    plt.show()
    


# ### Step 4: Comparision of Various ML Models (for chosen dataset)
# 
# Note: This is not required in the assignment, but I did it anyway as it was in reference code. Gives good insights on random forest performance
# 
# Note: I modularized the code in reference code into a single function for better readability

# In[4]:


def measure_performance(X_train, y_train, X_test, y_test):
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    kfold = KFold(n_splits=10, random_state=RSEED, shuffle=True)
    classifiers = ['Naive Bayes', 'Linear SVM', 'Radial SVM', 'Logistic Regression', 'Decision Tree', 'KNN', 'Random Forest']
    models = [GaussianNB(), svm.SVC(kernel='linear'), svm.SVC(kernel='rbf', C=1, gamma=0.22), LogisticRegression(), DecisionTreeClassifier(),
              KNeighborsClassifier(n_neighbors=9), RandomForestClassifier(n_estimators=25)]
    
    xyz = []
    accuracy = []
    std = []
    training_times = []
    testing_times = []
    training_scores = []
    testing_scores = []
    
    for i, model in enumerate(tqdm(models, desc="Model Training and Testing Progress")):
        print(f"Training and testing {classifiers[i]}...")
        
        # Measure training time
        train_start_time = time.time()
        model.fit(X_train, y_train)
        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        training_times.append(training_time)
        
        # Measure training accuracy
        train_score = model.score(X_train, y_train)
        training_scores.append(train_score)
        
        # Measure cross-validation accuracy
        cv_result = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
        xyz.append(cv_result.mean())
        std.append(cv_result.std())
        accuracy.append(cv_result)
        
        # Measure testing time
        test_start_time = time.time()
        test_score = model.score(X_test, y_test)
        test_end_time = time.time()
        testing_time = test_end_time - test_start_time
        testing_times.append(testing_time)
        
        # Measure testing accuracy
        testing_scores.append(test_score)
    
    models_dataframe = pd.DataFrame({
        'CV Mean': xyz,
        'Std': std,
        'Training Time (s)': training_times,
        'Testing Time (s)': testing_times,
        'Training Accuracy': training_scores,
        'Test Accuracy': testing_scores
    }, index=classifiers)
    
    end_time = time.time()
    final_memory = process.memory_info().rss
    
    execution_time = end_time - start_time
    memory_consumed = (final_memory - initial_memory) / (1024 * 1024) 
    
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    print(f"Total Memory Consumed: {memory_consumed:.2f} MB")
    
    return models_dataframe, execution_time, memory_consumed, models


# In[5]:


def compare_ML_models_graphs(models_dataframe, total_execution_time, total_memory_consumed):
    # Plot Training Time
    plt.figure(figsize=(12, 6))
    models_dataframe['Training Time (s)'].plot(kind='bar', color='skyblue')
    plt.title('Training Time for Each Model')
    plt.ylabel('Training Time (seconds)')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.show()
    
    # Plot Testing Time
    plt.figure(figsize=(12, 6))
    models_dataframe['Testing Time (s)'].plot(kind='bar', color='lightgreen')
    plt.title('Testing Time for Each Model')
    plt.ylabel('Testing Time (seconds)')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.show()
    
    # Plot Training Accuracy
    plt.figure(figsize=(12, 6))
    models_dataframe['Training Accuracy'].plot(kind='bar', color='salmon')
    plt.title('Training Accuracy for Each Model')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.show()
    
    # Plot Test Accuracy
    plt.figure(figsize=(12, 6))
    models_dataframe['Test Accuracy'].plot(kind='bar', color='orange')
    plt.title('Test Accuracy for Each Model')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.show()
    
    # Plot Cross-Validation Mean Accuracy
    plt.figure(figsize=(12, 6))
    models_dataframe['CV Mean'].plot(kind='bar', color='purple')
    plt.title('Cross-Validation Mean Accuracy for Each Model')
    plt.ylabel('CV Mean Accuracy')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.show()
    
    # Print total execution time and memory consumed
    print(f"Total Execution Time: {total_execution_time:.2f} seconds")
    print(f"Total Memory Consumed: {total_memory_consumed:.2f} MB")


# ### Step 5: Auxillary functions for Random Forest Testing (for various tasks such as visualization, training and testing)
# 
# Note: I took code from reference codes and modularized them into functions

# In[6]:


# Performs training and evaluation
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    start_train_time = time.time()
    model.fit(X_train, y_train)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    
    start_test_time = time.time()
    y_pred = model.predict(X_test)
    end_test_time = time.time()
    test_time = end_test_time - start_test_time
    
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return model, y_pred, accuracy, cm, train_time, test_time, classification_rep

# Plots single confusion matrix --> Not printed in notebook, but used
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
# I wrote this function for joining confusion matrices for better comparison 
def plot_all_confusion_matrices(confusion_matrices, model_names):
    num_plots = len(confusion_matrices)
    for i in range(0, num_plots, 4):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        for ax, cm, model_name in zip(axes, confusion_matrices[i:i+4], model_names[i:i+4]):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], ax=ax)
            ax.set_title(f"Confusion Matrix: {model_name}")
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
        plt.tight_layout()
        plt.show()

# Plots the ROC curves
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for model_name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, y_prob):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='best')
    plt.show()

# This function stores the tree image in a directory inside present working directory
# Note: This only stores tree with limited depth for easy visualization purposes

def visualize_tree(tree, feature_names, filename, max_depth=None):
    if not os.path.exists("./output/"):
        os.mkdir("./output/")
    
    # Export the tree to a DOT file with max_depth
    export_graphviz(
        tree, out_file=f'./output/{filename}.dot', 
        feature_names=feature_names, filled=True, rounded=True,
        special_characters=True, max_depth=max_depth  # Limit the depth of the tree for visualization
    )
    
    with open(f'./output/{filename}.dot') as f:
        dot_graph = f.read()
    graph = graphviz.Source(dot_graph)
    graph.format = 'svg'
    graph.render(filename=f'./output/{filename}', format='svg')
    
    # Returning the image path for --> Displaying the tree in the Jupyter notebook
    return SVG(filename=f'./output/{filename}.svg')



# This function helps us to compare different models on which feature is given more importance
def print_feature_importance(models, feature_names):
    feature_importances = {}
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({'feature': feature_names,
                               'importance': model.feature_importances_}).\
                                sort_values('importance', ascending=False)
            feature_importances[model_name] = fi
            print(f"\nFeature Importance for {model_name}:")
            print(fi.head())
    return feature_importances

# I wrote this function for joining feature importance plots for better comparison 
def plot_all_feature_importances(feature_importances):
    num_plots = len(feature_importances)
    model_names = list(feature_importances.keys())
    for i in range(0, num_plots, 4):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        for ax, model_name in zip(axes, model_names[i:i+4]):
            fi = feature_importances[model_name]
            sns.barplot(x='importance', y='feature', data=fi, ax=ax)
            ax.set_title(f'Feature Importance: {model_name}')
        plt.tight_layout()
        plt.show()

# Printing hyper-params as assignment asks us to
def print_hyperparameters(models):
    hyperparameters = {}
    for model_name, model in models.items():
        hyperparameters[model_name] = model.get_params()
        print(f"\nHyper-parameters for {model_name}:")
        print(model.get_params())
    return hyperparameters

# Based on accuracy, I am printing top-10 trees, bottom-10 trees
# I am also printing first 10 and last 10 trees
def evaluate_top_bottom_trees(random_forest, X_test, y_test):
    tree_accuracies = []
    for i, estimator in enumerate(random_forest.estimators_):
        y_pred = estimator.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        tree_accuracies.append((i, accuracy, estimator))
    
    # Capture the first 10 and last 10 trees before sorting
    first_10_trees = tree_accuracies[:10]
    last_10_trees = tree_accuracies[-10:]
    
    # Sort the trees based on accuracy
    tree_accuracies.sort(key=lambda x: x[1], reverse=True)
    top_10_trees = tree_accuracies[:10]
    bottom_10_trees = tree_accuracies[-10:]
    
    top_10_df = pd.DataFrame(top_10_trees, columns=['Tree Index', 'Accuracy', 'Estimator'])
    bottom_10_df = pd.DataFrame(bottom_10_trees, columns=['Tree Index', 'Accuracy', 'Estimator'])
    first_10_df = pd.DataFrame(first_10_trees, columns=['Tree Index', 'Accuracy', 'Estimator'])
    last_10_df = pd.DataFrame(last_10_trees, columns=['Tree Index', 'Accuracy', 'Estimator'])

    return {
        "top_10_trees": top_10_df,
        "bottom_10_trees": bottom_10_df,
        "first_10_trees": first_10_df,
        "last_10_trees": last_10_df
    }

# Wrapper function for training etc (For cleaner code)
def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test, feature_names, models, confusion_matrices, accuracies, tree_images, classification_reports, tree_evaluations=None):
    model, y_pred, accuracy, cm, train_time, test_time, classification_rep = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
    models[model_name] = model
    confusion_matrices.append(cm)
    accuracies.append(accuracy)
    classification_reports[model_name] = classification_rep
    if isinstance(model, RandomForestClassifier):
        tree_images[model_name] = visualize_tree(model.estimators_[0], feature_names, model_name.replace(" ", "_").lower(), max_depth=2)
        if tree_evaluations is not None:
            tree_evaluations[model_name] = evaluate_top_bottom_trees(model, X_test, y_test)
    else:
        tree_images[model_name] = visualize_tree(model, feature_names, model_name.replace(" ", "_").lower(), max_depth=2)
    return train_time, test_time


# ### Step 6: Main decision tree and random forest comparison function
# 
# Note: I modularized the function for cleaner readability. This will train the models, evaluate and plot the statistics.
# 
# This will compare the following models
# 
# 1. 'Decision Tree (CART)' 
# 2. 'Decision Tree (Entropy)' 
# 3. 'Random Forest (5 trees)'
# 4. 'Random Forest (10 trees)'
# 5. 'Random Forest (25 trees)'
# 6. 'Random Forest (50 trees)'
# 7. 'Random Forest (75 trees)'
# 8. 'Random Forest (100 trees)'
# 
# 
# 

# In[7]:


def compare_decision_tree_vs_random_forest(X_train, y_train, X_test, y_test, feature_names):
    models = {}
    confusion_matrices = []
    accuracies = []
    tree_images = {}
    train_times = []
    test_times = []
    classification_reports = {}
    tree_evaluations = {}
    
    # Train and evaluate Decision Tree with CART (default)
    train_time, test_time = train_and_evaluate(DecisionTreeClassifier(random_state=RSEED), "Decision Tree (CART)", X_train, y_train, X_test, y_test, feature_names, models, confusion_matrices, accuracies, tree_images, classification_reports)
    train_times.append(train_time)
    test_times.append(test_time)
    
    # Train and evaluate Decision Tree with entropy criterion (Approximate C4.5 but not true C4.5)
    train_time, test_time = train_and_evaluate(DecisionTreeClassifier(criterion='entropy', random_state=RSEED), "Decision Tree (Entropy)", X_train, y_train, X_test, y_test, feature_names, models, confusion_matrices, accuracies, tree_images, classification_reports)
    train_times.append(train_time)
    test_times.append(test_time)
    
    # Train and evaluate Random Forest with 5 trees
    train_time, test_time = train_and_evaluate(RandomForestClassifier(n_estimators=5, random_state=RSEED), "Random Forest (5 trees)", X_train, y_train, X_test, y_test, feature_names, models, confusion_matrices, accuracies, tree_images, classification_reports, tree_evaluations)
    train_times.append(train_time)
    test_times.append(test_time)
    
    # Train and evaluate Random Forest with 10 trees
    train_time, test_time = train_and_evaluate(RandomForestClassifier(n_estimators=10, random_state=RSEED), "Random Forest (10 trees)", X_train, y_train, X_test, y_test, feature_names, models, confusion_matrices, accuracies, tree_images, classification_reports, tree_evaluations)
    train_times.append(train_time)
    test_times.append(test_time)
    
    # Train and evaluate Random Forest with 25 trees
    train_time, test_time = train_and_evaluate(RandomForestClassifier(n_estimators=25, random_state=RSEED), "Random Forest (25 trees)", X_train, y_train, X_test, y_test, feature_names, models, confusion_matrices, accuracies, tree_images, classification_reports, tree_evaluations)
    train_times.append(train_time)
    test_times.append(test_time)
    
    # Train and evaluate Random Forest with 50 trees
    train_time, test_time = train_and_evaluate(RandomForestClassifier(n_estimators=50, random_state=RSEED), "Random Forest (50 trees)", X_train, y_train, X_test, y_test, feature_names, models, confusion_matrices, accuracies, tree_images, classification_reports, tree_evaluations)
    train_times.append(train_time)
    test_times.append(test_time)
    
    # Train and evaluate Random Forest with 75 trees
    train_time, test_time = train_and_evaluate(RandomForestClassifier(n_estimators=75, random_state=RSEED), "Random Forest (75 trees)", X_train, y_train, X_test, y_test, feature_names, models, confusion_matrices, accuracies, tree_images, classification_reports, tree_evaluations)
    train_times.append(train_time)
    test_times.append(test_time)
    
    # Train and evaluate Random Forest with 100 trees
    train_time, test_time = train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=RSEED), "Random Forest (100 trees)", X_train, y_train, X_test, y_test, feature_names, models, confusion_matrices, accuracies, tree_images, classification_reports, tree_evaluations)
    train_times.append(train_time)
    test_times.append(test_time)
    
    # Plot accuracies
    model_names = ['Decision Tree (CART)', 'Decision Tree (Entropy)', 'Random Forest (5 trees)', 'Random Forest (10 trees)', 'Random Forest (25 trees)', 'Random Forest (50 trees)', 'Random Forest (75 trees)', 'Random Forest (100 trees)']
    
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
    plt.title('Model Comparison: Decision Tree (CART) vs Decision Tree (Entropy) vs Random Forest')
    plt.ylabel('Accuracy')
    plt.xlabel('Models')
    plt.ylim(min(accuracies) - 0.05, max(accuracies) + 0.05)  #Note: I am shifting y axis for better readability
    plt.xticks(rotation=45)
    plt.show()
    
    # Plot ROC curves for all models
    plot_roc_curves(models, X_test, y_test)
    
    # Print feature importance for all models
    feature_importances = print_feature_importance(models, feature_names)
    
    # Plot all feature importances together
    plot_all_feature_importances(feature_importances)
    
    # Print hyper-parameters for all models
    hyperparameters = print_hyperparameters(models)
    
    # Plot all confusion matrices together
    plot_all_confusion_matrices(confusion_matrices, model_names)
    
    # Print classification reports 
    print("\nClassification Reports:")
    for model_name, report in classification_reports.items():
        print(f"\nClassification Report for {model_name}:")
        display(pd.DataFrame(report).transpose())
    
    # Tabulate training and testing times
    time_df = pd.DataFrame({
        'Model': model_names,
        'Training Time (s)': train_times,
        'Testing Time (s)': test_times
    })
    print("\nTraining and Testing Times:")
    display(time_df)
    
    # Plot training and testing times
    time_df.plot(x='Model', kind='bar', figsize=(10, 6))
    plt.title('Training and Testing Times')
    plt.ylabel('Time (s)')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.show()
    
    # Plot top 10, bottom 10 trees etc...
    for model_name, evaluations in tree_evaluations.items():
        combined_df = pd.DataFrame({
            'Top 10 Trees Index': evaluations['top_10_trees']['Tree Index'].tolist(),
            'Top 10 Trees Accuracy': evaluations['top_10_trees']['Accuracy'].tolist(),
            'Bottom 10 Trees Index': evaluations['bottom_10_trees']['Tree Index'].tolist(),
            'Bottom 10 Trees Accuracy': evaluations['bottom_10_trees']['Accuracy'].tolist(),
            'First 10 Trees Index': evaluations['first_10_trees']['Tree Index'].tolist(),
            'First 10 Trees Accuracy': evaluations['first_10_trees']['Accuracy'].tolist(),
            'Last 10 Trees Index': evaluations['last_10_trees']['Tree Index'].tolist(),
            'Last 10 Trees Accuracy': evaluations['last_10_trees']['Accuracy'].tolist()
        })
        print(f"\nTop and Bottom Trees for {model_name}:")
        display(combined_df)
    
    results = {
        "models": models,
        "accuracies": accuracies,
        "confusion_matrices": confusion_matrices,
        "feature_importances": feature_importances,
        "hyperparameters": hyperparameters,
        "classification_reports": classification_reports,
        "tree_images": tree_images,
        "train_times": train_times,
        "test_times": test_times
    }
    
    return results


# # ACTUAL CODE EXECUTION STARTS NOW

# ### Step 7: Read the data and clean it
# 
# Note: Data contains some redundant features etc

# In[8]:


income_df = pd.read_csv("./datasets/income/adult.csv")
income_df.head()


# In[9]:


income_df.describe()


# In[10]:


cleaned_df = clean_data(income_df)


# ### Step 8: Perform Exploratory Data Analysis and Train:Test Split

# In[11]:


perform_eda(cleaned_df)


# In[12]:


X_train, X_test, y_train, y_test, feature_names = prepare_data(cleaned_df)


# ### Step 9: Compare Different ML Models

# In[13]:


performance_results, total_execution_time, total_memory_consumed, trained_models = measure_performance(X_train, y_train, X_test, y_test)


# In[14]:


compare_ML_models_graphs(performance_results, total_execution_time, total_memory_consumed)


# ### Step 10: Compare Decision Tree and Random Forests with various configurations

# In[15]:


results = compare_decision_tree_vs_random_forest(X_train, y_train, X_test, y_test, feature_names)


# In[16]:


display(results["tree_images"]["Decision Tree (CART)"])


# In[17]:


display(results["tree_images"]["Decision Tree (Entropy)"])


# In[18]:


display(results["tree_images"]["Random Forest (5 trees)"])


# In[19]:


display(results["tree_images"]["Random Forest (10 trees)"])


# In[20]:


display(results["tree_images"]["Random Forest (25 trees)"])


# In[21]:


display(results["tree_images"]["Random Forest (50 trees)"])


# In[22]:


display(results["tree_images"]["Random Forest (75 trees)"])


# In[23]:


display(results["tree_images"]["Random Forest (100 trees)"])

