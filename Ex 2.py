import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load data
data_renee = pd.read_csv('C:/dataset/breast_cancer.csv')
data_renee.head()

# Data preprocessing
# Replace '?' with NaN and convert to float
data_renee['bare'] = data_renee['bare'].replace('?', np.nan).astype(float)

# Drop ID column
data_renee = data_renee.drop('ID', axis=1)

# Split features and target
X = data_renee.drop('class', axis=1)
y = data_renee['class']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(X_train)
print(X_test)


########### I stoped using this way since it ends up in a Accuracy of 0.35, beuse It is defined againg in the pipeline just in the next step
# # 6.	Using the preprocessing library to define two transformer objects to transform your training data:
#             # a.	Fill the missing values with the median (hint: checkout SimpleImputer)
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy='median')

# imputer.fit(X_train)
# X_train_SimpleImputer = imputer.fit_transform(X_train)
# print(X_train_SimpleImputer)

# # b.	Scale the data  (hint: checkout StandardScaler)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_StandardScaler = scaler.fit_transform(X_train)
# print(X_train_StandardScaler)


# Define preprocessing and model pipeline
# Correction: Pipeline defined with SimpleImputer and StandardScaler together, without applying fit_transform twice.
num_pipe_renee = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
    ('scaler', StandardScaler())  # Scale the data
])

# Create a new Pipeline that has two steps the first is the num_pipe_firstname and the second is an SVM classifier with random state = last two digits of your student number. Name the pipeline pipe_svm_firstname. (make note of the labels)
pipe_svm_renee = Pipeline([
    ('preprocessing', num_pipe_renee),
    ('classifier', SVC(random_state=5))
])

# Define the grid search parameters in an object and name it param_grid, as follows:
# a.	        'svc__kernel': ['linear', 'rbf','poly'],
# b.	        'svc__C':  [0.01,0.1, 1, 10, 100],
# c.	         'svc__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
# d.	         'svc__degree':[2,3]},
# Make sure you replace svc with the label you used in the pipe_svm_firstname for the model
param_grid = {
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    'classifier__degree': [2, 3]
}

# Create a grid search object name it grid_search_firstname with the following parameters:
    # a.	estimator= pipe_svm_firstname 
    # b.	param_grid=param_grid_svm
    # c.	scoring='accuracy' 
    # d.	refit = True
    # e.	verbose = 3
grid_search_renee = GridSearchCV(pipe_svm_renee, param_grid, cv=5, scoring='accuracy', refit=True, verbose=3)

# Fit your training data to the gird search object. (This will take some time but you will see the results on the console)
grid_search_renee.fit(X_train, y_train)

# Print out the best parameters and note it in your written response
best_params = grid_search_renee.best_params_
print("Best Parameters:", best_params)

# Printout the best estimator and note it in your written response
best_estimator = grid_search_renee.best_estimator_
print("Best Estimator:", best_estimator)

# Create an object that holds the best model i.e. best estimator to an object named best_model_firstname.
best_model_renee = grid_search_renee.best_estimator_

# Printout the accuracy score and note it in your written 
y_pred = best_model_renee.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#19.	Save the model using the joblib  (dump). Name the file svm_firstname.pkl
from joblib import dump
dump(best_model_renee, 'svm_renee.pkl')
print('Model saved as svm_renee.pkl')

# 20.	Save the full pipeline using the joblib – (dump). Name the file pipeline_firstname.pkl
dump(pipe_svm_renee, 'pipeline_renee.pkl')
print('Pipeline saved as pipeline_renee.pkl')

