import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#loading the data
data_renee = pd.read_csv('C:/dataset/breast_cancer.csv')
data_renee.head()

# Check the names and types of columns in the dataframe data_renee
print(data_renee.dtypes)

# check the missing values in the dataframe data_renee
print(data_renee.isnull().sum())

# c.	Check the statistics of the numeric fields (mean, min, max, median, count..etc.)
print(data_renee.describe())

#3.	Replace the ‘?’ mark in the ‘bare’ column by np.nan and change the type to ‘float’
data_renee['bare'] = data_renee['bare'].replace('?', np.nan).astype(float)

#4.	Fill any missing data with the median of the column.
data_renee = data_renee.fillna(data_renee.median())

# 5.	Drop the ID column
data_renee = data_renee.drop('ID', axis=1)

# 6.	Using Pandas, Matplotlib, seaborn (you can use any or a mix) generate 3-5 plots and add them to your written response explaining what are the key insights and findings from the plots.
#  Distribution of Cell Mass Thickness
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(data_renee['thickness'], bins=10, kde=True)
plt.title('Distribution of Cell Mass Thickness')
plt.xlabel('Thickness')
plt.ylabel('Frequency')
plt.show()

# Comparison of Cell Mass Size by Class
plt.figure(figsize=(10, 6))
sns.boxplot(x='class', y='size', data=data_renee)
plt.title('Comparison of Cell Mass Size by Class')
plt.xlabel('Class')
plt.ylabel('Size')
plt.show()

# Heatmap of Feature Correlations
plt.figure(figsize=(12, 8))
corr = data_renee.corr() 
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()

#7.	Separate the features from the class.
X = data_renee.drop('class', axis=1)
y = data_renee['class']

# 8.	Split your data into train 80% train and 20% test, use the last two digits of your student number for the seed. 
from sklearn.model_selection import train_test_split

# Build Classification Models
#Support vector machine classifier with linear kernel
# 9.	Train an SVM classifier using the training data, set the kernel to linear and set the regularization parameter to C= 0.1. Name the classifier clf_linear_firstname.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
from sklearn.svm import SVC
clf_linear_renee = SVC(kernel='linear', C=0.1)

# 10.	Print out two accuracy score one for the model on the training set i.e. X_train, y_train and the other on the testing set i.e. X_test, y_test. Record both results in your written response.
clf_linear_renee.fit(X_train, y_train)
print('Training Accuracy:', clf_linear_renee.score(X_train, y_train))
print('Testing Accuracy:', clf_linear_renee.score(X_test, y_test))

# 11.	Generate the accuracy matrix. Record the results in your written response.
from sklearn.metrics import classification_report, confusion_matrix
y_pred = clf_linear_renee.predict(X_test)
print(y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Support vector machine classifier with “rbf” kernel
# 12.	Repeat steps 9 to 11, in step 9 change the kernel to “rbf” and do not set any value for C.
clf_rbf_renee = SVC(kernel='rbf')
clf_rbf_renee.fit(X_train, y_train)
print('Training Accuracy:', clf_rbf_renee.score(X_train, y_train))
print('Testing Accuracy:', clf_rbf_renee.score(X_test, y_test))
y_pred = clf_rbf_renee.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Support vector machine classifier with “poly” kernel
# 13.	Repeat steps 9 to 11, in step 9 change the kernel to “poly” and do not set any value for C
clf_poly_renee = SVC(kernel='poly')
clf_poly_renee.fit(X_train, y_train)
print('Training Accuracy:', clf_poly_renee.score(X_train, y_train))
print('Testing Accuracy:', clf_poly_renee.score(X_test, y_test))
y_pred = clf_poly_renee.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Support vector machine classifier with “sigmoid” kernel
# 14.	Repeat steps 9 to 11, in step 9 change the kernel to “sigmoid” and do not set any value for C.
clf_sigmoid_renee = SVC(kernel='sigmoid')
clf_sigmoid_renee.fit(X_train, y_train)
print('Training Accuracy:', clf_sigmoid_renee.score(X_train, y_train))
y_pred = clf_sigmoid_renee.predict(X_test)
print('Testing Accuracy:', clf_sigmoid_renee.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

