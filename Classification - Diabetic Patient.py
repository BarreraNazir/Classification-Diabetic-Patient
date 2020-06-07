#!/usr/bin/env python
# coding: utf-8

# # Classification on Diabetic Patient

# Importing the libraries

# In[1]:


import pandas as pd # Used to loading files
import numpy as np # Operate with mathematical operations
import matplotlib.pyplot as plt # Used for plotting
import seaborn as sns  # Used for visualizing


# Reading the dataset from Kaggle repository

# In[2]:


data = pd.read_csv("diabetes.csv") # reading csv file


# In[3]:


data.head() # reading first five rows


# Analyzing the data

# In[4]:


data.info() # missing values in rating, type, current and android ver
            # only rating attribute contains numerical data , rest of them belong to categorical data


# # Pre-Processing

# In[5]:


# checking the missing values from the data
print(data.isnull().sum())


# Missing values not found in columns

# # Normalization

# In[6]:


norm_column = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']


# In[7]:


before_norm = data[norm_column]


# In[8]:


before_norm.head(5)


# In[9]:


after_norm = data[norm_column].apply(lambda x: ( (x-x.min() ) / (x.max() - x.min() ) ) )


# In[10]:


after_norm.head(5)


# # Data Visualization

# In[36]:


f, ax = plt.subplots(1, 2, figsize = (15,5))
f.suptitle("Diabetic Patient", fontsize = 18.)
_ = data.Outcome.value_counts().plot.bar(ax = ax[0], rot = 0, color = (sns.color_palette()[0], sns.color_palette()[2])).set(xticklabels = ["No", "Yes"])
_ = data.Outcome.value_counts().plot.pie(labels = ("un-diabetic","diabetic"), autopct = "%.2f%%", label = "", fontsize = 13., ax = ax[1],colors = (sns.color_palette()[0], sns.color_palette()[2]), wedgeprops = {"linewidth": 1.5, "edgecolor": "#F7F7F7"}), ax[1].texts[1].set_color("#F7F7F7"), ax[1].texts[3].set_color("#F7F7F7")


# # Classification

# In[11]:


X_Data = after_norm # independent variable


# In[12]:


Y_Data = data['Outcome'] # dependent variable or class


# Splitting train and test data using sklearn

# In[13]:


from sklearn import model_selection
# 0.3 means 30% will be used for testing and 0.7 or 70% data will be used for training
X_Train, X_Test, Y_Train, Y_Test = model_selection.train_test_split(X_Data, Y_Data, test_size = 0.3)


# In[14]:


print("Sample in training set...", X_Train.shape)
print("Sample in testing set...", X_Test.shape)
print("Sample in training set...", Y_Train.shape)
print("Sample in testing set...", Y_Test.shape)


# defining models to train the data

# In[15]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# # K Nearest Neighbor Classifier

# In[16]:


knnClassifier = KNeighborsClassifier()
knnClassifier.fit(X_Train, Y_Train)
Y_Pred = knnClassifier.predict(X_Test)
print(confusion_matrix(Y_Test, Y_Pred))
print(classification_report(Y_Test,Y_Pred))
accuracy = accuracy_score(Y_Test,Y_Pred)
print("Accuracy of KNN...", accuracy)


# # Decision Tree Classifier

# In[17]:


dtreeClassifier = DecisionTreeClassifier()
dtreeClassifier.fit(X_Train, Y_Train)
Y_Pred = dtreeClassifier.predict(X_Test)
print(confusion_matrix(Y_Test, Y_Pred))
print(classification_report(Y_Test,Y_Pred))
accuracy = accuracy_score(Y_Test,Y_Pred)
print("Accuracy of Decision Tree...", accuracy)


# # Stochastic Gradient Descent Classifier

# In[18]:


sgdClassifier = SGDClassifier()
sgdClassifier.fit(X_Train, Y_Train)
Y_Pred = sgdClassifier.predict(X_Test)
print(confusion_matrix(Y_Test, Y_Pred))
print(classification_report(Y_Test,Y_Pred))
accuracy = accuracy_score(Y_Test,Y_Pred)
print("Accuracy of SGD...", accuracy)


# In[19]:


svmClassifier = SVC(kernel='linear')
svmClassifier.fit(X_Train, Y_Train)
Y_Pred = svmClassifier.predict(X_Test)
print(confusion_matrix(Y_Test, Y_Pred))
print(classification_report(Y_Test,Y_Pred))
accuracy = accuracy_score(Y_Test,Y_Pred)
print("Accuracy of SVM...", accuracy)


# # Support Vector Machine Classifier

# # Comparison of Machine Learning Classifiers

# The decision tree classifier well performed among other classifiers. It means that given predictor will accurately guess the
# value of predicted attribute for a new data.

# In[20]:


Classifier = [['KNN', 74.8],['DT', 96.0 ],['SGD', 74.5 ],['SVM', 77.8] ]
result = pd.DataFrame(Classifier, columns = ['Classifier', 'Accuracy']) 
result.head()


# Plotting Accuries

# In[21]:


Accuracy = result['Accuracy'].values
Classifier = result['Classifier'].values
sns.set(style='darkgrid')
ax = result.plot(x="Classifier", y="Accuracy", kind="bar")
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
ax.get_legend().remove()
plt.show()

