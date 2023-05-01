import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df1 = pd.read_csv("Youtube01-Psy.csv")              # Psy youtube channel most viewed video comments dataset
df2 = pd.read_csv("Youtube02-KatyPerry.csv")        # KatyPerry youtube channel most viewed video comments dataset
df3 = pd.read_csv("Youtube03-LMFAO.csv")            # Psy LMFAO channel most viewed video comments dataset
df4 = pd.read_csv("Youtube04-Eminem.csv")           # Eminem youtube channel most viewed video comments dataset
df5 = pd.read_csv("Youtube05-Shakira.csv")           # Shakira youtube channel most viewed video comments dataset

# Merge all the datasset into single file
frames = [df1,df2,df3,df4,df5]                          # make a list of all file
df_merged = pd.concat(frames)                           # concatenate the all the file into single
keys = ["Psy","KatyPerry","LMFAO","Eminem","Shakira"]   # Merging with Keys
df_with_keys = pd.concat(frames,keys=keys)              # concatenate data with keys
dataset=df_with_keys

# Infomation about dataset
print(dataset.size)                 # size of dataset
print(dataset.shape)                # shape of datadet
print(dataset.keys())               # attributes of dataset

# working with text content
dataset = dataset[["CONTENT" , "CLASS"]]             # context = comments of viewers & Class = ham or Spam

# Predictor and Target attribute
dataset_X = dataset['CONTENT']                       # predictor attribute
dataset_y = dataset['CLASS']                         # target attribute

# Feature Extraction from Text using  TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer   # import TF-IDF model from scikit Learn

# Extract Feature With TF-IDF model
corpus = dataset_X                               # declare the variable
cv = TfidfVectorizer()                           # initialize the TF-IDF  model
X = cv.fit_transform(corpus).toarray()           # fit the corpus data into BOW model

# Split the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, dataset_y, test_size=0.2, random_state=0)

# shape of predictor attrbute after Extract Features
X.shape


# import the model from sklean
from sklearn.svm import SVC             # import the Support Vector Machine Classifier model

# initialize the model
classifier = SVC(kernel = 'linear', random_state= 0)

# fit the dataset into our classifier model for training
classifier.fit(X_train, y_train)

# predict the result
y_pred = classifier.predict(X_test)
print(y_pred)

# Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix= confusion_matrix(y_test, y_pred)
print(confusion_matrix)
#[row, column]
TP = confusion_matrix[1, 1]
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]

# Evaluate the Result
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, roc_auc_score

# Accuracy Score
print('Accuracy Score:', accuracy_score(y_test, y_pred))

# Precision Score
print('Precision Score:', precision_score(y_test, y_pred))

# True positive Rate (TPR) or Sensitivity or Recall
print('True positive Rate:', recall_score(y_test, y_pred))

# False positive Rate (FPR)
print('False positive Rate', FP / float(TN + FP))

# F1 Score or F-Measure or F-Score
print('F1 Score:', f1_score(y_test, y_pred))

# Specificity
print('Specificity:', TN / (TN + FP))

# Mean Absolute Error
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# ROC Area
print('ROC Area:', roc_auc_score(y_test, y_pred))

# import pickle library
import pickle               # pickle used for serializing and de-serializing a Python object structure

# save the model
Support_Vector_Machine = open("model.pkl","wb")          # open the file for writing
pickle.dump(classifier,Support_Vector_Machine)           # dumps an object to a file object
Support_Vector_Machine.close()                           # here we close the fileObject

# Load the model
ytb_model = open("model.pkl","rb")           # open the file for reading
new_model = pickle.load(ytb_model)           # load the object from the file into new_model
new_model

# Used the model for Prediction
comment = ["Hey Music Fans I really appreciate all of you,but see this song too"]
vect = cv.transform(comment).toarray()
new_model.predict(vect)

if new_model.predict(vect) == 1:
    print("Spam")
else:
    print("Ham")



