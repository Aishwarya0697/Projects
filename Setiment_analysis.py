# Natural Language Processing

# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import streamlit as st

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\Aishwarya\Desktop\Resume Project\Sentiment Analysis\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)
dataset.head()

print(dataset['Liked'].value_counts())

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []


for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
print(X.shape)
y = dataset.iloc[:, 1].values
print(type(X))
pickle.dump(cv , open("CVectorizor.pkl",'wb'))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print(type(X_train))
#Model training on Logistic Regression , Gaussianb , Random foreest without hyperparameter tuning
#1 Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predicting the Test set results
y_pred = logreg.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

print(accuracy_score(y_test , y_pred))
print(roc_auc_score(y_test , y_pred))

#2 GaussianNB
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

print(accuracy_score(y_test , y_pred))
print(roc_auc_score(y_test , y_pred))


pickle.dump(classifier , open('sentiment_clf.pkl','wb'))

