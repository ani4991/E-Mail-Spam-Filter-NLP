#  LIBRARIES IMPORT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT DATASET

spam_df = pd.read_csv("emails.csv")
print(spam_df.head(10),spam_df.tail())
print(spam_df.describe())

# VISUALIZE DATASET

#see which message is the most popular ham/spam message
spam_df.groupby('spam').describe()

# get the length of the messages
spam_df['length'] = spam_df['text'].apply(len)
print(spam_df.head())

spam_df['length'].plot(bins=100, kind='hist')

print(spam_df.length.describe())

# see the longest message 43952
spam_df[spam_df['length'] == 43952]['text'].iloc[0]

# Let's divide the messages into spam and ham

ham = spam_df[spam_df['spam']==0]
spam = spam_df[spam_df['spam']==1]

print(ham,spam)

spam['length'].plot(bins=60, kind='hist')

ham['length'].plot(bins=60, kind='hist')

print( 'Spam percentage =', (len(spam) / len(spam_df) )*100,"%")
print( 'Ham percentage =', (len(ham) / len(spam_df) )*100,"%")

sns.countplot(spam_df['spam'], label = "Count")

# CREATE TESTING AND TRAINING DATASET/DATA CLEANING

# REMOVE PUNCTUATION
import string
string.punctuation

Test = 'Hello Mr. Future, I am so happy to be learning AI now!!'

Test_punc_removed = [char for char in Test if char not in string.punctuation]
print(Test_punc_removed)

# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join

# REMOVE STOPWORDS
# download stopwords Package to execute this command

from nltk.corpus import stopwords
stopwords.words('english')

print(Test_punc_removed)

Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]

# COUNT VECTORIZER EXAMPLE

from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)

print(vectorizer.get_feature_names())
print(X.toarray())

# pipeline to clean up all the messages
# pipeline performs the following: (1) remove punctuation, (2) remove stopwords
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

# test the newly added function

spam_df_clean = spam_df['text'].apply(message_cleaning)
print(spam_df_clean[0],spam_df['text'][0])

from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning)
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])

print(vectorizer.get_feature_names())

print(spamham_countvectorizer.toarray())
print(spamham_countvectorizer.shape)

# TRAINING THE MODEL WITH ALL DATASET
from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
label = spam_df['spam'].values
NB_classifier.fit(spamham_countvectorizer, label)

testing_sample = ['Free money!!!', "Hi Kim, Please let me know if you need any further information. Thanks"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)

test_predict = NB_classifier.predict(testing_sample_countvectorizer)
print(test_predict)

# DIVIDE THE DATA INTO TRAINING AND TESTING PRIOR TO TRAINING

X = spamham_countvectorizer
y = label

print(X.shape,y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

# EVALUATING THE MODEL
from sklearn.metrics import classification_report, confusion_matrix

y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)

# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))

# Additon of FEATURE TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

emails_tfidf = TfidfTransformer().fit_transform(spamham_countvectorizer)
print(emails_tfidf.shape)

print(emails_tfidf[:,:])
# Sparse matrix with all the values of IF-IDF

X = emails_tfidf
y = label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))











