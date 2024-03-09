import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing, linear_model, metrics, svm
import nltk
# *************************************************************************

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
# *************************************************************************

# Read the data
dataset = pd.read_csv("movie reviews.csv", encoding='latin1')
print(dataset.head())

# shuffle data
data = dataset.sample(frac=1, ignore_index=True)
print(data.head())

# visualization of the dataset
plt.subplots(figsize=(9, 7))
# scatter plot
plt.scatter(data['index'], data['result'])
plt.title("scatter plot")
plt.ylabel('Review type')
plt.xlabel('Reviews')
plt.show()

data = data.drop(data.columns[0], axis=1)
# *************************************************************************

# pre-processing like lower case, remove punctuation and numeric tokens, remove stopwords, stemming and Lemmatizing
for row in data['review']:
    row.lower()
print('//convert data to lower cases')
print(data.head())
print('*********************************************************************')

# *************************************************************************
punctuation = string.punctuation
z = 0
for row in data['review']:
    words = nltk.word_tokenize(row)
    new_text = ""
    for word in words:
        if word not in punctuation and re.search('[a-z]', word):
            new_text = new_text + word + " "
    data['review'][z] = new_text
    z = z + 1
print('//Data after removing punctuation and numeric tokens')
print(data.head())
print('*********************************************************************')

# *************************************************************************
stop = stopwords.words('english')
i = 0
for row in data['review']:
    words = nltk.word_tokenize(row)
    new_text = ""
    for word in words:
        if word not in stop:
            new_text = new_text + word + " "
    data['review'][i] = new_text
    i = i + 1
print('//Data after removing stopwords')
print(data.head())
print('*********************************************************************')

# *************************************************************************
st = PorterStemmer()
j = 0
for row in data['review']:
    words = nltk.word_tokenize(row)
    new_text = ""
    for word in words:
        new_text = new_text + st.stem(word) + " "
    data['review'][j] = new_text
    j = j + 1
print('//Data after applying stemming')
print(data.head())
print('*********************************************************************')

# *************************************************************************
x = 0
for row in data['review']:
    words = nltk.word_tokenize(row)
    new_text = ""
    for word in words:
        new_text = new_text + Word(word).lemmatize() + " "
    data['review'][x] = new_text
    x = x + 1
print('//Data after applying lemmatization')
print(data.head())
print('*********************************************************************')

# *************************************************************************
# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['result'], test_size=0.20,
                                                    shuffle=False)

# TFIDF feature generation
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

tfidf_vec = TfidfVectorizer(analyzer='word', max_features=10000)
tfidf_vec.fit(data['review'])
x_train_tfidf = tfidf_vec.transform(X_train)
x_test_tfidf = tfidf_vec.transform(X_test)
print('//Data after applying TfidfVectorizer')
print(x_train_tfidf.data)
print('*********************************************************************')
# *************************************************************************

logistic_model = linear_model.LogisticRegression().fit(x_train_tfidf, y_train)
lin_svc = svm.LinearSVC(C=1).fit(x_train_tfidf, y_train)
empty_list = []
for i, clf in enumerate((logistic_model, lin_svc)):
    if i == 0:
        predictions = clf.predict(x_test_tfidf)
        accuracy = metrics.accuracy_score(predictions, y_test)
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        plt.title("model 1")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        sn.heatmap(confusion_matrix, annot=True)
        plt.show()
        empty_list.append(accuracy)
        print('accuracy of model ' + str(i + 1) + ' = ' + str(accuracy*100))

    else:
        predictions = clf.predict(x_test_tfidf)
        accuracy = metrics.accuracy_score(predictions, y_test)
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        plt.title("model 2")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        sn.heatmap(confusion_matrix, annot=True)
        plt.show()
        empty_list.append(accuracy)
        print('accuracy of model ' + str(i + 1) + ' = ' + str(accuracy * 100))

data = {'logistic_model': empty_list[0], 'lin_svc': empty_list[1]}
courses = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize=(10, 5))
# creating the bar plot
plt.bar(courses, values, color='maroon', width=0.4)
plt.xlabel("models")
plt.ylabel("accuracy")
plt.title("bar graph")
plt.show()
