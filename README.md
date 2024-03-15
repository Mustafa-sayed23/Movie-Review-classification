# Movie-Review-classification
## program to classify the movie reviews into negative or positive review by applying Sentiment Analysis.

### •	Dataset
##### a csv file contains 3 columns:
##### 1)	review index column.
##### 2)	review column that contains string movie review.
##### 3)	result column that contains type of the review (negative or positive).

##### dataset samples scatterplot shows that data is balanced
![image](https://github.com/Mustafa-sayed23/Movie-Review-classification/assets/162192046/c7f176c1-db2e-4837-b52c-4ebcc9c1cd71)

### • Text preprocessing
##### read the dataset, shuffle the data samples and drop index column from the data set.
##### convert all dataset samples into lower case.
##### then remove punctuation and numeric tokens from the samples as they are not important in the classification.
##### After this remove English stop words from the data to make the model focus on important words.
##### pass the data to the stemmer, then to the lemmatizer.
##### In all of the previous preprocessing, word tokenization have been used to split each data sample.

##### before training the classification models dataset has been splited into 80% training set and 20% testing set.
##### TfidfVectorizer has been used to encode (convert) the categorical data into numerical data on each word using max_features =  10000, Then transform the data sample.

### • Implemented Models
##### two models have been applied on the data:
##### 1)	Logistic Regression model from ski-learn library.
##### - This model has gived testing accuracy in range 86% - 90% (it differs every time as data is shuffled).
##### 2)	LinearSVC model from ski-learn library that takes one hyperparameter c that handling the margin distance in the model.
##### - This model has gived testing accuracy in range 87% - 91% using c = 1 (also it differs every time as data is shuffled).

##### models results have been traced using confusion matrix:

##### Logistic Regression model result
![image](https://github.com/Mustafa-sayed23/Movie-Review-classification/assets/162192046/54e3b9e6-cafb-4f83-a3e8-481abf60bc89)

##### LinearSVC model using c = 1 result
![image](https://github.com/Mustafa-sayed23/Movie-Review-classification/assets/162192046/52f0e8a8-e87e-42f5-a3bd-7dbecc936418)



