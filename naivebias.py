# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load dataset
msg = pd.read_csv('./P6_naivetext1.csv')
print('Total instances in the dataset:', msg.shape[0])

# Map text labels to numeric values: 'pos' -> 1, 'neg' -> 0
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

# Separate features (X) and target labels (Y)
X, Y = msg.message, msg.labelnum

# Preview first five messages and their original labels
print('\nThe message and its label of the first five instances are listed below:')
for x, y in zip(X[:5], msg.label[:5]):
    print(f"{x}, {y}")

# Split the data into training and test sets (default 75%-25%)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

print('\nDataset training and testing:')
print('Total training:', xtrain.shape[0])
print('Total testing:', xtest.shape[0])

# Text preprocessing using Count Vectorizer
vectorizer = CountVectorizer()
xtrain_dtm = vectorizer.fit_transform(xtrain)  # Fit and transform training data
xtest_dtm = vectorizer.transform(xtest)        # Transform test data

# Show all feature names (vocabulary)
print('\nFeatures (vocabulary):')
print(vectorizer.get_feature_names_out())
print('\nTotal number of features:', xtrain_dtm.shape[1])

# Display feature matrix for first five training examples
print('\nFeature vectors for first five training instances:')
df = pd.DataFrame(xtrain_dtm.toarray(), columns=vectorizer.get_feature_names_out())
print(df.head())

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB().fit(xtrain_dtm, ytrain)

# Predict labels on test data
predicted = clf.predict(xtest_dtm)

# Output predicted results for test instances
print('\nClassification Results:')
for doc, p in zip(xtest, predicted):
    print(f"{doc} -> {'pos' if p == 1 else 'neg'}")

# Accuracy and performance metrics
print('\nAccuracy Metrics:')
print('Accuracy:', metrics.accuracy_score(ytest, predicted))
print('\nConfusion Matrix:')
print(metrics.confusion_matrix(ytest, predicted))
print('\nRecall:', metrics.recall_score(ytest, predicted, zero_division=0))
print('Precision:', metrics.precision_score(ytest, predicted, zero_division=0))
