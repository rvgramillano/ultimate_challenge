import pandas as pd
import numpy as np
import json
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

DIR = '/Users/rvg/Documents/springboard_ds/ultimate_challenge/'

# Load json file and put into pandas DataFrame
with open(DIR + 'ultimate_data_challenge.json') as f:
    data = json.load(f)

df = pd.DataFrame.from_records(data)

# first, we convert object variable columns to datetime
# those variable are last_trip_date (datetime), and signupdate (datetime)

df['last_trip_date'] = pd.to_datetime(df['last_trip_date'], format='%Y-%m-%d')
df['signup_date'] = pd.to_datetime(df['signup_date'], format='%Y-%m-%d')

# next, let's fill columns with null values (avg_rating_by_driver, avg_rating_of_driver, phone)
# first lets look at the statistics to see whether to use a mean or median for filling null values
df[['avg_rating_by_driver', 'avg_rating_of_driver']].describe()

# both have low std and the mean is close to the median, so we fill with the mean
df['avg_rating_by_driver'].fillna(df['avg_rating_by_driver'].mean(), inplace=True)
df['avg_rating_of_driver'].fillna(df['avg_rating_of_driver'].mean(), inplace=True)

# fill missing values in phone column with mode
df['phone'].fillna(df['phone'].mode()[0], inplace=True)

# get dummy variables for categorical data
df = pd.get_dummies(df, columns=['city', 'phone'], drop_first=True)

# finally we generate the retained column. pick the latest last_trip_date, subtract 30 days
# and find what fraction of users have a last_trip_date greater than that date
thirty_days_date = df['last_trip_date'].max() - timedelta(days=30)
retained_frac = float(len(df[df['last_trip_date'] > thirty_days_date])) / float(len(df))
print 'fraction retained = %.2f percent' % (retained_frac * 100)

df['retained'] = (df['last_trip_date'] > thirty_days_date) * 1

# PREDICTIVE MODELING

# drop datetime features because they cannot be used for modeling
df.drop(['signup_date', 'last_trip_date'], axis=1, inplace=True)

# separate dependent from independent variables
X = df.drop('retained', axis=1)
y = df['retained']

# do test/train split of 20/80
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# use a random forest classifier and 5-fold cross validation to determine the performance of the model
clf = RandomForestClassifier()
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print cv_scores
print "Average 5-Fold CV Score: %.2f" % (np.mean(cv_scores))

# Define function to get metrics of the model

def get_metrics(true_labels, predicted_labels):
    print ('Accuracy: ', accuracy_score(true_labels, predicted_labels))
    print (classification_report(true_labels, predicted_labels))
    return None

rfc = RandomForestClassifier()
# build model
rfc.fit(X_train, y_train)
# predict using model
y_predict = rfc.predict(X_test)

print ('Test set performance:')
get_metrics(true_labels=y_test, predicted_labels=y_predict)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_predict)
pd.DataFrame(cm, index=range(0, 2), columns=range(0, 2))

# Compute predicted probabilities
y_pred_prob = rfc.predict_proba(X_test)[:, 1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig(DIR + 'ROC_curve.png', dpi=350)

# Compute and print AUC score
print("AUC: {:.4f}".format(roc_auc_score(y_test, y_pred_prob)))

# get feature importances
fi = pd.DataFrame(list(zip(X.columns, rfc.feature_importances_)), columns=['features', 'Importance'])
fi.sort_values(by='Importance', ascending=False).head(5)
