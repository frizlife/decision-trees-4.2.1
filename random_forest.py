import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

colnames = pd.read_csv('features_x_labels.txt', sep=" ", skipinitialspace=True, names = ['a', 'b'])
colnames = colnames.drop('a', 1)
train = pd.read_csv('X_train_x_labels.txt', sep=" ", skipinitialspace = True, header=None)
validate = pd.read_csv('X_validate_x_labels.txt', sep=" ", skipinitialspace = True, header=None)
test = pd.read_csv('X_test_x_labels.txt', sep=" ", skipinitialspace = True, header=None)
for x in colnames:
    test.columns = colnames[x]
    train.columns = colnames[x]
    validate.columns = colnames[x]

y_train = pd.read_csv('Y_train_x_labels.txt', names=['activity'])
y_test = pd.read_csv('y_test_x_labels.txt', names=['activity'])
y_validate = pd.read_csv('y_validate_x_labels.txt', names=['activity'])

#map activity to a categorical variable
y_train['activity'] = y_train['activity'].astype('category')
y_test['activity'] = y_test['activity'].astype('category')
y_validate['activity'] = y_validate['activity'].astype('category')
yy=y_train.values

#RF classifier
clf = RandomForestClassifier(n_estimators=50, oob_score=True)
clf.fit(train, yy.ravel())
output = clf.predict(test) 

print""
print "Out of Bag Score"
print clf.oob_score_
print ""
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), test.columns), reverse=True)
#http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
print ""
print "Mean Accuracy Score test set:"
mast = clf.score(test, y_test)
print mast
print ""
print "Mean Accuracy Score validation set:"
masv = clf.score(validate, y_validate)
print masv
print ""
print "Recall score on test set:"
#y_train_np = y_train.to_records()
recall = recall_score(y_test, output)
print recall

print ""
print "Confusion matrix"
cm = confusion_matrix(y_test, output)
print cm

print""
print"F1_score"
f = f1_score(y_test, output)
print f

