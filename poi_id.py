#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from time import time


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### features_list = ['poi','salary'] # You will need to use more features

##features_list = ['poi','salary','deferral_payments','expenses','deferred_income','long_term_incentive',
##                  'restricted_stock_deferred','loan_advances','director_fees','bonus','other',
##                  'total_stock_value','restricted_stock','total_payments','exercised_stock_options',
##                  'shared_receipt_with_poi','to_poi_ratio','from_poi_ratio']

features_list = ['poi', 'bonus', 'expenses', 'exercised_stock_options', 'to_poi_ratio', 'restricted_stock', 'deferred_income'] # selected sum_feature_scores >0.7



### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

print "Total number of data points (people):", len(data_dict.keys())
print "Number of features:", (len(data_dict['METTS MARK'])) 
print data_dict['METTS MARK']

### count POIs
i = 0
for key in data_dict:
    if data_dict[key]['poi'] == True:
        i = i + 1
        # print key 
print "Number of POIs:", i



### Task 2: Remove outliers
### plot features

features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()

### check outliers

top10 = []
for key in data_dict:
    if data_dict[key]['bonus'] != 'NaN':
        top10.append((key,(data_dict[key]['bonus'])))
        
print "Top 10 bonus:", (sorted (top10, key = lambda x:x[1], reverse = True)[:10])



### Task 3: Create new feature(s)
### new features: from_poi_ratio; to_poi_ratio

for key in data_dict:
    if data_dict[key]['from_poi_to_this_person'] != 'NaN' and data_dict[key]['from_messages'] != 'NaN':
        from_ratio = float (data_dict[key]['from_poi_to_this_person'])/float(data_dict[key]['from_messages'])
    else:
        from_ratio = float(0.)
    data_dict[key]['from_poi_ratio'] = from_ratio
        

for key in data_dict:       
    if data_dict[key]['from_this_person_to_poi'] != 'NaN' and data_dict[key]['to_messages'] != 'NaN':
        to_ratio = float(data_dict[key]['from_this_person_to_poi'])/float(data_dict[key]['to_messages'])
    else:
        to_ratio = float(0.)
    data_dict[key]['to_poi_ratio'] = to_ratio


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### feature selection using Decision Tree Classifier with KFold validation
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

acc = []
precision = []
recall = []
feature_scores = []

t0 = time()
k = KFold(len(labels), 10)
for train_index, test_index in k:
    features_train = [features[ii] for ii in train_index]
    features_test= [features[ii] for ii in test_index]
    labels_train = [labels[ii] for ii in train_index]
    labels_test= [labels[ii] for ii in test_index]

    DTclf = DecisionTreeClassifier(random_state = 11)
    DTclf.fit(features_train, labels_train)
    pred = DTclf.predict(features_test)
    acc.append(accuracy_score(labels_test, pred))
    precision.append(precision_score(labels_test, pred))
    recall.append(recall_score(labels_test, pred))
    feature_scores.append(DTclf.feature_importances_)
   
print "Time:", round(time()-t0, 3), "s"
print "Accuracy:", acc, np.mean(acc)
print "Precision:", precision, np.mean(precision)
print "Recall:", recall, np.mean(recall)

sum_feature_scores = [sum(x) for x in zip(*feature_scores)]
print "10-fold sum feature scores:", sum_feature_scores
for i in range (len(features_list)-1):
    print (i+1, features_list[i+1], sum_feature_scores[i])



### Try out the Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier

acc = []
precision = []
recall = []

t0 = time()
k = KFold(len(labels), 10)
for train_index, test_index in k:
    features_train = [features[ii] for ii in train_index]
    features_test= [features[ii] for ii in test_index]
    labels_train = [labels[ii] for ii in train_index]
    labels_test= [labels[ii] for ii in test_index]
    
    ABclf = AdaBoostClassifier(n_estimators = 100, random_state = 11)
    ABclf.fit(features_train, labels_train)
    pred = ABclf.predict(features_test)
    acc.append(accuracy_score(labels_test, pred))
    precision.append(precision_score(labels_test, pred))
    recall.append(recall_score(labels_test, pred))

print "Time for Adaboost:", round(time()-t0, 3), "s"
print "Accuracy for Adaboost:", acc, np.mean(acc)
print "Precision for Adaboost:", precision, np.mean(precision)
print "Recall for Adaboost:", recall, np.mean(recall)


### Try out the GaussianNB
from sklearn.naive_bayes import GaussianNB

acc = []
precision = []
recall = []

t0 = time()
k = KFold(len(labels), 10)
for train_index, test_index in k:
    features_train = [features[ii] for ii in train_index]
    features_test= [features[ii] for ii in test_index]
    labels_train = [labels[ii] for ii in train_index]
    labels_test= [labels[ii] for ii in test_index]
    
    NBclf = GaussianNB()
    NBclf. fit(features_train, labels_train)
    pred = NBclf.predict(features_test)
    acc.append(accuracy_score(labels_test, pred))
    precision.append(precision_score(labels_test, pred))
    recall.append(recall_score(labels_test, pred))
    
print "Time for GaussianNB:", round(time()-t0, 3), "s"
print "Accuracy for GaussianNB:", acc, np.mean(acc)
print "Precision for GaussianNB:", precision, np.mean(precision)
print "Recall for GaussianNB:", recall, np.mean(recall)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.grid_search import GridSearchCV

### Tune Decision Tree Classifier
parameters = {"min_samples_split": [2, 3, 4, 5, 6, 7, 8],
              "criterion": ["gini", "entropy"],
              "splitter": ["best", "random"],
              }
scores = ['precision', 'recall']

for score in scores:
    print ("Tuning hyper-parameters for", score)    
    tree = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "auto", max_depth = None)
    DTclf = GridSearchCV(tree, param_grid = parameters, cv=5, scoring = score)
    DTclf.fit(features, labels)
    print "The best parameters for decision tree:"
    print (DTclf.best_params_)


### Tune Adaboost Classifier
parameters = {"n_estimators": [1, 10, 50, 100, 200],
              }
scores = ['precision', 'recall']
for score in scores:
    print ("Tuning hyper-parameters for", score)
    tree = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "auto", max_depth = None)
    Adaboost_tuned = AdaBoostClassifier(base_estimator = tree, random_state = 11)
    ABclf = GridSearchCV(Adaboost_tuned, param_grid = parameters, cv=5, scoring = score)
    ABclf.fit(features, labels)
    print "The best parameters for Adaboost:"
    print (ABclf.best_params_)
  


### The Final Classifier

### Split data into training and testing sets

# from sklearn import cross_validation
# features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = 0.1, random_state=42)

from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(labels, n_iter = 5, test_size = 0.1, random_state = 0)

acc = []
precision = []
recall = []

t0 = time()
for train_index, test_index in sss:
    features_train = [features[ii] for ii in train_index]
    features_test= [features[ii] for ii in test_index]
    labels_train = [labels[ii] for ii in train_index]
    labels_test= [labels[ii] for ii in test_index]

    ### Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "auto", max_depth = None, min_samples_split = 3, splitter = 'random', criterion = 'gini')

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc.append(accuracy_score(labels_test, pred))
    precision.append(precision_score(labels_test, pred))
    recall.append(recall_score(labels_test, pred))

print "Final model - Accuracy:", acc, np.mean(acc)
print "Final model - Precision:", precision, np.mean(precision)
print "Final model - Recall:", recall, np.mean(recall) 



test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)

pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(my_dataset, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

