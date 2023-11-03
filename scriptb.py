#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from scipy.stats import loguniform

# Load the data from “reduceddataset.csv” into a data frame object.
samples = pd.read_csv('reduceddataset.csv')

# Add the column “label” to hold the binary label (1 for malware and 0 baningware).
samples['target'] = np.where(samples['label'] == 'benignware', 0, 1)

# Create feature data frame without label information [ drop the md5hash, label and target columns].
features = samples.drop(['MD5', 'label', 'target'], axis=1)

# Select the features using SelectKBest with chi2 and k=15.
selector = SelectKBest(chi2, k=15)
selector.fit(features, samples['label'])

# Get the list of selected features and print.
selected_features = features.columns[selector.get_support(indices=True)].tolist()
print(selected_features)

# Use the model we made earlier(saved_detector.pkl).
model = SVC(kernel='linear', C=1.0, tol=1e-5, probability=True)

# Set the values for x_train and y_train.
x_train = samples[selected_features]
y_train = samples['target']

# Create cross validation object.
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define space search.
space = dict()
space['kernel'] = ["linear", "poly", "rbf", "sigmoid"]
space['C'] = loguniform(1e-5, 100)
space['tol'] = loguniform(1e-5, 100)

# Define search.
search = RandomizedSearchCV(model, space, n_iter=500, scoring='recall', cv=kfold)

# Execute search to fetch the results.
search.fit(x_train, y_train)

# Print the results, best scores and best parameters.
print("Best score: %0.3f" % search.best_score_)
print("Best parameters set:")
best_parameters = search.best_estimator_.get_params()
for param_name in sorted(space.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
