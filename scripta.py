#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load the data from “reduceddataset.csv” into a data frame object.
samples = pd.read_csv('reduceddataset.csv')

# Add the column “label” to hold the binary label (1 for malware and 0 baningware).
samples['target'] = np.where(samples['label'] == 'benignware', 0, 1)

# Create feature data frame without label information [ drop the md5hash, label and target columns].
features = samples.drop(['MD5', 'label', 'target'], axis=1)

# Select the features using SelectKBest with chi2 and k=15.
selector = SelectKBest(chi2, k=15)
selector.fit(features, samples['target'])

# Get the list of selected features and print.
selected_features = features.columns[selector.get_support(indices=True)].tolist()
print(selected_features)

# Use the model we made earlier(saved_detector.pkl).
model = SVC(kernel='linear', C=1, probability=True)

# Set the values for x_train and y_train.
x_train = samples[selected_features]
y_train = samples['target']

# Create cross validation object.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define space search. Use the following values for kernel:[linear, poly, rbf and sigmoid].
param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [100, 10, 1.0, 0.1, 0.001]}

# Calculate and print the cross-validation score for each scenario (4 X 5:Total 20).
for kernel in param_grid['kernel']:
    for c in param_grid['C']:
        model.set_params(kernel=kernel, C=c)
        scores = cross_val_score(model, x_train, y_train, cv=cv)
        print(f"Kernel: {kernel}, C: {c}, Scores: {scores}")
        print(f"Mean: {scores.mean()}, Standard Deviation: {scores.std()}")
