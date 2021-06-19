import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report

dataset = pd.read_csv('BrainTumor.csv')

print(dataset['2'].value_counts())

dataset['2'].replace(np.nan, 'b', inplace=True)

mappings = list()
encoder = LabelEncoder()
for column in range(len(dataset.columns)):
    dataset[dataset.columns[column]] = encoder.fit_transform(dataset[dataset.columns[column]])
    mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
    mappings.append(mappings_dict)

X = dataset.drop('class', axis=1)
y = dataset['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3, random_state=42)
DTC = DecisionTreeClassifier(random_state=42)
DTC.fit(X_train, y_train)
predDTC = DTC.predict(X_test)
reportDTC = classification_report(y_test, predDTC, output_dict=True)
crDTC = pd.DataFrame(reportDTC).transpose()
print(crDTC)

fig = plt.figure(figsize=(100, 80))
plot = plot_tree(DTC, feature_names=list(dataset.columns), class_names=['0', '1'], filled=True)
for i in plot:
    arrow = i.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('black')
        arrow.set_linewidth(2)
