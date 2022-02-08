import time
start_time = time.time()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, roc_auc_score, precision_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=25000)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

algorithm = RandomForestClassifier().fit(x_train, y_train)

algorithm_pred = algorithm.predict(x_test)

#metrics
print('accuracy:')
print(accuracy_score(y_test, algorithm_pred),"\n")

print('f1 score:')
print(f1_score(y_test, algorithm_pred),"\n")

print('precision:')
print(precision_score(y_test, algorithm_pred),"\n")

print('recall_score:')
print(recall_score(y_test, algorithm_pred),"\n")

print('roc_auc_score:')
print(roc_auc_score(y_test, algorithm_pred),"\n")

print("--- %s seconds ---" % (time.time() - start_time))