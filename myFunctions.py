from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats.stats import pearsonr
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

def crossValidation(X, y, count, kneigbors = 3, metric = 'euclidean'):
    scores = []
    for i in range(0, count):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        knn = KNeighborsClassifier(n_neighbors=kneigbors)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        score1 = accuracy_score(y_test, pred)
        scores.append(score1)

        knn = KNeighborsClassifier(n_neighbors=kneigbors)
        knn.fit(X_test, y_test)
        pred = knn.predict(X_train)
        score2 = accuracy_score(y_train, pred)
        scores.append(score2)

    return sum(scores)/len(scores)