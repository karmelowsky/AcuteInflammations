from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.neighbors import NearestCentroid

def cv2NN(X_train, X_test, y_train, y_test, kneighbors, metric ='euclidean', scalling = False):

    trainDataX = X_train
    if scalling:
        trainDataX = preprocessing.scale(X_train)

    knn = KNeighborsClassifier(n_neighbors=kneighbors, metric=metric)
    knn.fit(trainDataX, y_train)
    pred = knn.predict(X_test)
    score1 = accuracy_score(y_test, pred)

    trainDataX = X_test
    if scalling:
        trainDataX = preprocessing.scale(X_test)

    knn = KNeighborsClassifier(n_neighbors=kneighbors, metric=metric)
    knn.fit(trainDataX, y_test)
    pred = knn.predict(X_train)
    score2 = accuracy_score(y_train, pred)

    return (score2+score1)/2

def cv2NM(X_train, X_test, y_train, y_test, metric = 'euclidean', scalling = False):

    trainDataX = X_train
    if scalling:
        trainDataX = preprocessing.scale(X_train)

    nm = NearestCentroid(metric=metric)
    nm.fit(trainDataX, y_train)
    pred = nm.predict(X_test)
    score1 = accuracy_score(y_test, pred)

    trainDataX = X_test
    if scalling:
        trainDataX = preprocessing.scale(X_test)

    nm = NearestCentroid(metric=metric)
    nm.fit(trainDataX, y_test)
    pred = nm.predict(X_train)
    score2 = accuracy_score(y_train, pred)
    return (score2 + score1) / 2

